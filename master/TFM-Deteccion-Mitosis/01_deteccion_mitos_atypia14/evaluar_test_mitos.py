#!/usr/bin/env python3
from __future__ import annotations

import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import parches_mitos as P

TEST_ROOT = Path(f'{WORKSPACE}/Test_A04')
TEST_PACIENTES = [('aperio', 'A04'), ('hamamatsu', 'H04')]
PRED_THRESHOLD = 0.05

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def centro(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def dist_centros(a, b):
    ca, cb = centro(a), centro(b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5

def casa(pb, gb, sc):
    if MATCH_MODE == 'iou':
        return iou(pb[:4], gb) >= MATCH_IOU
    return dist_centros(pb, gb) <= MATCH_DIST_PX[sc]

def mejor_gt(pb, gts, usado, sc):
    best, bestq = -1, None
    for gi, gb in enumerate(gts):
        if usado[gi] or not casa(pb, gb, sc):
            continue
        q = iou(pb[:4], gb) if MATCH_MODE == 'iou' else -dist_centros(pb, gb)
        if bestq is None or q > bestq:
            best, bestq = gi, q
    return best

def dedup(boxes, sc):
    keep = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if MATCH_MODE == 'iou':
            ok = all(iou(box[:4], k[:4]) < DEDUP_IOU for k in keep)
        else:
            ok = all(dist_centros(box, k) > MATCH_DIST_PX[sc] for k in keep)
        if ok:
            keep.append(box)
    return keep

def predecir(modelo, ckpt, image_paths):
    out = defaultdict(list)
    if modelo == 'rfdetr':
        import rfdetr
        model = rfdetr.RFDETRSmall(pretrain_weights=str(ckpt))
        bs = 8
        for i in range(0, len(image_paths), bs):
            batch = image_paths[i:i + bs]
            res = model.predict([str(p) for p in batch], threshold=PRED_THRESHOLD)
            if not isinstance(res, list):
                res = [res]
            for p, det in zip(batch, res):
                if det is None:
                    continue
                for box, conf in zip(det.xyxy, det.confidence):
                    out[p.name].append((float(box[0]), float(box[1]),
                                        float(box[2]), float(box[3]), float(conf)))
    else:
        from ultralytics import YOLO
        model = YOLO(str(ckpt))
        bs = 64
        for i in range(0, len(image_paths), bs):
            batch = [str(p) for p in image_paths[i:i + bs]]
            for p, r in zip(batch, model.predict(batch, conf=PRED_THRESHOLD,
                                                 imgsz=512, verbose=False)):
                name = Path(p).name
                b = r.boxes
                if b is None:
                    continue
                xyxy = b.xyxy.cpu().numpy(); confs = b.conf.cpu().numpy()
                for box, conf in zip(xyxy, confs):
                    out[name].append((float(box[0]), float(box[1]),
                                      float(box[2]), float(box[3]), float(conf)))
    return out

def evaluar(modelo, ckpt, salida, nombre):
    half = P.PATCH_SIZE / 2
    global MATCH_DIST_PX
    MATCH_DIST_PX = {sc: MATCH_DIST_UM / mpp for sc, mpp in P.MPP_BY_SCANNER.items()}
    if MATCH_MODE == 'distancia':
        print(f'  matching por DISTANCIA {MATCH_DIST_UM} µm = '
              f'{ {sc: round(d,1) for sc,d in MATCH_DIST_PX.items()} } px')
    pred_por_frame = defaultdict(list)
    gt_por_frame = {}
    not_por_frame = {}

    for sc, pac in TEST_PACIENTES:
        bbox_r = P.bbox_radius_for(sc)
        vdir = TEST_ROOT / 'mitos14' / sc / pac / 'sliding_window'
        man = {r['patch_filename']: r for r in csv.DictReader(open(vdir / '_manifest.csv'))}
        image_paths = sorted((vdir / 'images').glob('*.jpg'))
        print(f'  prediciendo {sc}/{pac}: {len(image_paths)} teselas…')
        preds = predecir(modelo, ckpt, image_paths)
        for fname, boxes in preds.items():
            m = man.get(fname)
            if not m:
                continue
            ox = float(m['patch_center_x']) - half
            oy = float(m['patch_center_y']) - half
            frame = m['frame_stem']
            for (x1, y1, x2, y2, c) in boxes:
                pred_por_frame[(sc, frame)].append((ox + x1, oy + y1, ox + x2, oy + y2, c))
        for fr in P.load_patient_frames(sc, pac):
            gt_por_frame[(sc, fr['frame_stem'])] = [
                (wx - bbox_r, wy - bbox_r, wx + bbox_r, wy + bbox_r)
                for (wx, wy, conf, mid) in fr['mit_pts']]
            not_por_frame[(sc, fr['frame_stem'])] = [
                (wx - bbox_r, wy - bbox_r, wx + bbox_r, wy + bbox_r)
                for (wx, wy, conf) in fr['not_pts']]

    for k in pred_por_frame:
        pred_por_frame[k] = dedup(pred_por_frame[k], k[0])

    filas_pred = []
    for key, preds in pred_por_frame.items():
        sc = key[0]
        gts = gt_por_frame.get(key, []); usado = [False] * len(gts)
        nots = not_por_frame.get(key, [])
        for pb in sorted(preds, key=lambda b: -b[4]):
            mejor = mejor_gt(pb, gts, usado, sc)
            if mejor >= 0:
                usado[mejor] = True; cat = 'tp'
            elif any(casa(pb, nb, sc) for nb in nots):
                cat = 'fp_hardneg'
            else:
                cat = 'fp_empty'
            filas_pred.append({'score': round(pb[4], 4), 'categoria': cat})
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / f'{nombre}_preds.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['score', 'categoria']); w.writeheader(); w.writerows(filas_pred)

    n_gt = sum(len(v) for v in gt_por_frame.values())
    filas = []
    for thr in [round(0.05 * i, 2) for i in range(1, 19)]:
        TP = FP = 0
        matched_gt = 0
        for key, gts in gt_por_frame.items():
            sc = key[0]
            preds = sorted([b for b in pred_por_frame.get(key, []) if b[4] >= thr],
                           key=lambda b: -b[4])
            usado = [False] * len(gts)
            for pb in preds:
                mejor = mejor_gt(pb, gts, usado, sc)
                if mejor >= 0:
                    usado[mejor] = True; TP += 1
                else:
                    FP += 1
            matched_gt += sum(usado)
        FN = n_gt - matched_gt
        P_ = TP / (TP + FP) if TP + FP else 0.0
        R_ = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0.0
        filas.append({'conf': thr, 'TP': TP, 'FP': FP, 'FN': FN,
                      'precision': round(P_, 4), 'recall': round(R_, 4), 'f1': round(F1, 4)})

    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / f'{nombre}_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)

    f030 = min(filas, key=lambda r: abs(r['conf'] - 0.30))
    fbest = max(filas, key=lambda r: r['f1'])
    resumen = {'modelo': modelo, 'nombre': nombre, 'n_gt': n_gt,
               'conf_0.30': f030, 'best_f1': fbest}
    (salida / f'{nombre}_resumen.json').write_text(json.dumps(resumen, indent=2))
    print(f'  GT mitosis (A04+H04): {n_gt}')
    print(f'  @conf0.30 -> P={f030["precision"]} R={f030["recall"]} F1={f030["f1"]} '
          f'(TP{f030["TP"]} FP{f030["FP"]} FN{f030["FN"]})')
    print(f'  best F1   -> F1={fbest["f1"]} @conf={fbest["conf"]} '
          f'(P={fbest["precision"]} R={fbest["recall"]})')
    return resumen

DEDUP_IOU = 0.30
MATCH_IOU = 0.50
MATCH_MODE = 'iou'
MATCH_DIST_UM = 7.5
MATCH_DIST_PX = {}

def main():
    global DEDUP_IOU, MATCH_IOU, TEST_ROOT, MATCH_MODE, MATCH_DIST_UM
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelo', required=True, choices=['rfdetr', 'yolo'])
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--salida', type=Path, required=True)
    ap.add_argument('--nombre', default=None, help='nombre del run (default = nombre del ckpt)')
    ap.add_argument('--dedup-iou', type=float, default=0.30)
    ap.add_argument('--match-iou', type=float, default=0.50)
    ap.add_argument('--match-mode', choices=['iou', 'distancia'], default='iou',
                    help="'iou' (IoU>=match-iou) o 'distancia' (centroide <=match-dist-um, estándar MIDOG)")
    ap.add_argument('--match-dist-um', type=float, default=7.5,
                    help='tolerancia de centroide en µm (modo distancia; estándar 7.5)')
    ap.add_argument('--test-root', type=Path, default=TEST_ROOT,
                    help=f'raíz del test sliding-window (default {WORKSPACE}/Test_A04)')
    args = ap.parse_args()
    DEDUP_IOU, MATCH_IOU = args.dedup_iou, args.match_iou
    MATCH_MODE, MATCH_DIST_UM = args.match_mode, args.match_dist_um
    TEST_ROOT = args.test_root
    nombre = args.nombre or args.ckpt.parent.name
    print(f'EVAL A04 | modelo={args.modelo} | ckpt={args.ckpt} | nombre={nombre}')
    evaluar(args.modelo, args.ckpt, args.salida, nombre)

if __name__ == '__main__':
    main()
