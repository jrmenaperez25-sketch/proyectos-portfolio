#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import parches_tupac as P

PRED_THRESHOLD = 0.05
DEDUP_IOU = 0.30
MATCH_IOU = 0.50
MATCH_MODE = 'iou'
MATCH_DIST_UM = 7.5
DIST_PX = MATCH_DIST_UM / P.MPP

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

def casa(pb, gb):
    if MATCH_MODE == 'iou':
        return iou(pb[:4], gb) >= MATCH_IOU
    return dist_centros(pb, gb) <= DIST_PX

def mejor_gt(pb, gts, usado):
    best, bestq = -1, None
    for gi, gb in enumerate(gts):
        if usado[gi] or not casa(pb, gb):
            continue
        q = iou(pb[:4], gb) if MATCH_MODE == 'iou' else -dist_centros(pb, gb)
        if bestq is None or q > bestq:
            best, bestq = gi, q
    return best

def dedup(boxes):
    keep = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if MATCH_MODE == 'iou':
            ok = all(iou(box[:4], k[:4]) < DEDUP_IOU for k in keep)
        else:
            ok = all(dist_centros(box, k) > DIST_PX for k in keep)
        if ok:
            keep.append(box)
    return keep

def predecir(modelo, ckpt, image_paths):
    out = defaultdict(list)
    if modelo == 'rfdetr':
        import rfdetr
        model = rfdetr.RFDETRSmall(pretrain_weights=str(ckpt))
        for i in range(0, len(image_paths), 8):
            batch = image_paths[i:i + 8]
            res = model.predict([str(p) for p in batch], threshold=PRED_THRESHOLD)
            if not isinstance(res, list):
                res = [res]
            for p, det in zip(batch, res):
                if det is None:
                    continue
                for box, conf in zip(det.xyxy, det.confidence):
                    out[p.name].append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))
    else:
        from ultralytics import YOLO
        model = YOLO(str(ckpt))
        for i in range(0, len(image_paths), 64):
            batch = [str(p) for p in image_paths[i:i + 64]]
            for p, r in zip(batch, model.predict(batch, conf=PRED_THRESHOLD, imgsz=512, verbose=False)):
                b = r.boxes
                if b is None:
                    continue
                for box, conf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                    out[Path(p).name].append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))
    return out

def predecir_frames(modelo, ckpt, overlap):
    half = P.PATCH_SIZE / 2; bbox_r = P.BBOX_RADIUS_PX
    version = f'sliding{overlap}'
    pred = defaultdict(list); gt = {}; notb = {}
    for caso in P.SPLIT['test']:
        vdir = P.VERS / caso / version
        man = {r['patch_filename']: r for r in csv.DictReader(open(vdir / '_manifest.csv'))}
        imgs = sorted((vdir / 'images').glob('*.jpg'))
        print(f'  predic caso {caso} ov{overlap}: {len(imgs)} teselas', flush=True)
        preds = predecir(modelo, ckpt, imgs)
        for fname, boxes in preds.items():
            m = man.get(fname)
            if not m:
                continue
            ox = float(m['patch_center_x']) - half; oy = float(m['patch_center_y']) - half
            for (x1, y1, x2, y2, c) in boxes:
                pred[m['frame_stem']].append((ox + x1, oy + y1, ox + x2, oy + y2, c))
        for fr in P.load_case_frames(caso):
            gt[fr['frame_stem']] = [(x - bbox_r, y - bbox_r, x + bbox_r, y + bbox_r) for (x, y, c, mid) in fr['mit_pts']]
            notb[fr['frame_stem']] = [(x - bbox_r, y - bbox_r, x + bbox_r, y + bbox_r) for (x, y, c) in fr['not_pts']]
    return pred, gt, notb

def metricas(pred_raw, gt, notb, crit, salida, nombre):
    global MATCH_MODE
    MATCH_MODE = crit
    pred = {k: dedup(list(v)) for k, v in pred_raw.items()}
    filas_pred = []
    for key, preds in pred.items():
        gts = gt.get(key, []); usado = [False] * len(gts); nots = notb.get(key, [])
        for pb in sorted(preds, key=lambda b: -b[4]):
            mejor = mejor_gt(pb, gts, usado)
            if mejor >= 0:
                usado[mejor] = True; cat = 'tp'
            elif any(casa(pb, nb) for nb in nots):
                cat = 'fp_hardneg'
            else:
                cat = 'fp_empty'
            filas_pred.append({'score': round(pb[4], 4), 'categoria': cat})
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / f'{nombre}_preds.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['score', 'categoria']); w.writeheader(); w.writerows(filas_pred)
    n_gt = sum(len(v) for v in gt.values())
    filas = []
    for thr in [round(0.05 * i, 2) for i in range(1, 19)]:
        TP = FP = 0; matched = 0
        for key, gts in gt.items():
            preds = sorted([b for b in pred.get(key, []) if b[4] >= thr], key=lambda b: -b[4])
            usado = [False] * len(gts)
            for pb in preds:
                mejor = mejor_gt(pb, gts, usado)
                if mejor >= 0:
                    usado[mejor] = True; TP += 1
                else:
                    FP += 1
            matched += sum(usado)
        FN = n_gt - matched
        Pr = TP / (TP + FP) if TP + FP else 0.0; Re = TP / (TP + FN) if TP + FN else 0.0
        F1 = 2 * Pr * Re / (Pr + Re) if Pr + Re else 0.0
        filas.append({'conf': thr, 'TP': TP, 'FP': FP, 'FN': FN,
                      'precision': round(Pr, 4), 'recall': round(Re, 4), 'f1': round(F1, 4)})
    with open(salida / f'{nombre}_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    f030 = min(filas, key=lambda r: abs(r['conf'] - 0.30)); fbest = max(filas, key=lambda r: r['f1'])
    (salida / f'{nombre}_resumen.json').write_text(json.dumps(
        {'nombre': nombre, 'crit': crit, 'n_gt': n_gt, 'conf_0.30': f030, 'best_f1': fbest}, indent=2))
    print(f'  [{nombre}] n_gt={n_gt} | @0.30 F1={f030["f1"]} (R{f030["recall"]}) | best F1={fbest["f1"]}@{fbest["conf"]}', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelo', required=True, choices=['rfdetr', 'yolo'])
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--salida', type=Path, required=True)
    ap.add_argument('--nombre', required=True)
    args = ap.parse_args()
    print(f'EVAL TUPAC {args.nombre} ({args.modelo})', flush=True)
    for overlap in ['20', '50']:
        pred_raw, gt, notb = predecir_frames(args.modelo, args.ckpt, overlap)
        for crit in ['distancia', 'iou']:
            metricas(pred_raw, gt, notb, crit, args.salida, f'{args.nombre}_ov{overlap}_{crit}')

if __name__ == '__main__':
    main()
