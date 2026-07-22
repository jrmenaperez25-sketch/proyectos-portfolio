#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import csv, math, sys
from collections import defaultdict
from pathlib import Path

import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_REPO)):
    _p = os.path.join(_REPO, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import parches_mitos as PM
import parches_tupac as PT

OUT = Path(f'{WORKSPACE}/Cross_Scanner'); OUT.mkdir(exist_ok=True)
CONF = 0.30
FOLDS = ['03', '05', '06', '07', '09', '10', '11', '14', '16', '17']
TUPAC_APERIO = ['07', '11', '18']
TUPAC_LEICA = ['31', '38', '45', '52', '59', '66', '73']

CKPT = {
    'mitos_rfdetr': [('rfdetr', f'{WORKSPACE}/Runs_TFM/rfdetr/fold_{f}_leve_hsv/checkpoint_best_total.pth') for f in FOLDS],
    'mitos_yolo':   [('yolo',   f'{WORKSPACE}/Runs_TFM/yolo/fold_{f}_default/weights/best.pt') for f in FOLDS],
    'tupac_rfdetr': [('rfdetr', f'{WORKSPACE}/EXPERIMENTO_TUPAC73/run/rfdetr_ovs/checkpoint_best_total.pth')],
    'tupac_yolo':   [('yolo',   f'{WORKSPACE}/EXPERIMENTO_TUPAC73/run/yolo/yolo_ovs/weights/best.pt')],
}

_models = {}
def cargar(modelo, ckpt):
    if ckpt in _models:
        return _models[ckpt]
    if modelo == 'rfdetr':
        import rfdetr; m = rfdetr.RFDETRSmall(pretrain_weights=ckpt)
    else:
        from ultralytics import YOLO; m = YOLO(ckpt)
    _models.clear(); _models[ckpt] = (modelo, m)
    return modelo, m

def predecir(modelo, m, paths, bs=16):
    out = defaultdict(list)
    for i in range(0, len(paths), bs):
        batch = paths[i:i+bs]
        if modelo == 'rfdetr':
            res = m.predict([str(p) for p in batch], threshold=0.05)
            res = res if isinstance(res, list) else [res]
            for p, det in zip(batch, res):
                if det is None: continue
                for b, c in zip(det.xyxy, det.confidence):
                    out[p.name].append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
        else:
            for p, r in zip(batch, m.predict([str(x) for x in batch], conf=0.05, imgsz=512, verbose=False)):
                if r.boxes is None: continue
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    out[p.name].append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
    return out

def dist(a, b):
    return math.hypot((a[0]+a[2])/2-(b[0]+b[2])/2, (a[1]+a[3])/2-(b[1]+b[3])/2)

def dedup(boxes, dpx):
    keep = []
    for box in sorted(boxes, key=lambda b: -b[4]):
        if all(dist(box, k) > dpx for k in keep):
            keep.append(box)
    return keep

def evaluar_frames(pred_por_frame, gt_por_frame, dpx, conf):
    TP = FP = 0; NG = sum(len(v) for v in gt_por_frame.values())
    for frame, gts in gt_por_frame.items():
        preds = dedup([b for b in pred_por_frame.get(frame, []) if b[4] >= conf], dpx)
        preds = sorted(preds, key=lambda b: -b[4]); usado = [False]*len(gts)
        for pb in preds:
            best, bd = -1, dpx
            for gi, gb in enumerate(gts):
                if usado[gi]: continue
                d = dist(pb, gb)
                if d <= bd: bd = d; best = gi
            if best >= 0: usado[best] = True; TP += 1
            else: FP += 1
    FN = NG - TP
    P = TP/(TP+FP) if TP+FP else 0.0; R = TP/(TP+FN) if TP+FN else 0.0
    return {'TP': TP, 'FP': FP, 'FN': FN, 'P': round(P, 4), 'R': round(R, 4),
            'F1': round(2*P*R/(P+R), 4) if P+R else 0.0}

def tiles_mitos(sc, pac):
    vdir = Path(f'{WORKSPACE}/Test_A04_overlap20/mitos14/{sc}/{pac}/sliding_window')
    man = {r['patch_filename']: r for r in csv.DictReader(open(vdir / '_manifest.csv'))}
    return sorted((vdir / 'images').glob('*.jpg')), man

def gt_mitos(sc, pac):
    r = PM.bbox_radius_for(sc); out = {}
    for fr in PM.load_patient_frames(sc, pac):
        out[fr['frame_stem']] = [(x-r, y-r, x+r, y+r) for (x, y, c, i) in fr['mit_pts']]
    return out, 7.5 / PM.MPP_BY_SCANNER[sc]

def tiles_tupac(caso):
    vdir = PT.VERS / caso / 'sliding20'
    man = {r['patch_filename']: r for r in csv.DictReader(open(vdir / '_manifest.csv'))}
    return sorted((vdir / 'images').glob('*.jpg')), man

def gt_tupac(caso):
    r = PT.BBOX_RADIUS_PX; out = {}
    for fr in PT.load_case_frames(caso):
        out[fr['frame_stem']] = [(x-r, y-r, x+r, y+r) for (x, y, c, i) in fr['mit_pts']]
    return out, 7.5 / PT.MPP

def reproyectar(preds, man):
    porframe = defaultdict(list)
    for nm, bx in preds.items():
        m = man.get(nm)
        if not m: continue
        ox = float(m['patch_center_x']) - 256; oy = float(m['patch_center_y']) - 256
        for (x1, y1, x2, y2, c) in bx:
            porframe[m['frame_stem']].append((ox+x1, oy+y1, ox+x2, oy+y2, c))
    return porframe

def target_mitos():
    res = {}
    for sc, pac in [('aperio', 'A04'), ('hamamatsu', 'H04')]:
        tiles, man = tiles_mitos(sc, pac); gt, dpx = gt_mitos(sc, pac)
        res[('MITOS', sc)] = [(tiles, man, gt, dpx)]
    return res

def target_tupac():
    res = {('TUPAC', 'aperio'): [], ('TUPAC', 'leica'): []}
    for caso in TUPAC_APERIO + TUPAC_LEICA:
        sc = 'aperio' if caso in TUPAC_APERIO else 'leica'
        tiles, man = tiles_tupac(caso); gt, dpx = gt_tupac(caso)
        res[('TUPAC', sc)].append((tiles, man, gt, dpx))
    return res

def main():
    targets = {**target_mitos(), **target_tupac()}
    todos = [('MITOS', 'aperio'), ('MITOS', 'hamamatsu'), ('TUPAC', 'aperio'), ('TUPAC', 'leica')]
    plan = [('tupac_rfdetr', todos), ('tupac_yolo', todos), ('mitos_rfdetr', todos), ('mitos_yolo', todos)]
    filas = []
    seen = set()
    for grupo, tgts in plan:
        for (modelo, ckpt) in CKPT[grupo]:
            fold = ckpt.split('fold_')[1][:2] if 'fold_' in ckpt else 'unico'
            mod, m = cargar(modelo, ckpt)
            for tgt in tgts:
                key = (grupo, fold, tgt)
                if key in seen: continue
                seen.add(key)
                pred_all = defaultdict(list); gt_all = {}
                dpx = None
                for (tiles, man, gt, d) in targets[tgt]:
                    dpx = d
                    pr = reproyectar(predecir(mod, m, tiles), man)
                    for fr, bx in pr.items(): pred_all[fr].extend(bx)
                    gt_all.update(gt)
                met = evaluar_frames(pred_all, gt_all, dpx, CONF)
                indom = (grupo.startswith('mitos') and tgt[0] == 'MITOS') or (grupo.startswith('tupac') and tgt[0] == 'TUPAC')
                filas.append({'modelo': grupo, 'fold': fold, 'tipo': 'in-domain' if indom else 'CROSS',
                              'target_dataset': tgt[0], 'target_scanner': tgt[1], **met})
                print(f'  {grupo} fold{fold} -> {tgt[0]}/{tgt[1]} [{"in" if indom else "CROSS"}]: F1={met["F1"]} R={met["R"]} P={met["P"]}', flush=True)
    with open(OUT / 'deteccion_cross.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    print(f'GUARDADO {OUT/"deteccion_cross.csv"} ({len(filas)} filas)', flush=True)

if __name__ == '__main__':
    main()
