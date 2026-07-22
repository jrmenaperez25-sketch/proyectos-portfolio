#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import sys, csv, math
from collections import defaultdict
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_REPO)):
    _p = os.path.join(_REPO, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import virchow_nucleo as XC
import parches_tupac as PT

THR = 0.85
clf, proc = XC.load_clf(f'{WORKSPACE}/segunda_fase_tupac/clasificadores/lora_model.pt')
TEST = PT.SPLIT['test']; r = PT.BBOX_RADIUS_PX
gt = {}; finfo = {}
for caso in TEST:
    for fr in PT.load_case_frames(caso):
        gt[fr['frame_stem']] = [(x-r, y-r, x+r, y+r) for (x, y, c, m) in fr['mit_pts']]
        finfo[fr['frame_stem']] = (str(fr['img_path']), 0.25)
cands = defaultdict(list)
for row in csv.DictReader(open(f'{WORKSPACE}/segunda_fase_tupac/pool/candidatos.csv')):
    cands[row['frame']].append({'cx': float(row['cx']), 'cy': float(row['cy']), 'score': float(row['score']),
                                'box': (float(row['x1']), float(row['y1']), float(row['x2']), float(row['y2']))})
print(f'frames {len(gt)}, candidatos {sum(len(v) for v in cands.values())}', flush=True)
XC.añadir_probs(clf, proc, cands, finfo)
DP = {k: 30.0 for k in gt}

def prf(tp, fp, fn):
    P = tp/(tp+fp) if tp+fp else 0.; R = tp/(tp+fn) if tp+fn else 0.
    return round(P, 4), round(R, 4), round(2*P*R/(P+R), 4) if P+R else 0.
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1); inter = iw*ih
    if inter <= 0: return 0.
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.
ngt = sum(len(v) for v in gt.values())
def cuenta(crit, conf, casc):
    tp = fp = 0
    for k, gts in gt.items():
        d = DP[k]; used = [False]*len(gts)
        cs = [c for c in cands.get(k, []) if c['score'] >= conf and (not casc or c.get('prob', 1) >= THR)]
        for c in sorted(cs, key=lambda z: -z['score']):
            best, bq = -1, None
            for gi, g in enumerate(gts):
                if used[gi]: continue
                if crit == 'iou': q = _iou(c['box'], g); ok = q >= 0.5
                else:
                    q = -math.hypot(c['cx']-(g[0]+g[2])/2, c['cy']-(g[1]+g[3])/2); ok = -q <= d
                if ok and (bq is None or q > bq): best, bq = gi, q
            if best >= 0: used[best] = True; tp += 1
            else: fp += 1
    return prf(tp, fp, ngt-tp)
print(f'\nTUPAC TEST end-to-end, umbral FIJO {THR} ({ngt} mitosis):')
for crit in ['dist', 'iou']:
    for conf in [0.30, 0.50]:
        b = cuenta(crit, conf, False); c = cuenta(crit, conf, True)
        print(f'  conf{conf} {crit}: detector {b[0]}/{b[1]}/{b[2]}  ->  dos fases {c[0]}/{c[1]}/{c[2]}', flush=True)
