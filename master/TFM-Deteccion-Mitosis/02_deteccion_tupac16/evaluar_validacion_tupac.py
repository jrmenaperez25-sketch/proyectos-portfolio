#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import sys, json, math
from collections import defaultdict
from pathlib import Path
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_REPO)):
    _p = os.path.join(_REPO, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import evaluar_test_tupac as E

DIST_PX = 30.0
modelo, ckpt, salida, nombre = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
salida.mkdir(parents=True, exist_ok=True)
coco = json.loads(Path(f'{WORKSPACE}/EXPERIMENTO_TUPAC73/ds_ovs/valid/_annotations.coco.json').read_text())
gt = defaultdict(list)
for a in coco['annotations']:
    if a['category_id'] == 1:
        x, y, w, h = a['bbox']; gt[a['image_id']].append((x, y, x+w, y+h))
imgs = [Path(im['file_name']) for im in coco['images']]
preds = E.predecir(modelo, ckpt, imgs)

def cen(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def match(gts, ps, crit):
    used = [False]*len(gts)
    for pb in sorted(ps, key=lambda z: -z[4]):
        best = -1
        if crit == 'distancia':
            pc = cen(pb); bd = DIST_PX
            for gi, g in enumerate(gts):
                if used[gi]: continue
                d = math.hypot(pc[0]-(g[0]+g[2])/2, pc[1]-(g[1]+g[3])/2)
                if d <= bd: bd = d; best = gi
        else:
            bi = 0.5
            for gi, g in enumerate(gts):
                if used[gi]: continue
                i = E.iou(pb[:4], g)
                if i >= bi: bi = i; best = gi
        yield best, used

res = {'nombre': nombre, 'modelo': modelo}
for crit in ['distancia', 'iou']:
    sweep = []
    for thr in [round(0.05*i, 2) for i in range(1, 13)]:
        TP = FP = FN = 0
        for im in coco['images']:
            gts = gt[im['id']]; used = [False]*len(gts)
            ps = sorted([p for p in preds.get(im['file_name'].split('/')[-1], preds.get(Path(im['file_name']).name, [])) if p[4] >= thr], key=lambda z: -z[4])
            for pb in ps:
                best = -1
                if crit == 'distancia':
                    pc = cen(pb); bd = DIST_PX
                    for gi, g in enumerate(gts):
                        if used[gi]: continue
                        d = math.hypot(pc[0]-(g[0]+g[2])/2, pc[1]-(g[1]+g[3])/2)
                        if d <= bd: bd = d; best = gi
                else:
                    bi = 0.5
                    for gi, g in enumerate(gts):
                        if used[gi]: continue
                        i = E.iou(pb[:4], g)
                        if i >= bi: bi = i; best = gi
                if best >= 0: used[best] = True; TP += 1
                else: FP += 1
            FN += len(gts)-sum(used)
        P = TP/(TP+FP) if TP+FP else 0; R = TP/(TP+FN) if TP+FN else 0
        sweep.append({'conf': thr, 'precision': round(P, 4), 'recall': round(R, 4), 'f1': round(2*P*R/(P+R), 4) if P+R else 0})
    res[crit] = {'sweep': sweep}
    at = lambda c: next(r for r in sweep if abs(r['conf']-c) < 1e-6)
    print(f'  [{nombre}|{crit}] @0.3 F1={at(0.30)["f1"]} R={at(0.30)["recall"]} | @0.5 F1={at(0.50)["f1"]}', flush=True)
(salida / f'{nombre}_valsweep.json').write_text(json.dumps(res, indent=2))
