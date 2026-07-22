#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import sys, csv, math
from collections import defaultdict
import numpy as np, joblib, torch
from pathlib import Path
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_REPO)):
    _p = os.path.join(_REPO, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import pipeline_tupac as CT
import virchow_nucleo as XC
from sklearn.metrics import roc_auc_score
P = CT.P; TEST = CT.TEST; DIST_PX = CT.DIST_PX
FROZEN = f'{WORKSPACE}/segunda_fase_tupac/clasificadores/frozen_clf.joblib'
LORA = f'{WORKSPACE}/segunda_fase_tupac/clasificadores/lora_model.pt'

def _prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.; r = tp/(tp+fn) if tp+fn else 0.
    return p, r, (2*p*r/(p+r) if p+r else 0.)

gt_mit, gt_not = {}, {}
for caso in TEST:
    for fr in P.load_case_frames(caso):
        gt_mit[fr['frame_stem']] = [(x, y) for (x, y, c, m) in fr['mit_pts']]
        gt_not[fr['frame_stem']] = [(x, y) for (x, y, c) in fr['not_pts']]
N_gt = sum(len(v) for v in gt_mit.values())
print(f'TEST: {len(gt_mit)} frames, {N_gt} mitosis, {sum(len(v) for v in gt_not.values())} hard-neg anotados', flush=True)

cands = list(csv.DictReader(open(f'{WORKSPACE}/segunda_fase_tupac/pool/candidatos.csv')))
for r in cands:
    r['cx'] = float(r['cx']); r['cy'] = float(r['cy']); r['score'] = float(r['score'])
print(f'candidatos: {len(cands)}', flush=True)

base, proc = CT.cargar_virchow()
pts = [{'caso': r['caso'], 'frame': r['frame'], 'cx': r['cx'], 'cy': r['cy']} for r in cands]
tens = CT.crops_tensores(pts, proc)
keys = [f"{r['caso']}|{r['frame']}|{r['cx']}|{r['cy']}" for r in cands]
klist = [k for k in keys if k in tens]
emb = CT.embed_all(base, tens, klist)
frozen_clf = joblib.load(FROZEN)
pf = frozen_clf.predict_proba(emb)[:, 1]
prob_frozen = {k: float(p) for k, p in zip(klist, pf)}
lora, _ = XC.load_clf(LORA)
prob_lora = {}
for i in range(0, len(klist), 64):
    xb = torch.stack([tens[k].float() for k in klist[i:i+64]]).cuda()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        pr = torch.sigmoid(lora(xb)).float().cpu().tolist()
    for k, pp in zip(klist[i:i+64], pr): prob_lora[k] = pp
for r, k in zip(cands, keys):
    r['pf'] = prob_frozen.get(k, 1.0); r['pl'] = prob_lora.get(k, 1.0)
torch.cuda.empty_cache()

valrows = [r for r in csv.DictReader(open(f'{WORKSPACE}/segunda_fase_tupac/pool/pool.csv')) if r['split'] == 'val']
vpts = [{'caso': r['caso'], 'frame': r['frame'], 'cx': float(r['cx']), 'cy': float(r['cy'])} for r in valrows]
yv = np.array([1 if r['label'] == 'mit' else 0 for r in valrows])
vtens = CT.crops_tensores(vpts, proc)
vkeys = [f"{r['caso']}|{r['frame']}|{float(r['cx'])}|{float(r['cy'])}" for r in valrows]
mask = [i for i, k in enumerate(vkeys) if k in vtens]
vemb = CT.embed_all(base, vtens, [vkeys[i] for i in mask])
pv = frozen_clf.predict_proba(vemb)[:, 1]; yvm = yv[mask]
taus = [round(0.05*i, 2) for i in range(1, 20)]
tau_frozen = max(taus, key=lambda t: _prf(int(((pv >= t) & (yvm == 1)).sum()), int(((pv >= t) & (yvm == 0)).sum()), int(((pv < t) & (yvm == 1)).sum()))[2])
vP, vR, vF = _prf(int(((pv >= tau_frozen) & (yvm == 1)).sum()), int(((pv >= tau_frozen) & (yvm == 0)).sum()), int(((pv < tau_frozen) & (yvm == 1)).sum()))
print(f'\n[τ frozen] argmax F1 en VALIDACIÓN = {tau_frozen}  (val P{vP:.4f} R{vR:.4f} F1{vF:.4f}, AUC{roc_auc_score(yvm,pv):.4f})', flush=True)

RBB = P.BBOX_RADIUS_PX
def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1); inter = iw*ih
    if inter <= 0: return 0.
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.
def contar(conf, keep, crit='dist'):
    tp = fph = fpe = 0
    porframe = defaultdict(list)
    for r in cands:
        if r['score'] >= conf and keep(r): porframe[r['frame']].append(r)
    for frame, mits in gt_mit.items():
        preds = sorted(porframe.get(frame, []), key=lambda r: -r['score'])
        used = [False]*len(mits)
        nots = gt_not.get(frame, [])
        for r in preds:
            box = (float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']))
            best, bq = -1, None
            for gi, (mx, my) in enumerate(mits):
                if used[gi]: continue
                if crit == 'iou':
                    q = _iou(box, (mx-RBB, my-RBB, mx+RBB, my+RBB)); ok = q >= 0.5
                else:
                    q = -math.hypot(r['cx']-mx, r['cy']-my); ok = -q <= DIST_PX
                if ok and (bq is None or q > bq): best, bq = gi, q
            if best >= 0:
                used[best] = True; tp += 1
            else:
                hard = any(math.hypot(r['cx']-nx, r['cy']-ny) <= DIST_PX for (nx, ny) in nots)
                fph += hard; fpe += (not hard)
    fn = N_gt - tp
    return tp, fph, fpe, fn

print('\n########## PARTE A — END-TO-END TUPAC (τ FIJO de validación) ##########')
out=[]
for crit in ['dist', 'iou']:
    print(f'  -- criterio {crit.upper()} --')
    for name, keep in [('detector solo', lambda r: True),
                       (f'+ frozen(τ{tau_frozen})', lambda r: r['pf'] >= tau_frozen),
                       ('+ LoRA(τ0.85)', lambda r: r['pl'] >= 0.85)]:
        linea=f'  {name:16} {crit:4}:'
        for conf in [0.30, 0.50]:
            tp, fph, fpe, fn = contar(conf, keep, crit); p, rr, f1 = _prf(tp, fph+fpe, fn)
            linea+=f'  conf{conf}: P{p:.3f} R{rr:.3f} F1{f1:.3f} |'
        print(linea); out.append(linea)

print('\n########## PARTE B — DESGLOSE FP (TUPAC, conf det 0.3, distancia) ##########')
btp, bfph, bfpe, bfn = contar(0.30, lambda r: True)
print(f'  BASE (detector solo): TP {btp} | FP_hard {bfph} | FP_empty {bfpe} (N_gt {N_gt})')
def pct(a, b): return f'{a}/{b} ({100*a/b:.0f}%)' if b else f'{a}/0 (-)'
for name, pk, tau in [('frozen', 'pf', tau_frozen), ('LoRA', 'pl', 0.85)]:
    ctp, cfph, cfpe, cfn = contar(0.30, lambda r, pk=pk, tau=tau: r[pk] >= tau)
    print(f'  {name:6} (τ{tau}): FP_empty elim {pct(bfpe-cfpe,bfpe)} | FP_hard elim {pct(bfph-cfph,bfph)} | mitosis perdidas {pct(btp-ctp,btp)}')

Path(f'{WORKSPACE}/segunda_fase_tupac/evaluacion/PARTE_A_B_frozen.txt').write_text('\n'.join(out))
print('\nOK')
