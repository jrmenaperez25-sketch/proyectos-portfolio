#!/usr/bin/env python3
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import config_mitos as CF
import evaluar_test_mitos as E

NGT = 531

def pmit_frozen(fm, crop):
    import joblib
    clf = joblib.load(CF.CLF_DIR / f'{fm}_crop{crop}.joblib')
    emb = CF._load_emb(fm, crop)
    keys = list(emb); X = np.array([emb[k] for k in keys])
    p = clf.predict_proba(X)[:, 1]
    return {k: float(p[i]) for i, k in enumerate(keys)}, \
        json.loads((CF.CLF_DIR / f'{fm}_crop{crop}.json').read_text())['clf_thr']

def pmit_lora(fm, crop):
    pm = json.loads((CF.ROOT / 'lora' / f'{fm}_crop{crop}_pmit.json').read_text())
    thr = json.loads((CF.ROOT / 'lora' / f'{fm}_crop{crop}_clf.json').read_text())['clf_thr']
    return pm, thr

def contar(cands, GT, crit):
    tp = fp = fn = fph = fpe = 0
    porf = defaultdict(list)
    for r in cands:
        porf[(r['scanner'], r['frame'])].append(r)
    for sc in GT:
        for fr, (mitb, notb) in GT[sc].items():
            thr = CF.dist_px(sc)
            preds = sorted(porf.get((sc, fr), []), key=lambda r: -float(r['score']))
            usado = [False] * len(mitb)
            for r in preds:
                cx, cy = float(r['cx']), float(r['cy'])
                pb = (float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']))
                best, bq = -1, None
                for gi, gb in enumerate(mitb):
                    if usado[gi]:
                        continue
                    if crit == 'iou':
                        ok = E.iou(pb, gb) >= 0.50; q = E.iou(pb, gb)
                    else:
                        gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                        d = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5
                        ok = d <= thr; q = -d
                    if ok and (bq is None or q > bq):
                        best, bq = gi, q
                if best >= 0:
                    usado[best] = True; tp += 1
                else:
                    fp += 1
                    hard = any(((cx - (n[0]+n[2])/2)**2 + (cy - (n[1]+n[3])/2)**2)**0.5 <= thr for n in notb)
                    fph += hard; fpe += (not hard)
            fn += len(mitb) - sum(usado)
    return tp, fp, fn, fph, fpe

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0)

def ms(xs):
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)

def clf_cands(cands, GT, crit, tau):
    from sklearn.metrics import roc_auc_score
    labels, probs = [], []
    porf = defaultdict(list)
    for r in cands:
        porf[(r['scanner'], r['frame'])].append(r)
    for sc in GT:
        for fr, (mitb, notb) in GT[sc].items():
            thr = CF.dist_px(sc)
            preds = sorted(porf.get((sc, fr), []), key=lambda r: -float(r['score']))
            usado = [False] * len(mitb)
            for r in preds:
                cx, cy = float(r['cx']), float(r['cy'])
                pb = (float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']))
                best, bq = -1, None
                for gi, gb in enumerate(mitb):
                    if usado[gi]:
                        continue
                    if crit == 'iou':
                        ok = E.iou(pb, gb) >= 0.50; q = E.iou(pb, gb)
                    else:
                        gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                        d = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5; ok = d <= thr; q = -d
                    if ok and (bq is None or q > bq):
                        best, bq = gi, q
                if best >= 0:
                    usado[best] = True; labels.append(1)
                else:
                    labels.append(0)
                probs.append(float(r['p']))
    labels = np.array(labels); probs = np.array(probs); yhat = (probs >= tau).astype(int)
    tp = int(((labels == 1) & (yhat == 1)).sum()); fp = int(((labels == 0) & (yhat == 1)).sum())
    fn = int(((labels == 1) & (yhat == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float('nan')
    return P, R, F1, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fm', default='virchow')
    ap.add_argument('--crop', type=int, default=64)
    ap.add_argument('--metodo', choices=['frozen', 'lora'], required=True)
    args = ap.parse_args()
    pmit, tau = (pmit_frozen if args.metodo == 'frozen' else pmit_lora)(args.fm, args.crop)
    GT = {sc: CF.gt_boxes(sc, pac) for sc, pac in CF.TEST_PAC}
    print(f"== {args.fm} crop{args.crop} {args.metodo} | tau(val)={tau} | media±std de {len(CF.FOLDS)} folds (A04) ==")
    out = []
    for cdet in (0.20, 0.30, 0.50):
        for crit in ('dist', 'iou'):
            B = defaultdict(list); C = defaultdict(list); L = defaultdict(list)
            for f in CF.FOLDS:
                cands = [r for r in csv.DictReader(open(CF.CAND_DIR / f'A04_fold{f}_ov20.csv'))
                         if float(r['score']) >= cdet]
                for r in cands:
                    r['p'] = pmit.get(f"{r['scanner']}|{r['frame']}|{r['cx']}|{r['cy']}", 1.0)
                tb = contar(cands, GT, crit)
                tc = contar([r for r in cands if r['p'] >= tau], GT, crit)
                for nm, t, D in [('b', tb, B), ('c', tc, C)]:
                    p, r, f1 = prf(t[0], t[1], t[2])
                    D['TP'].append(t[0]); D['FPh'].append(t[3]); D['FPe'].append(t[4])
                    D['P'].append(p); D['R'].append(r); D['F1'].append(f1)
                cp, cr, cf1, cauc = clf_cands(cands, GT, crit, tau)
                L['P'].append(cp); L['R'].append(cr); L['F1'].append(cf1)
                if cauc == cauc:
                    L['AUC'].append(cauc)
            row = {'fm': args.fm, 'crop': args.crop, 'metodo': args.metodo, 'conf_det': cdet,
                   'criterio': crit, 'tau': tau}
            for tag, D in [('base', B), ('casc', C)]:
                for m in ['P', 'R', 'F1', 'TP', 'FPh', 'FPe']:
                    mu, sd = ms(D[m]); row[f'{tag}_{m}'] = round(mu, 3); row[f'{tag}_{m}_sd'] = round(sd, 3)
            for m in ['P', 'R', 'F1', 'AUC']:
                if L[m]:
                    mu, sd = ms(L[m]); row[f'clf_{m}'] = round(mu, 3); row[f'clf_{m}_sd'] = round(sd, 3)
            out.append(row)
            if crit == 'dist':
                print(f"\n  conf {cdet} ({crit}):")
                print(f"    BASELINE  P={row['base_P']}±{row['base_P_sd']}  R={row['base_R']}±{row['base_R_sd']}  "
                      f"F1={row['base_F1']}±{row['base_F1_sd']}  | TP {row['base_TP']:.0f} FP_hard {row['base_FPh']:.0f} FP_empty {row['base_FPe']:.0f}")
                print(f"    DOS FASES P={row['casc_P']}±{row['casc_P_sd']}  R={row['casc_R']}±{row['casc_R_sd']}  "
                      f"F1={row['casc_F1']}±{row['casc_F1_sd']}  | TP {row['casc_TP']:.0f} FP_hard {row['casc_FPh']:.0f} FP_empty {row['casc_FPe']:.0f}")
                print(f"    -> pierde {row['base_TP']-row['casc_TP']:.0f} mitosis, quita "
                      f"{row['base_FPh']-row['casc_FPh']:.0f} FP_hard y {row['base_FPe']-row['casc_FPe']:.0f} FP_empty (por A04)")
                print(f"    CLASIF/candidatos  P={row.get('clf_P')}±{row.get('clf_P_sd')}  "
                      f"R={row.get('clf_R')}±{row.get('clf_R_sd')}  F1={row.get('clf_F1')}±{row.get('clf_F1_sd')}  "
                      f"AUC={row.get('clf_AUC')}±{row.get('clf_AUC_sd')}")
    sal = CF.EVAL_DIR / f'{args.fm}_crop{args.crop}_{args.metodo}_perfold.csv'
    with open(sal, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"\n-> {sal}")

if __name__ == '__main__':
    main()
