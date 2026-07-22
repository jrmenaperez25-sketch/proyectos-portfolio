#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch
import torch.nn as nn

import config_mitos as CF
import evaluar_test_mitos as E

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
REF_MPP = 0.25
TAU_DET = 0.30
LORA_DIR = CF.ROOT / 'lora'; LORA_DIR.mkdir(exist_ok=True)
CACHE = CF.ROOT / 'lora_cache'; CACHE.mkdir(exist_ok=True)
VAL_PAC = {'05', '09', '11'}

def precompute(fm, crop, points, proc, tag):
    cache = CACHE / f'{fm}_crop{crop}_{tag}.pt'
    if cache.exists():
        return torch.load(cache)
    rad = {sc: round((crop / 2) * REF_MPP / CF.P.MPP_BY_SCANNER[sc]) for sc in CF.P.MPP_BY_SCANNER}
    byframe = defaultdict(list)
    for (sc, fr, cx, cy) in points:
        byframe[(sc, fr)].append((cx, cy))
    out = {}
    for i, ((sc, fr), cents) in enumerate(byframe.items()):
        fp = CF._frame_path(sc, fr)
        if fp is None:
            continue
        arr = np.asarray(Image.open(fp).convert('RGB')); r = rad[sc]
        for (cx, cy) in cents:
            sub = CF._crop_reflect(arr, float(cx), float(cy), r)
            pil = Image.fromarray(sub).resize((224, 224), Image.BICUBIC)
            out[f'{sc}|{fr}|{cx}|{cy}'] = proc(pil).half()
        if (i + 1) % 200 == 0:
            print(f'   crops {tag}: {i+1}/{len(byframe)} frames', flush=True)
    torch.save(out, cache)
    return out

class MitClf(nn.Module):
    def __init__(self, base, dim=2560):
        super().__init__()
        self.base = base
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        out = self.base(x)
        feat = torch.cat([out[:, 0], out[:, -256:].mean(1)], dim=-1)
        return self.head(feat).squeeze(-1)

def build_model(fm):
    from peft import LoraConfig, get_peft_model
    base, proc = CF.cargar_fm(fm)
    try:
        base.set_grad_checkpointing(True)
    except Exception:
        pass
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=['qkv'], lora_dropout=0.1, bias='none')
    base = get_peft_model(base, cfg, autocast_adapter_dtype=False)
    model = MitClf(base).cuda()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_tot = sum(p.numel() for p in model.parameters())
    print(f'  LoRA mínimo (r=4, qkv) + cabeza: {n_train/1e6:.2f}M entrenables / {n_tot/1e6:.0f}M '
          f'({100*n_train/n_tot:.2f}%)', flush=True)
    return model, proc

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1

def evaluar_end2end(pmit, salida_csv):
    GT = {sc: CF.gt_boxes(sc, pac) for sc, pac in CF.TEST_PAC}
    taus = [round(0.05 * i, 2) for i in range(1, 20)]
    filas = []
    for cdet, crit in [(c, k) for c in (0.20, 0.30, 0.50) for k in ('dist', 'iou')]:
        agg_base = [0, 0, 0, 0, 0]; agg = {t: [0, 0, 0, 0, 0] for t in taus}
        for f in CF.FOLDS:
            cands = [r for r in csv.DictReader(open(CF.CAND_DIR / f'A04_fold{f}_ov20.csv'))
                     if float(r['score']) >= cdet]
            for r in cands:
                r['p'] = pmit.get(f"{r['scanner']}|{r['frame']}|{r['cx']}|{r['cy']}", 1.0)

            def contar(keep):
                tp = fp = fn = fph = fpe = 0
                porf = defaultdict(list)
                for r in cands:
                    if keep(r):
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
            b = contar(lambda r: True)
            for i in range(5):
                agg_base[i] += b[i]
            for t in taus:
                c = contar(lambda r, t=t: r['p'] >= t)
                for i in range(5):
                    agg[t][i] += c[i]
        pb, rb, f1b = _prf(*agg_base[:3])
        bt = max(taus, key=lambda t: _prf(*agg[t][:3])[2])
        pc, rc, f1c = _prf(*agg[bt][:3])
        filas.append({'criterio': crit, 'conf_det': cdet,
                      'base_P': round(pb, 4), 'base_R': round(rb, 4), 'base_F1': round(f1b, 4),
                      'base_FPhard': agg_base[3], 'base_FPempty': agg_base[4],
                      'casc_tau': bt, 'casc_P': round(pc, 4), 'casc_R': round(rc, 4),
                      'casc_F1': round(f1c, 4), 'casc_FPhard': agg[bt][3], 'casc_FPempty': agg[bt][4]})
    with open(salida_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    return filas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fm', required=True, choices=['virchow', 'virchow2'])
    ap.add_argument('--crop', type=int, default=64, choices=[64, 96])
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()
    fm, crop = args.fm, args.crop

    model, proc = build_model(fm)

    pool = list(csv.DictReader(open(CF.POOL_DIR / 'pool.csv')))
    pts = [(r['scanner'], r['frame'], r['cx'], r['cy']) for r in pool]
    print(f'Precomputando recortes del pool ({len(pts)})…', flush=True)
    cache = precompute(fm, crop, pts, proc, 'pool')
    X, y, grp, keys = [], [], [], []
    for r in pool:
        k = f"{r['scanner']}|{r['frame']}|{r['cx']}|{r['cy']}"
        if k not in cache:
            continue
        keys.append(k); X.append(cache[k]); y.append(1.0 if r['label'] == 'mit' else 0.0)
        grp.append(r['frame'][1:3])
    X = torch.stack(X); y = torch.tensor(y); grp = np.array(grp)
    tr = np.array([g not in VAL_PAC for g in grp]); va = ~tr
    pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
    print(f'  pool: {len(y)} (train {tr.sum()} / val {va.sum()}); pos_weight={pos_w:.1f}', flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w).cuda())
    idx_tr = np.where(tr)[0]

    def val_probs():
        model.eval(); ps = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, int(va.sum()), 64):
                xb = X[np.where(va)[0][i:i+64]].float().cuda()
                ps.append(torch.sigmoid(model(xb)).float().cpu())
        return torch.cat(ps).numpy()

    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    yval = y[va].numpy()
    best_f1, best_state, best_metr = -1, None, None
    for ep in range(args.epochs):
        model.train(); np.random.shuffle(idx_tr); tot = 0.0
        for i in range(0, len(idx_tr), args.batch):
            bi = idx_tr[i:i+args.batch]
            xb = X[bi].float().cuda(); yb = y[bi].cuda()
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                logit = model(xb); loss = lossf(logit, yb)
            loss.backward(); opt.step(); tot += loss.item() * len(bi)
        pv = val_probs()
        thr = max(np.linspace(0.05, 0.95, 19), key=lambda t: f1_score(yval, (pv >= t).astype(int)))
        yh = (pv >= thr).astype(int)
        f1 = f1_score(yval, yh)
        print(f'  ep{ep+1}/{args.epochs} loss={tot/len(idx_tr):.4f} | valF1={f1:.4f} '
              f'P={precision_score(yval,yh):.3f} R={recall_score(yval,yh):.3f} AUC={roc_auc_score(yval,pv):.4f}',
              flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_metr = {'fm': fm, 'crop': crop, 'metodo': 'lora_r4_qkv', 'epoca': ep+1,
                         'clf_thr': round(float(thr), 3), 'clf_precision': round(float(precision_score(yval, yh)), 4),
                         'clf_recall': round(float(recall_score(yval, yh)), 4), 'clf_f1': round(float(f1), 4),
                         'clf_auc': round(float(roc_auc_score(yval, pv)), 4),
                         'n_val': int(va.sum()), 'n_train': int(tr.sum()), 'pos_weight': round(pos_w, 1)}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.named_parameters() if v.requires_grad}
    if best_state:
        model.load_state_dict(best_state, strict=False)
        torch.save(best_state, LORA_DIR / f'{fm}_crop{crop}_model.pt')
    (LORA_DIR / f'{fm}_crop{crop}_clf.json').write_text(json.dumps(best_metr, indent=2))
    print(f'[CLASIFICADOR LoRA {fm} crop{crop}] mejor val: {best_metr}', flush=True)

    candpts = set()
    for f in CF.FOLDS:
        for r in csv.DictReader(open(CF.CAND_DIR / f'A04_fold{f}_ov20.csv')):
            if float(r['score']) >= TAU_DET:
                candpts.add((r['scanner'], r['frame'], r['cx'], r['cy']))
    print(f'Precomputando recortes de candidatos A04 ({len(candpts)})…', flush=True)
    cc = precompute(fm, crop, list(candpts), proc, 'candA04')
    model.eval(); pmit = {}
    ck = list(cc.keys())
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(ck), 64):
            kb = ck[i:i+64]; xb = torch.stack([cc[k] for k in kb]).float().cuda()
            pr = torch.sigmoid(model(xb)).float().cpu().numpy()
            for k, p in zip(kb, pr):
                pmit[k] = float(p)
    (LORA_DIR / f'{fm}_crop{crop}_pmit.json').write_text(json.dumps(pmit))

    out = CF.EVAL_DIR / f'{fm}_crop{crop}_lora.csv'
    filas = evaluar_end2end(pmit, out)
    print(f'[END2END LoRA {fm} crop{crop}] (conf 0.30, solape 20):', flush=True)
    for r in filas:
        print(f"  {r['criterio']}: base F1={r['base_F1']} R={r['base_R']} -> "
              f"casc F1={r['casc_F1']} R={r['casc_R']} (tau={r['casc_tau']}) "
              f"FPhard {r['base_FPhard']}->{r['casc_FPhard']}", flush=True)
    print('->', out)

if __name__ == '__main__':
    main()
