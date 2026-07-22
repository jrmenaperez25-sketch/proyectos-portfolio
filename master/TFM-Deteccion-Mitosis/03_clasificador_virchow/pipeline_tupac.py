#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import argparse, csv, json, math, os, shutil, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_REPO)):
    _p = os.path.join(_REPO, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import parches_tupac as P
import evaluar_test_tupac as E
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(f'{WORKSPACE}/segunda_fase_tupac')
for d in [ROOT, ROOT / 'pool', ROOT / 'embeddings', ROOT / 'clasificadores', ROOT / 'evaluacion']:
    d.mkdir(parents=True, exist_ok=True)
CKPT = f'{WORKSPACE}/EXPERIMENTO_TUPAC73/run/rfdetr_ovs/checkpoint_best_total.pth'
TAU_DET = 0.20
DIST_PX = 7.5 / P.MPP
CROP = 64
RAD = round(CROP / 2 * 0.25 / P.MPP)
SEED = 42

TRAIN, VAL, TEST = P.SPLIT['train'], P.SPLIT['val'], P.SPLIT['test']

def frame_path(caso, frame_stem):
    hh = frame_stem.split('_', 1)[1]
    return P.IMG_ROOT / caso / f'{hh}.tif'

def origenes(L, stride=512):
    if L <= 512:
        return [0]
    xs = list(range(0, L - 512 + 1, stride))
    if xs[-1] != L - 512:
        xs.append(L - 512)
    return xs

def cargar_detector():
    import rfdetr
    return rfdetr.RFDETRSmall(pretrain_weights=CKPT)

def predict_frame(model, img, W, H):
    tmp = Path('/tmp/dos_fases_tiles')
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True, exist_ok=True)
    meta = {}
    for y0 in origenes(H):
        for x0 in origenes(W):
            nm = f'{x0:05d}_{y0:05d}.jpg'
            img.crop((x0, y0, x0 + 512, y0 + 512)).save(tmp / nm, quality=92)
            meta[nm] = (x0, y0)
    preds = model.predict([str(p) for p in sorted(tmp.glob('*.jpg'))], threshold=0.05)
    if not isinstance(preds, list):
        preds = [preds]
    boxes = []
    for p, det in zip(sorted(tmp.glob('*.jpg')), preds):
        if det is None:
            continue
        ox, oy = meta[p.name]
        for b, c in zip(det.xyxy, det.confidence):
            boxes.append((ox + float(b[0]), oy + float(b[1]), ox + float(b[2]), oy + float(b[3]), float(c)))
    shutil.rmtree(tmp, ignore_errors=True)
    return boxes

def cmd_detectar(args):
    E.MATCH_MODE = 'distancia'; E.DIST_PX = DIST_PX
    model = cargar_detector()
    pool = []
    for caso in TRAIN + VAL:
        split = 'train' if caso in TRAIN else 'val'
        nfp = 0
        for fr in P.load_case_frames(caso):
            for (x, y, c, mid) in fr['mit_pts']:
                pool.append({'caso': caso, 'split': split, 'frame': fr['frame_stem'],
                             'cx': round(x, 2), 'cy': round(y, 2), 'label': 'mit', 'source': 'gt_mit'})
            for (x, y, c) in fr['not_pts']:
                pool.append({'caso': caso, 'split': split, 'frame': fr['frame_stem'],
                             'cx': round(x, 2), 'cy': round(y, 2), 'label': 'neg', 'source': 'not_mit'})
            if split != 'val':
                continue
            img = Image.open(fr['img_path']).convert('RGB'); W, H = img.size
            boxes = predict_frame(model, img, W, H); img.close()
            r = P.BBOX_RADIUS_PX
            mitb = [(x - r, y - r, x + r, y + r) for (x, y, c, mid) in fr['mit_pts']]
            for (x1, y1, x2, y2, c) in E.dedup(boxes):
                if c < TAU_DET:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                hit = any(((cx - (g[0]+g[2])/2)**2 + (cy - (g[1]+g[3])/2)**2)**0.5 <= DIST_PX for g in mitb)
                if not hit:
                    pool.append({'caso': caso, 'split': split, 'frame': fr['frame_stem'],
                                 'cx': round(cx, 2), 'cy': round(cy, 2), 'label': 'neg', 'source': 'mined_fp'})
                    nfp += 1
        if split == 'val':
            print(f'  [val] caso {caso}: +{nfp} FP minados', flush=True)
    with open(ROOT / 'pool' / 'pool.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['caso', 'split', 'frame', 'cx', 'cy', 'label', 'source'])
        w.writeheader(); w.writerows(pool)
    from collections import Counter
    print('POOL:', dict(Counter(f"{r['split']}/{r['label']}/{r['source']}" for r in pool)), flush=True)

    cands = []
    for caso in TEST:
        vdir = P.VERS / caso / 'sliding20'
        man = {r['patch_filename']: r for r in csv.DictReader(open(vdir / '_manifest.csv'))}
        imgs = sorted((vdir / 'images').glob('*.jpg'))
        preds = predecir_persist(model, imgs)
        porframe = defaultdict(list)
        for nm, bx in preds.items():
            m = man.get(nm)
            if not m:
                continue
            ox = float(m['patch_center_x']) - 256; oy = float(m['patch_center_y']) - 256
            for (x1, y1, x2, y2, c) in bx:
                porframe[m['frame_stem']].append((ox + x1, oy + y1, ox + x2, oy + y2, c))
        gt = {fr['frame_stem']: [(x, y) for (x, y, c, mid) in fr['mit_pts']] for fr in P.load_case_frames(caso)}
        for frame, boxes in porframe.items():
            for (x1, y1, x2, y2, c) in E.dedup(boxes):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                tp = any(((cx - gx)**2 + (cy - gy)**2)**0.5 <= DIST_PX for (gx, gy) in gt.get(frame, []))
                cands.append({'caso': caso, 'frame': frame, 'cx': round(cx, 2), 'cy': round(cy, 2),
                              'x1': round(x1, 2), 'y1': round(y1, 2), 'x2': round(x2, 2), 'y2': round(y2, 2),
                              'score': round(c, 4), 'es_tp': int(tp)})
        print(f'  [test] caso {caso}: {sum(1 for r in cands if r["caso"]==caso)} candidatos', flush=True)
    with open(ROOT / 'pool' / 'candidatos.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['caso', 'frame', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'score', 'es_tp'])
        w.writeheader(); w.writerows(cands)
    print(f'CANDIDATOS test: {len(cands)} (TP={sum(r["es_tp"] for r in cands)})', flush=True)

def predecir_persist(model, imgs, bs=16):
    out = defaultdict(list)
    for i in range(0, len(imgs), bs):
        batch = imgs[i:i+bs]
        res = model.predict([str(p) for p in batch], threshold=0.05)
        if not isinstance(res, list):
            res = [res]
        for p, det in zip(batch, res):
            if det is None:
                continue
            for b, c in zip(det.xyxy, det.confidence):
                out[p.name].append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)))
    return out

def _crop_reflect(arr, cx, cy, r):
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = int(round(cx - r)), int(round(cy - r)), int(round(cx + r)), int(round(cy + r))
    px0, py0 = max(0, -x0), max(0, -y0); px1, py1 = max(0, x1 - W), max(0, y1 - H)
    sub = arr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if sub.size == 0:
        return np.zeros((2 * r, 2 * r, 3), np.uint8)
    if px0 or py0 or px1 or py1:
        sub = np.pad(sub, ((py0, py1), (px0, px1), (0, 0)), mode='reflect')
    return sub

def cargar_virchow():
    import torch, timm
    from timm.layers import SwiGLUPacked
    from timm.data import resolve_data_config, create_transform
    model = timm.create_model('hf-hub:paige-ai/Virchow', pretrained=True,
                              mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().cuda()
    proc = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, proc

def crops_tensores(points, proc):
    porframe = defaultdict(list)
    for r in points:
        porframe[(r['caso'], r['frame'])].append((r['cx'], r['cy']))
    out = {}
    for i, ((caso, frame), cents) in enumerate(porframe.items()):
        fp = frame_path(caso, frame)
        if not fp.exists():
            continue
        arr = np.asarray(Image.open(fp).convert('RGB'))
        for (cx, cy) in cents:
            sub = _crop_reflect(arr, float(cx), float(cy), RAD)
            pil = Image.fromarray(sub).resize((224, 224), Image.BICUBIC)
            out[f'{caso}|{frame}|{cx}|{cy}'] = proc(pil).half()
        if (i + 1) % 100 == 0:
            print(f'   crops {i+1}/{len(porframe)} frames', flush=True)
    return out

def embed_all(base, tensores, keys):
    import torch
    embs = []
    for i in range(0, len(keys), 32):
        xb = torch.stack([tensores[k].float() for k in keys[i:i+32]]).cuda()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            out = base(xb)
        emb = torch.cat([out[:, 0], out[:, -256:].mean(1)], dim=-1) if out.dim() == 3 else out
        embs.append(emb.float().cpu().numpy())
    return np.concatenate(embs)

import torch
import torch.nn as nn
class MitClf(nn.Module):
    def __init__(self, base, dim=2560):
        super().__init__(); self.base = base; self.head = nn.Linear(dim, 1)
    def forward(self, x):
        out = self.base(x)
        feat = torch.cat([out[:, 0], out[:, -256:].mean(1)], dim=-1)
        return self.head(feat).squeeze(-1)

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2*p*r/(p+r) if p+r else 0.0)

def metrics_at(y, prob, thr):
    yh = (prob >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    p, r, f1 = _prf(tp, fp, fn)
    return {'P': round(p, 4), 'R': round(r, 4), 'F1': round(f1, 4)}

def end2end(prob_cand, cands, salida):
    gt = {}
    for caso in TEST:
        for fr in P.load_case_frames(caso):
            r = P.BBOX_RADIUS_PX
            gt[fr['frame_stem']] = [(x - r, y - r, x + r, y + r) for (x, y, c, m) in fr['mit_pts']]
    taus = [round(0.05*i, 2) for i in range(1, 20)]
    filas = []
    for cdet in (0.30, 0.50):
        for crit in ('dist', 'iou'):
            base = [0, 0, 0]; agg = {t: [0, 0, 0] for t in taus}
            porframe = defaultdict(list)
            for r in cands:
                if float(r['score']) >= cdet:
                    r2 = dict(r); r2['p'] = prob_cand.get(f"{r['caso']}|{r['frame']}|{r['cx']}|{r['cy']}", 1.0)
                    porframe[r['frame']].append(r2)
            def contar(keep):
                tp = fp = fn = 0
                for frame, gts in gt.items():
                    preds = sorted([r for r in porframe.get(frame, []) if keep(r)], key=lambda r: -float(r['score']))
                    usado = [False]*len(gts)
                    for r in preds:
                        pb = (float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']))
                        cx, cy = float(r['cx']), float(r['cy'])
                        best, bq = -1, None
                        for gi, gb in enumerate(gts):
                            if usado[gi]:
                                continue
                            if crit == 'iou':
                                q = E.iou(pb, gb); ok = q >= 0.50
                            else:
                                d = ((cx-(gb[0]+gb[2])/2)**2 + (cy-(gb[1]+gb[3])/2)**2)**0.5; ok = d <= DIST_PX; q = -d
                            if ok and (bq is None or q > bq):
                                best, bq = gi, q
                        if best >= 0:
                            usado[best] = True; tp += 1
                        else:
                            fp += 1
                    fn += len(gts) - sum(usado)
                return tp, fp, fn
            b = contar(lambda r: True); base = b
            for t in taus:
                agg[t] = contar(lambda r, t=t: r['p'] >= t)
            pb, rb, f1b = _prf(*base)
            bt = max(taus, key=lambda t: _prf(*agg[t])[2]); pc, rc, f1c = _prf(*agg[bt])
            filas.append({'conf_det': cdet, 'criterio': crit,
                          'base_P': round(pb, 4), 'base_R': round(rb, 4), 'base_F1': round(f1b, 4),
                          'casc_tau': bt, 'casc_P': round(pc, 4), 'casc_R': round(rc, 4), 'casc_F1': round(f1c, 4)})
    with open(salida, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    return filas

def cmd_entrenar(args):
    import torch
    np.random.seed(SEED); torch.manual_seed(SEED)
    pool = list(csv.DictReader(open(ROOT / 'pool' / 'pool.csv')))
    cands = list(csv.DictReader(open(ROOT / 'pool' / 'candidatos.csv')))
    print(f'pool {len(pool)} | candidatos {len(cands)}', flush=True)
    base, proc = cargar_virchow()

    print('Recortando pool…', flush=True)
    t_pool = crops_tensores(pool, proc)
    print('Recortando candidatos test…', flush=True)
    t_cand = crops_tensores([{'caso': r['caso'], 'frame': r['frame'], 'cx': r['cx'], 'cy': r['cy']} for r in cands], proc)

    kpool = [f"{r['caso']}|{r['frame']}|{r['cx']}|{r['cy']}" for r in pool if f"{r['caso']}|{r['frame']}|{r['cx']}|{r['cy']}" in t_pool]
    pool = [r for r in pool if f"{r['caso']}|{r['frame']}|{r['cx']}|{r['cy']}" in t_pool]
    y = np.array([1 if r['label'] == 'mit' else 0 for r in pool])
    is_val = np.array([r['split'] == 'val' for r in pool])
    kcand = [f"{r['caso']}|{r['frame']}|{r['cx']}|{r['cy']}" for r in cands]
    ycand = np.array([int(r['es_tp']) for r in cands])

    print('Embeddings frozen…', flush=True)
    emb_pool = embed_all(base, t_pool, kpool)
    emb_cand = embed_all(base, t_cand, [k for k in kcand if k in t_cand])
    kcand_ok = [k for k in kcand if k in t_cand]; ycand_ok = np.array([ycand[i] for i, k in enumerate(kcand) if k in t_cand])

    resultados = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    tr = ~is_val
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0))
    clf.fit(emb_pool[tr], y[tr])
    import joblib; joblib.dump(clf, ROOT / 'clasificadores' / 'frozen_clf.joblib')
    pv = clf.predict_proba(emb_pool[is_val])[:, 1]
    thr = max(np.linspace(0.05, 0.95, 19), key=lambda t: _prf(int(((pv>=t)&(y[is_val]==1)).sum()), int(((pv>=t)&(y[is_val]==0)).sum()), int(((pv<t)&(y[is_val]==1)).sum()))[2])
    val_m = metrics_at(y[is_val], pv, thr); val_m['AUC'] = round(roc_auc_score(y[is_val], pv), 4)
    pt = clf.predict_proba(emb_cand)[:, 1]
    test_m = metrics_at(ycand_ok, pt, thr); test_m['AUC'] = round(roc_auc_score(ycand_ok, pt), 4)
    resultados['frozen'] = {'thr': round(float(thr), 3), 'val': val_m, 'test_cand': test_m}
    prob_cand_frozen = {k: float(p) for k, p in zip(kcand_ok, pt)}
    print(f'  FROZEN val {val_m} | test {test_m}', flush=True)
    e2e_frozen = end2end(prob_cand_frozen, cands, ROOT / 'evaluacion' / 'end2end_frozen.csv')

    from peft import LoraConfig, get_peft_model
    try: base.set_grad_checkpointing(True)
    except Exception: pass
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=['qkv'], lora_dropout=0.1, bias='none')
    base = get_peft_model(base, cfg, autocast_adapter_dtype=False)
    model = MitClf(base).cuda()
    pos_w = float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w).cuda())
    idx_tr = np.where(tr)[0]
    def probs(keys):
        model.eval(); ps = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(keys), 64):
                xb = torch.stack([t_pool.get(k, t_cand.get(k))[:] for k in keys[i:i+64]]).float().cuda()
                ps.append(torch.sigmoid(model(xb)).float().cpu())
        return torch.cat(ps).numpy()
    kval = [kpool[i] for i in np.where(is_val)[0]]; yval = y[is_val]
    best = {'f1': -1}
    for ep in range(8):
        model.train(); np.random.shuffle(idx_tr)
        for i in range(0, len(idx_tr), 16):
            bi = idx_tr[i:i+16]
            xb = torch.stack([t_pool[kpool[j]].float() for j in bi]).cuda()
            yb = torch.tensor([float(y[j]) for j in bi]).cuda()
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        pv = probs(kval)
        thr = max(np.linspace(0.05, 0.95, 19), key=lambda t: _prf(int(((pv>=t)&(yval==1)).sum()), int(((pv>=t)&(yval==0)).sum()), int(((pv<t)&(yval==1)).sum()))[2])
        vm = metrics_at(yval, pv, thr); auc = roc_auc_score(yval, pv)
        print(f'  LoRA ep{ep+1}/8 valF1={vm["F1"]} P={vm["P"]} R={vm["R"]} AUC={auc:.3f}', flush=True)
        if vm['F1'] > best['f1']:
            pt = probs(kcand_ok); tm = metrics_at(ycand_ok, pt, thr); tm['AUC'] = round(roc_auc_score(ycand_ok, pt), 4)
            best = {'f1': vm['F1'], 'epoca': ep+1, 'thr': round(float(thr), 3),
                    'val': {**vm, 'AUC': round(auc, 4)}, 'test_cand': tm,
                    'prob_cand': {k: float(p) for k, p in zip(kcand_ok, pt)}}
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(best_state, ROOT / 'clasificadores' / 'lora_model.pt')
    resultados['lora'] = {k: v for k, v in best.items() if k != 'prob_cand'}
    print(f'  LoRA mejor val: {best["val"]} | test {best["test_cand"]}', flush=True)
    e2e_lora = end2end(best['prob_cand'], cands, ROOT / 'evaluacion' / 'end2end_lora.csv')

    (ROOT / 'clasificadores' / 'metricas_clasificacion.json').write_text(json.dumps(resultados, indent=2))
    print('GUARDADO metricas_clasificacion.json + end2end_{frozen,lora}.csv', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('detectar')
    sub.add_parser('entrenar')
    args = ap.parse_args()
    if args.cmd == 'detectar':
        cmd_detectar(args)
    elif args.cmd == 'entrenar':
        cmd_entrenar(args)
