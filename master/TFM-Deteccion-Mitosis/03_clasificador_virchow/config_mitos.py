#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED); np.random.seed(SEED)
os.environ.setdefault('PYTHONHASHSEED', str(SEED))

sys.path.insert(0, str(Path(__file__).parent))
import parches_mitos as P
import evaluar_test_mitos as E

ROOT = Path(f'{WORKSPACE}/segunda_fase_mitos')
CAND_DIR = ROOT / 'candidatos'
POOL_DIR = ROOT / 'pool'
EMB_DIR = ROOT / 'embeddings'
CLF_DIR = ROOT / 'clasificadores'
EVAL_DIR = ROOT / 'evaluacion'
FIG_DIR = ROOT / 'figuras'
for d in (ROOT, CAND_DIR, POOL_DIR, EMB_DIR, CLF_DIR, EVAL_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

FOLDS = ['03', '05', '06', '07', '09', '10', '11', '14', '16', '17']
TEST_PAC = [('aperio', 'A04'), ('hamamatsu', 'H04')]
POOL_PAC_NUMS = [n for n in P.TODOS if n != '04']
HELDOUT_FOLDS = FOLDS
RFDETR_CKPT = f'{WORKSPACE}/Runs_TFM/rfdetr/fold_{f}_leve_hsv/checkpoint_best_total.pth'
TEST_ROOTS = {'20': Path(f'{WORKSPACE}/Test_A04_overlap20'),
              '50': Path(f'{WORKSPACE}/Test_A04_overlap50')}
TAU_DET = 0.20
DIST_UM = 7.5
HALF = P.PATCH_SIZE / 2

FM_REGISTRY = {
    'virchow':  {'repo': 'paige-ai/Virchow',  'loader': 'timm'},
    'virchow2': {'repo': 'paige-ai/Virchow2', 'loader': 'timm'},
}
CROPS = [64, 96, 128]
REF_MPP = 0.25

def gt_boxes(scanner, patient):
    r = P.bbox_radius_for(scanner)
    out = {}
    for fr in P.load_patient_frames(scanner, patient):
        mit = [(x - r, y - r, x + r, y + r) for (x, y, c, i) in fr['mit_pts']]
        notb = [(x - r, y - r, x + r, y + r) for (x, y, c) in fr['not_pts']]
        out[fr['frame_stem']] = (mit, notb)
    return out

def dist_px(scanner):
    return DIST_UM / P.MPP_BY_SCANNER[scanner]

def cmd_candidatos(args):
    E.MATCH_MODE = 'iou'; E.DEDUP_IOU = 0.30
    for ov, troot in TEST_ROOTS.items():
        for f in FOLDS:
            out = CAND_DIR / f'A04_fold{f}_ov{ov}.csv'
            if out.exists() and not args.force:
                print(f'  ya existe {out.name}'); continue
            ckpt = RFDETR_CKPT.format(f=f)
            filas = []
            for sc, pac in TEST_PAC:
                vdir = troot / 'mitos14' / sc / pac / 'sliding_window'
                man = {r['patch_filename']: r for r in csv.DictReader(open(vdir / '_manifest.csv'))}
                imgs = sorted((vdir / 'images').glob('*.jpg'))
                if args.limit:
                    imgs = imgs[:args.limit]
                print(f'  [fold {f} ov{ov}] {sc}/{pac}: {len(imgs)} teselas…', flush=True)
                preds = E.predecir('rfdetr', ckpt, imgs)
                porframe = defaultdict(list)
                for fname, boxes in preds.items():
                    m = man.get(fname)
                    if not m:
                        continue
                    ox = float(m['patch_center_x']) - HALF
                    oy = float(m['patch_center_y']) - HALF
                    for (x1, y1, x2, y2, c) in boxes:
                        porframe[m['frame_stem']].append((ox + x1, oy + y1, ox + x2, oy + y2, c))
                for frame, bx in porframe.items():
                    for (x1, y1, x2, y2, c) in E.dedup(bx, sc):
                        if c < TAU_DET:
                            continue
                        filas.append({'fold': f, 'overlap': ov, 'scanner': sc, 'frame': frame,
                                      'cx': round((x1 + x2) / 2, 2), 'cy': round((y1 + y2) / 2, 2),
                                      'x1': round(x1, 2), 'y1': round(y1, 2),
                                      'x2': round(x2, 2), 'y2': round(y2, 2),
                                      'score': round(c, 4)})
            with open(out, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=['fold', 'overlap', 'scanner', 'frame',
                                                   'cx', 'cy', 'x1', 'y1', 'x2', 'y2', 'score'])
                w.writeheader(); w.writerows(filas)
            print(f'  -> {out.name}: {len(filas)} candidatos (conf>={TAU_DET})', flush=True)

def _origenes(L, stride=P.PATCH_SIZE):
    if L <= P.PATCH_SIZE:
        return [0]
    pos = list(range(0, L - P.PATCH_SIZE + 1, stride))
    if pos[-1] != L - P.PATCH_SIZE:
        pos.append(L - P.PATCH_SIZE)
    return pos

def _minar_fp_paciente(num, fold, rng):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    E.MATCH_MODE = 'iou'; E.DEDUP_IOU = 0.30
    ckpt = RFDETR_CKPT.format(f=fold)
    tmp = ROOT / '_tiles_tmp' / f'{num}'
    fp = []
    for sc, pre in [('aperio', 'A'), ('hamamatsu', 'H')]:
        pac = f'{pre}{num}'; r = P.bbox_radius_for(sc); thr = dist_px(sc)
        gt = gt_boxes(sc, pac)
        for fr in P.load_patient_frames(sc, pac):
            import shutil
            shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True, exist_ok=True)
            img = Image.open(fr['img_path']).convert('RGB'); W, H = img.size
            meta = {}
            for y0 in _origenes(H):
                for x0 in _origenes(W):
                    nm = f'{x0:05d}_{y0:05d}.jpg'
                    img.crop((x0, y0, x0 + P.PATCH_SIZE, y0 + P.PATCH_SIZE)).save(tmp / nm, quality=92)
                    meta[nm] = (x0, y0)
            img.close()
            preds = E.predecir('rfdetr', ckpt, sorted(tmp.glob('*.jpg')))
            boxes = []
            for nm, bx in preds.items():
                ox, oy = meta[nm]
                for (x1, y1, x2, y2, c) in bx:
                    boxes.append((ox + x1, oy + y1, ox + x2, oy + y2, c))
            mitb, _ = gt.get(fr['frame_stem'], ([], []))
            for (x1, y1, x2, y2, c) in E.dedup(boxes, sc):
                if c < TAU_DET:
                    continue
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                casa = any(((cx - (g[0] + g[2]) / 2) ** 2 + (cy - (g[1] + g[3]) / 2) ** 2) ** 0.5 <= thr
                           for g in mitb)
                if not casa:
                    fp.append({'scanner': sc, 'frame': fr['frame_stem'],
                               'cx': round(cx, 2), 'cy': round(cy, 2), 'label': 'neg',
                               'source': 'mined_fp'})
        import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return fp

def cmd_pool(args):
    rng = random.Random(SEED)
    filas = []
    for num in POOL_PAC_NUMS:
        for sc, pre in [('aperio', 'A'), ('hamamatsu', 'H')]:
            pac = f'{pre}{num}'
            for fr in P.load_patient_frames(sc, pac):
                for (x, y, c, i) in fr['mit_pts']:
                    filas.append({'scanner': sc, 'frame': fr['frame_stem'],
                                  'cx': round(x, 2), 'cy': round(y, 2), 'label': 'mit', 'source': 'gt_mit'})
                for (x, y, c) in fr['not_pts']:
                    filas.append({'scanner': sc, 'frame': fr['frame_stem'],
                                  'cx': round(x, 2), 'cy': round(y, 2), 'label': 'neg', 'source': 'not_mit'})
        print(f'  positivos/not_mit de paciente {num} acumulados ({len(filas)})', flush=True)
    if not args.sin_mineria:
        for num, fold in zip(HELDOUT_FOLDS, HELDOUT_FOLDS):
            print(f'  minando FP del paciente {num} (fold {fold})…', flush=True)
            filas += _minar_fp_paciente(num, fold, rng)
    out = POOL_DIR / 'pool.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['scanner', 'frame', 'cx', 'cy', 'label', 'source'])
        w.writeheader(); w.writerows(filas)
    from collections import Counter
    print('POOL:', dict(Counter(f"{r['label']}/{r['source']}" for r in filas)))
    print('->', out)

def _frame_path(scanner, frame):
    pac = ('A' if scanner == 'aperio' else 'H') + frame[1:3]
    name = 'Aperio' if scanner == 'aperio' else 'Hamamatsu'
    base = Path(f'{WORKSPACE}/Dataset_Mitosis14_TUPAC16')
    for sub in (f'Training {name}', f'Testing {name}'):
        p = base / sub / pac / pac / 'frames' / 'x40' / f'{frame}.tiff'
        if p.exists():
            return p
    return None

def _crop_reflect(arr, cx, cy, r):
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = int(round(cx - r)), int(round(cy - r)), int(round(cx + r)), int(round(cy + r))
    px0, py0 = max(0, -x0), max(0, -y0)
    px1, py1 = max(0, x1 - W), max(0, y1 - H)
    sub = arr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if sub.size == 0:
        return np.zeros((2 * r, 2 * r, 3), np.uint8)
    if px0 or py0 or px1 or py1:
        sub = np.pad(sub, ((py0, py1), (px0, px1), (0, 0)), mode='reflect')
    return sub

def cargar_fm(fm):
    import torch, timm
    from timm.layers import SwiGLUPacked
    from timm.data import resolve_data_config, create_transform
    repo = FM_REGISTRY[fm]['repo']
    model = timm.create_model(f"hf-hub:{repo}", pretrained=True,
                              mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().cuda()
    proc = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, proc

def _embed_batch(model, proc, pil_list):
    import torch
    batch = torch.stack([proc(p) for p in pil_list]).cuda()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        out = model(batch)
    if out.dim() == 3:
        emb = torch.cat([out[:, 0], out[:, -256:].mean(1)], dim=-1)
    else:
        emb = out
    return emb.float().cpu().numpy()

def _puntos_necesarios():
    pts = {}
    pool = list(csv.DictReader(open(POOL_DIR / 'pool.csv')))
    for r in pool:
        pts[(r['scanner'], r['frame'], r['cx'], r['cy'])] = True
    for cf in CAND_DIR.glob('A04_*_ov20.csv'):
        for r in csv.DictReader(open(cf)):
            pts[(r['scanner'], r['frame'], r['cx'], r['cy'])] = True
    return list(pts)

def cmd_embeddings(args):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    fm, crop = args.fm, args.crop
    cache = EMB_DIR / f'{fm}_crop{crop}.npz'
    if cache.exists() and not args.force:
        print('  ya existe', cache.name); return
    model, proc = cargar_fm(fm)
    pts = _puntos_necesarios()
    porframe = defaultdict(list)
    for (sc, frame, cx, cy) in pts:
        porframe[(sc, frame)].append((cx, cy))
    keys, vecs = [], []
    rad = {sc: round((crop / 2) * REF_MPP / P.MPP_BY_SCANNER[sc]) for sc in P.MPP_BY_SCANNER}
    t0 = time.time(); done = 0
    batch_pil, batch_key = [], []

    def flush():
        nonlocal batch_pil, batch_key
        if not batch_pil:
            return
        v = _embed_batch(model, proc, batch_pil)
        keys.extend(batch_key); vecs.append(v)
        batch_pil, batch_key = [], []

    for (sc, frame), cents in porframe.items():
        fp = _frame_path(sc, frame)
        if fp is None:
            continue
        arr = np.asarray(Image.open(fp).convert('RGB'))
        r = rad[sc]
        for (cx, cy) in cents:
            sub = _crop_reflect(arr, float(cx), float(cy), r)
            pil = Image.fromarray(sub).resize((224, 224), Image.BICUBIC)
            batch_pil.append(pil)
            batch_key.append(f'{sc}|{frame}|{cx}|{cy}')
            if len(batch_pil) >= 64:
                flush()
        done += 1
        if done % 50 == 0:
            print(f'  [{fm} crop{crop}] {done}/{len(porframe)} frames, {len(keys)} embeddings, '
                  f'{time.time()-t0:.0f}s', flush=True)
    flush()
    X = np.concatenate(vecs, 0) if vecs else np.zeros((0, 1))
    np.savez_compressed(cache, keys=np.array(keys), X=X.astype(np.float16))
    print(f'-> {cache.name}: {X.shape} embeddings en {(time.time()-t0)/60:.1f} min')
    import torch
    del model
    torch.cuda.empty_cache()

def _load_emb(fm, crop):
    d = np.load(EMB_DIR / f'{fm}_crop{crop}.npz', allow_pickle=True)
    keys = list(d['keys']); X = d['X'].astype(np.float32)
    return {k: X[i] for i, k in enumerate(keys)}

def _patient_of(scanner, frame):
    return frame[1:3]

def cmd_entrenar(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    import joblib
    fm, crop = args.fm, args.crop
    emb = _load_emb(fm, crop)
    pool = list(csv.DictReader(open(POOL_DIR / 'pool.csv')))
    X, y, grp = [], [], []
    for r in pool:
        k = f"{r['scanner']}|{r['frame']}|{r['cx']}|{r['cy']}"
        if k not in emb:
            continue
        X.append(emb[k]); y.append(1 if r['label'] == 'mit' else 0)
        grp.append(_patient_of(r['scanner'], r['frame']))
    X = np.array(X); y = np.array(y); grp = np.array(grp)
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tr, va in gkf.split(X, y, grp):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0))
        clf.fit(X[tr], y[tr]); oof[va] = clf.predict_proba(X[va])[:, 1]
    auc = roc_auc_score(y, oof)
    thrs = np.linspace(0.05, 0.95, 19)
    thr_best = float(max(thrs, key=lambda t: f1_score(y, (oof >= t).astype(int))))
    yhat = (oof >= thr_best).astype(int)
    clf_p = precision_score(y, yhat); clf_r = recall_score(y, yhat); clf_f1 = f1_score(y, yhat)
    metr = {'fm': fm, 'crop': crop, 'n': int(len(y)), 'n_pos': int(y.sum()), 'n_neg': int((y == 0).sum()),
            'clf_thr': round(thr_best, 3), 'clf_precision': round(float(clf_p), 4),
            'clf_recall': round(float(clf_r), 4), 'clf_f1': round(float(clf_f1), 4),
            'clf_auc': round(float(auc), 4)}
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0))
    clf.fit(X, y)
    joblib.dump(clf, CLF_DIR / f'{fm}_crop{crop}.joblib')
    (CLF_DIR / f'{fm}_crop{crop}.json').write_text(json.dumps(metr, indent=2))
    print(f"[CLASIFICADOR {fm} crop{crop}] n={metr['n']} (pos={metr['n_pos']}/neg={metr['n_neg']}) | "
          f"AUC={auc:.4f} P={clf_p:.3f} R={clf_r:.3f} F1={clf_f1:.3f} @thr={thr_best:.2f}")

def _metric(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1

def _match_fold(cands, gt, scanner_dist, modo):
    pass

def cmd_evaluar(args):
    import joblib
    fm, crop = args.fm, args.crop
    clf = joblib.load(CLF_DIR / f'{fm}_crop{crop}.joblib')
    emb = _load_emb(fm, crop)
    GT = {}
    for sc, pac in TEST_PAC:
        GT[sc] = gt_boxes(sc, pac)

    taus = [round(0.05 * i, 2) for i in range(1, 20)]
    confs_det = [0.20, 0.30, 0.50]
    resultados = []

    for ov in ['20']:
        for crit in ['dist', 'iou']:
            for cdet in confs_det:
                base_PRF = []; casc_best = []; casc_fixed = []
                agg_base = [0, 0, 0, 0, 0]
                agg_tau = {t: [0, 0, 0, 0, 0] for t in taus}
                for f in FOLDS:
                    cf = CAND_DIR / f'A04_fold{f}_ov{ov}.csv'
                    cands = [r for r in csv.DictReader(open(cf)) if float(r['score']) >= cdet]
                    for r in cands:
                        k = f"{r['scanner']}|{r['frame']}|{r['cx']}|{r['cy']}"
                        r['pmit'] = float(clf.predict_proba(emb[k][None])[0, 1]) if k in emb else 1.0
                    def contar(keep_fn):
                        tp = fp = fn = fph = fpe = 0
                        porframe = defaultdict(list)
                        for r in cands:
                            if keep_fn(r):
                                porframe[(r['scanner'], r['frame'])].append(r)
                        frames = set((sc, fr) for sc in GT for fr in GT[sc])
                        for (sc, fr) in frames:
                            mitb, notb = GT[sc].get(fr, ([], []))
                            thr = dist_px(sc)
                            preds = sorted(porframe.get((sc, fr), []), key=lambda r: -float(r['score']))
                            usado = [False] * len(mitb)
                            for r in preds:
                                cx, cy = float(r['cx']), float(r['cy'])
                                pb = (float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']))
                                best = -1; bestq = None
                                for gi, gb in enumerate(mitb):
                                    if usado[gi]:
                                        continue
                                    if crit == 'iou':
                                        ok = E.iou(pb, gb) >= 0.50; q = E.iou(pb, gb)
                                    else:
                                        gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                                        d = ((cx - gcx) ** 2 + (cy - gcy) ** 2) ** 0.5
                                        ok = d <= thr; q = -d
                                    if ok and (bestq is None or q > bestq):
                                        best, bestq = gi, q
                                if best >= 0:
                                    usado[best] = True; tp += 1
                                else:
                                    fp += 1
                                    hard = False
                                    for nb in notb:
                                        ncx, ncy = (nb[0] + nb[2]) / 2, (nb[1] + nb[3]) / 2
                                        if ((cx - ncx) ** 2 + (cy - ncy) ** 2) ** 0.5 <= thr:
                                            hard = True; break
                                    fph += hard; fpe += (not hard)
                            fn += len(mitb) - sum(usado)
                        return tp, fp, fn, fph, fpe
                    b = contar(lambda r: True)
                    for i in range(5):
                        agg_base[i] += b[i]
                    for t in taus:
                        c = contar(lambda r, t=t: r['pmit'] >= t)
                        for i in range(5):
                            agg_tau[t][i] += c[i]
                pb, rb, f1b = _metric(agg_base[0], agg_base[1], agg_base[2])
                best_t = max(taus, key=lambda t: _metric(agg_tau[t][0], agg_tau[t][1], agg_tau[t][2])[2])
                pc, rc, f1c = _metric(agg_tau[best_t][0], agg_tau[best_t][1], agg_tau[best_t][2])
                pf, rf_, f1f = _metric(agg_tau[0.5][0], agg_tau[0.5][1], agg_tau[0.5][2])
                resultados.append({
                    'fm': fm, 'crop': crop, 'overlap': ov, 'criterio': crit, 'conf_det': cdet,
                    'base_P': round(pb, 4), 'base_R': round(rb, 4), 'base_F1': round(f1b, 4),
                    'base_FPhard': agg_base[3], 'base_FPempty': agg_base[4],
                    'casc_tau_bestF1': best_t,
                    'casc_P': round(pc, 4), 'casc_R': round(rc, 4), 'casc_F1': round(f1c, 4),
                    'casc_FPhard': agg_tau[best_t][3], 'casc_FPempty': agg_tau[best_t][4],
                    'casc_tau0.5_P': round(pf, 4), 'casc_tau0.5_R': round(rf_, 4), 'casc_tau0.5_F1': round(f1f, 4),
                })
                print(f'  [{fm} c{crop} ov{ov} {crit} det{cdet}] base F1={f1b:.3f}(R{rb:.3f}) '
                      f'-> casc F1={f1c:.3f}(R{rc:.3f})@tau{best_t} FPhard {agg_base[3]}->{agg_tau[best_t][3]}',
                      flush=True)
    out = EVAL_DIR / f'{fm}_crop{crop}.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        w.writeheader(); w.writerows(resultados)
    print('->', out)

def cmd_tabla(args):
    filas = []
    for cf in sorted(EVAL_DIR.glob('*_crop*.csv')):
        filas += list(csv.DictReader(open(cf)))
    if not filas:
        print('No hay evaluaciones todavía.'); return
    out = ROOT / 'tabla_comparativa.csv'
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    val = {}
    for jf in CLF_DIR.glob('*.json'):
        d = json.loads(jf.read_text()); val[(d['fm'], d['crop'])] = d.get('val_f1_clasif')
    L = ['DOS FASES: RF-DETR + FM (frozen) — comparativa', '=' * 60]
    L.append('Selección de ganador por F1 de clasificación en CV agrupada (no toca A04).')
    L.append(f"{'fm':14}{'crop':>5}{'valF1clf':>9}")
    for (fm, crop), v in sorted(val.items(), key=lambda x: -(x[1] or 0)):
        L.append(f"{fm:14}{crop:>5}{(v or 0):>9.4f}")
    L.append('')
    L.append('Detección A04 (distancia 7.5µm, solape 20, conf_det 0.30): base vs dos fases')
    for r in filas:
        if r['criterio'] == 'dist' and r['overlap'] == '20' and float(r['conf_det']) == 0.30:
            L.append(f"  {r['fm']:12} c{r['crop']:>3}: F1 {r['base_F1']}->{r['casc_F1']} "
                     f"R {r['base_R']}->{r['casc_R']} FPhard {r['base_FPhard']}->{r['casc_FPhard']}")
    (ROOT / 'RESUMEN_DOS_FASES.txt').write_text('\n'.join(L))
    print('\n'.join(L)); print('->', out)

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ['candidatos', 'pool', 'tabla']:
        sp = sub.add_parser(name)
        sp.add_argument('--force', action='store_true')
        sp.add_argument('--limit', type=int, default=0)
        sp.add_argument('--sin-mineria', action='store_true')
    for name in ['embeddings', 'entrenar', 'evaluar']:
        sp = sub.add_parser(name)
        sp.add_argument('--fm', required=True, choices=list(FM_REGISTRY))
        sp.add_argument('--crop', type=int, required=True, choices=CROPS)
        sp.add_argument('--force', action='store_true')
        sp.add_argument('--limit', type=int, default=0)
        sp.add_argument('--sin-mineria', action='store_true')
    args = ap.parse_args()
    {'candidatos': cmd_candidatos, 'pool': cmd_pool, 'embeddings': cmd_embeddings,
     'entrenar': cmd_entrenar, 'evaluar': cmd_evaluar, 'tabla': cmd_tabla}[args.cmd](args)

if __name__ == '__main__':
    main()
