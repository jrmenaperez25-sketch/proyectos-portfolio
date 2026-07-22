#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_REPO)):
    _p = os.path.join(_REPO, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import cross_scanner as XS
import parches_mitos as PM
import parches_tupac as PT
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

OUT = Path(f'{WORKSPACE}/Cross_Scanner'); OUT.mkdir(exist_ok=True)
CONF_DET = 0.30
FOLDS = XS.FOLDS

PIPES = [
    ('MITOS', f'{WORKSPACE}/segunda_fase_mitos/lora/virchow_crop64_model.pt', 0.20,
     [('rfdetr', f'{WORKSPACE}/Runs_TFM/rfdetr/fold_{f}_leve_hsv/checkpoint_best_total.pth', f) for f in FOLDS]),
    ('TUPAC', f'{WORKSPACE}/segunda_fase_tupac/clasificadores/lora_model.pt', 0.85,
     [('rfdetr', f'{WORKSPACE}/EXPERIMENTO_TUPAC73/run/rfdetr_ovs/checkpoint_best_total.pth', 'unico')]),
]

def cargar_virchow():
    import timm
    from timm.layers import SwiGLUPacked
    from timm.data import resolve_data_config, create_transform
    base = timm.create_model('hf-hub:paige-ai/Virchow', pretrained=True,
                             mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU).eval().cuda()
    proc = create_transform(**resolve_data_config(base.pretrained_cfg, model=base))
    return base, proc

class MitClf(nn.Module):
    def __init__(self, base, dim=2560):
        super().__init__(); self.base = base; self.head = nn.Linear(dim, 1)
    def forward(self, x):
        out = self.base(x)
        feat = torch.cat([out[:, 0], out[:, -256:].mean(1)], dim=-1)
        return self.head(feat).squeeze(-1)

def load_clf(model_path):
    from peft import LoraConfig, get_peft_model
    base, proc = cargar_virchow()
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=['qkv'], lora_dropout=0.1, bias='none')
    base = get_peft_model(base, cfg, autocast_adapter_dtype=False)
    model = MitClf(base).cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    return model, proc

def _crop_reflect(arr, cx, cy, r):
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = int(round(cx-r)), int(round(cy-r)), int(round(cx+r)), int(round(cy+r))
    px0, py0 = max(0, -x0), max(0, -y0); px1, py1 = max(0, x1-W), max(0, y1-H)
    sub = arr[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    if sub.size == 0:
        return np.zeros((2*r, 2*r, 3), np.uint8)
    if px0 or py0 or px1 or py1:
        sub = np.pad(sub, ((py0, py1), (px0, px1), (0, 0)), mode='reflect')
    return sub

def añadir_probs(model, proc, cands_por_frame, frame_info):
    for frame, cands in cands_por_frame.items():
        if not cands:
            continue
        ip, mpp = frame_info[frame]; rad = round(32 * 0.25 / mpp)
        arr = np.asarray(Image.open(ip).convert('RGB'))
        tens = []
        for c in cands:
            sub = _crop_reflect(arr, c['cx'], c['cy'], rad)
            tens.append(proc(Image.fromarray(sub).resize((224, 224), Image.BICUBIC)))
        probs = []
        for i in range(0, len(tens), 64):
            xb = torch.stack(tens[i:i+64]).cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                probs.extend(torch.sigmoid(model(xb)).float().cpu().tolist())
        for c, p in zip(cands, probs):
            c['prob'] = p

def prf(tp, fp, fn):
    P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
    return round(P, 4), round(R, 4), round(2*P*R/(P+R), 4) if P+R else 0.0

def match(cands_por_frame, gt, dpx, keep):
    tp = fp = 0; NG = sum(len(v) for v in gt.values())
    for frame, gts in gt.items():
        preds = sorted([c for c in cands_por_frame.get(frame, []) if keep(c)], key=lambda c: -c['score'])
        usado = [False]*len(gts)
        for c in preds:
            best, bd = -1, dpx
            for gi, gb in enumerate(gts):
                if usado[gi]:
                    continue
                d = ((c['cx']-(gb[0]+gb[2])/2)**2 + (c['cy']-(gb[1]+gb[3])/2)**2)**0.5
                if d <= bd:
                    bd = d; best = gi
            if best >= 0:
                usado[best] = True; tp += 1
            else:
                fp += 1
    return prf(tp, fp, NG - tp)

def build_targets():
    T = {}
    for sc, pac in [('aperio', 'A04'), ('hamamatsu', 'H04')]:
        tiles, man = XS.tiles_mitos(sc, pac); gt, dpx = XS.gt_mitos(sc, pac)
        finfo = {fr['frame_stem']: (fr['img_path'], PM.MPP_BY_SCANNER[sc]) for fr in PM.load_patient_frames(sc, pac)}
        T[('MITOS', sc)] = [(tiles, man, gt, dpx, finfo)]
    for grp, casos in [('aperio', XS.TUPAC_APERIO), ('leica', XS.TUPAC_LEICA)]:
        T[('TUPAC', grp)] = []
        for caso in casos:
            tiles, man = XS.tiles_tupac(caso); gt, dpx = XS.gt_tupac(caso)
            finfo = {fr['frame_stem']: (str(fr['img_path']), PT.MPP) for fr in PT.load_case_frames(caso)}
            T[('TUPAC', grp)].append((tiles, man, gt, dpx, finfo))
    return T

def main():
    targets = build_targets()
    todos = [('MITOS', 'aperio'), ('MITOS', 'hamamatsu'), ('TUPAC', 'aperio'), ('TUPAC', 'leica')]
    filas = []
    for pipe, clf_path, thr, dets in PIPES:
        print(f'### PIPELINE {pipe} (Virchow thr {thr}) ###', flush=True)
        clf, proc = load_clf(clf_path)
        for (modelo, ckpt, fold) in dets:
            mod, m = XS.cargar(modelo, ckpt)
            for tgt in todos:
                cands_all = {}; gt_all = {}; finfo_all = {}; dpx = None
                for (tiles, man, gt, d, finfo) in targets[tgt]:
                    dpx = d; gt_all.update(gt); finfo_all.update(finfo)
                    porframe = XS.reproyectar(XS.predecir(mod, m, tiles), man)
                    for frame, boxes in porframe.items():
                        ded = XS.dedup([b for b in boxes if b[4] >= 0.2], d)
                        cands_all[frame] = [{'cx': (b[0]+b[2])/2, 'cy': (b[1]+b[3])/2, 'score': b[4]} for b in ded]
                añadir_probs(clf, proc, cands_all, finfo_all)
                bP, bR, bF = match(cands_all, gt_all, dpx, lambda c: c['score'] >= CONF_DET)
                cP, cR, cF = match(cands_all, gt_all, dpx, lambda c: c['score'] >= CONF_DET and c.get('prob', 1.0) >= thr)
                indom = (pipe == tgt[0])
                filas.append({'pipeline': pipe, 'fold': fold, 'tipo': 'in-domain' if indom else 'CROSS',
                              'target_dataset': tgt[0], 'target_scanner': tgt[1],
                              'base_P': bP, 'base_R': bR, 'base_F1': bF,
                              'casc_P': cP, 'casc_R': cR, 'casc_F1': cF})
                print(f'  {pipe} f{fold} -> {tgt[0]}/{tgt[1]} [{"in" if indom else "X"}]: base F1={bF} (R{bR}) -> dos fases F1={cF} (R{cR})', flush=True)
    with open(OUT / 'dos_fases_cross.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys())); w.writeheader(); w.writerows(filas)
    print(f'GUARDADO {OUT/"dos_fases_cross.csv"}', flush=True)

if __name__ == '__main__':
    main()
