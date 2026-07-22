#!/usr/bin/env python3
from __future__ import annotations
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SEED = 42
PATCH_SIZE = 512
HALF = PATCH_SIZE // 2
ANGLES = [0, 36, 72, 108, 144]
FLIP_OPTIONS = ['none', 'h', 'v']
ALL_COMBOS = [(a, f) for a in ANGLES for f in FLIP_OPTIONS]

SCANNER = 'tupac'
MPP = 0.25
BBOX_RADIUS_PX = 25
N_EMPTY_PER_FRAME = 2
MAX_TRIES = 5000
IMAGE_QUALITY = 95
SW_OVERLAP = 0.20

ROOT = Path(f'{WORKSPACE}/EXPERIMENTO_TUPAC73')
GT_ROOT = ROOT / 'gt'
IMG_ROOT = Path(f'{WORKSPACE}/Dataset_Mitosis14_TUPAC16/Tupac16-Only-Training/mitosis imágenes')
VERS = ROOT / 'versiones'
DATASET = ROOT / 'dataset'
SPLIT = json.loads((ROOT / 'split' / 'split.json').read_text())

def bbox_radius_for(_sc=SCANNER) -> int:
    return BBOX_RADIUS_PX

def angle_margin(deg: float) -> int:
    r = math.radians(deg)
    return int(math.ceil(PATCH_SIZE / 2 * (abs(math.cos(r)) + abs(math.sin(r))))) + 1

def extract_patch_img(img, cx_c, cy_c, angle_deg, flip):
    r = math.radians(angle_deg); cos_a, sin_a = math.cos(r), math.sin(r); half = PATCH_SIZE / 2
    a = cos_a; b = -sin_a; c = -cos_a * half + sin_a * half + cx_c
    d = sin_a; e = cos_a;  f = -sin_a * half - cos_a * half + cy_c
    patch = img.transform((PATCH_SIZE, PATCH_SIZE), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    if flip == 'h':
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == 'v':
        patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
    return patch

def world_to_patch(wx, wy, cx_c, cy_c, angle_deg, flip):
    r = math.radians(angle_deg); cos_a, sin_a = math.cos(r), math.sin(r)
    dx, dy = wx - cx_c, wy - cy_c
    pu = cos_a * dx + sin_a * dy + HALF
    pv = -sin_a * dx + cos_a * dy + HALF
    if flip == 'h':
        pu = PATCH_SIZE - pu
    elif flip == 'v':
        pv = PATCH_SIZE - pv
    return pu, pv

def find_center_for_point(W, H, px, py, angle_deg, flip, bbox_r, rng, max_tries=MAX_TRIES):
    r = math.radians(angle_deg); cos_a, sin_a = math.cos(r), math.sin(r); half = PATCH_SIZE / 2
    margin = angle_margin(angle_deg); lo, hi = float(bbox_r), float(PATCH_SIZE - bbox_r)
    for _ in range(max_tries):
        mx = rng.uniform(lo, hi); my = rng.uniform(lo, hi)
        mx_ = PATCH_SIZE - mx if flip == 'h' else mx
        my_ = PATCH_SIZE - my if flip == 'v' else my
        dx, dy = mx_ - half, my_ - half
        cx_c = px - cos_a * dx + sin_a * dy
        cy_c = py - sin_a * dx - cos_a * dy
        if margin <= cx_c <= W - margin and margin <= cy_c <= H - margin:
            return cx_c, cy_c
    return None

def find_random_empty_center(W, H, angle_deg, rng):
    margin = angle_margin(angle_deg)
    return rng.uniform(margin, W - margin), rng.uniform(margin, H - margin)

def annotations_in_patch(mit_pts, not_pts, cx_c, cy_c, angle_deg, flip, bbox_r):
    def project(pts, cat_id, is_mit):
        out = []
        for tup in pts:
            wx, wy, conf = tup[0], tup[1], tup[2]
            pu, pv = world_to_patch(wx, wy, cx_c, cy_c, angle_deg, flip)
            if 0 <= pu <= PATCH_SIZE and 0 <= pv <= PATCH_SIZE:
                x1 = max(0.0, pu - bbox_r); y1 = max(0.0, pv - bbox_r)
                x2 = min(float(PATCH_SIZE), pu + bbox_r); y2 = min(float(PATCH_SIZE), pv + bbox_r)
                out.append({'world_x': round(wx, 2), 'world_y': round(wy, 2), 'conf': conf,
                            'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                            'category_id': cat_id,
                            'punto_id': tup[3] if is_mit else f'not_{wx:.2f}_{wy:.2f}',
                            'mitosis_id': tup[3] if is_mit else ''})
        return out
    return project(mit_pts, 1, True), project(not_pts, 2, False)

def read_csv_coords(path: Path) -> List[Tuple[float, float]]:
    coords = []
    if not path.exists():
        return coords
    for row in csv.reader(open(path, newline='')):
        row = [c.strip() for c in row if c.strip()]
        if len(row) >= 2:
            try:
                coords.append((float(row[0]), float(row[1])))
            except ValueError:
                pass
    return coords

def load_case_frames(caso: str) -> List[Dict[str, Any]]:
    cdir = IMG_ROOT / caso
    if not cdir.exists():
        return []
    gtd = GT_ROOT / caso
    frames = []
    for img_path in sorted(cdir.glob('*.tif')):
        hh = img_path.stem
        with Image.open(img_path) as _im:
            W, H = _im.size
        mit_raw = read_csv_coords(gtd / f'{hh}_mitosis.csv')
        not_raw = read_csv_coords(gtd / f'{hh}_not_mitosis.csv')
        stem = f'{caso}_{hh}'
        mit_pts = [(x, y, None, f'mit_tupac_{caso}_{hh}_{x:.2f}_{y:.2f}') for (x, y) in mit_raw]
        not_pts = [(x, y, None) for (x, y) in not_raw]
        frames.append({'img_path': img_path, 'W': W, 'H': H, 'dataset': 'tupac16',
                       'scanner': SCANNER, 'patient': caso, 'frame_stem': stem,
                       'mit_pts': mit_pts, 'not_pts': not_pts,
                       'is_empty': not mit_pts and not not_pts})
    return frames

def _spec(fr, final_class, mitosis_id, wx, wy, conf, cx_c, cy_c, angle, flip, rep_idx, source_index):
    return {'scanner': SCANNER, 'patient': fr['patient'], 'frame_stem': fr['frame_stem'],
            'frame': fr, 'final_class': final_class, 'mitosis_id': mitosis_id,
            'source_world_x': wx, 'source_world_y': wy, 'source_conf': conf,
            'cx_c': cx_c, 'cy_c': cy_c, 'angle': angle, 'flip': flip,
            'representative_index': rep_idx, 'source_index': source_index}

def patch_filename(version, sp):
    base = f'{sp["frame_stem"]}_{sp["final_class"]}_{sp["source_index"]}'
    if version == 'oversampling':
        base += f'_a{sp["angle"]}_f{sp["flip"]}'
    return base + '.jpg'

def _clusterizar(items, maxspan):
    clusters = [{'idx': [i], 'minx': it['x'], 'maxx': it['x'], 'miny': it['y'], 'maxy': it['y']}
                for i, it in enumerate(items)]
    cabe = lambda a, b: (max(a['maxx'], b['maxx']) - min(a['minx'], b['minx']) <= maxspan and
                         max(a['maxy'], b['maxy']) - min(a['miny'], b['miny']) <= maxspan)
    cen = lambda c: ((c['minx'] + c['maxx']) / 2, (c['miny'] + c['maxy']) / 2)
    while True:
        mejor = None
        for i in range(len(clusters)):
            xi, yi = cen(clusters[i])
            for j in range(i + 1, len(clusters)):
                if not cabe(clusters[i], clusters[j]):
                    continue
                xj, yj = cen(clusters[j]); d = (xi - xj) ** 2 + (yi - yj) ** 2
                if mejor is None or d < mejor[0]:
                    mejor = (d, i, j)
        if mejor is None:
            break
        _, i, j = mejor; a, b = clusters[i], clusters[j]
        clusters[i] = {'idx': a['idx'] + b['idx'], 'minx': min(a['minx'], b['minx']),
                       'maxx': max(a['maxx'], b['maxx']), 'miny': min(a['miny'], b['miny']),
                       'maxy': max(a['maxy'], b['maxy'])}
        clusters.pop(j)
    return clusters

def gen_coverage(frames, rng):
    specs = []; bbox_r = bbox_radius_for(); maxspan = PATCH_SIZE - 2 * bbox_r; half = PATCH_SIZE / 2
    for fr in frames:
        items = [{'x': wx, 'y': wy, 'conf': c, 'clase': 'mitosis', 'src': i + 1, 'pid': mid}
                 for i, (wx, wy, c, mid) in enumerate(fr['mit_pts'])]
        items += [{'x': wx, 'y': wy, 'conf': c, 'clase': 'not_mitosis', 'src': i + 1,
                   'pid': f'not_{wx:.2f}_{wy:.2f}'} for i, (wx, wy, c) in enumerate(fr['not_pts'])]
        if not items:
            continue
        for cl in _clusterizar(items, maxspan):
            cx_c = min(max((cl['minx'] + cl['maxx']) / 2, half), fr['W'] - half)
            cy_c = min(max((cl['miny'] + cl['maxy']) / 2, half), fr['H'] - half)
            rep = min((items[k] for k in cl['idx']), key=lambda it: (it['clase'] != 'mitosis', it['src']))
            specs.append(_spec(fr, rep['clase'], rep['pid'] if rep['clase'] == 'mitosis' else '',
                               rep['x'], rep['y'], rep['conf'], cx_c, cy_c, 0, 'none', 0, rep['src']))
    return specs

def gen_oversampling(frames, rng, max_combos=len(ALL_COMBOS)):
    specs = []; bbox_r = bbox_radius_for()
    for fr in frames:
        for source_index, (wx, wy, conf, mid) in enumerate(fr['mit_pts'], start=1):
            combos = list(ALL_COMBOS); rng.shuffle(combos); rep_idx = 0
            for (angle, flip) in combos:
                if rep_idx >= max_combos:
                    break
                center = find_center_for_point(fr['W'], fr['H'], wx, wy, angle, flip, bbox_r, rng)
                if center is None:
                    continue
                specs.append(_spec(fr, 'mitosis', mid, wx, wy, conf, center[0], center[1], angle, flip, rep_idx, source_index))
                rep_idx += 1
    for fr in frames:
        for source_index, (wx, wy, conf) in enumerate(fr['not_pts'], start=1):
            combos = list(ALL_COMBOS); rng.shuffle(combos)
            for (angle, flip) in combos:
                center = find_center_for_point(fr['W'], fr['H'], wx, wy, angle, flip, bbox_r, rng)
                if center is None:
                    continue
                mit_in, _ = annotations_in_patch(fr['mit_pts'], [], center[0], center[1], angle, flip, bbox_r)
                if mit_in:
                    continue
                specs.append(_spec(fr, 'not_mitosis', '', wx, wy, conf, center[0], center[1], angle, flip, 0, source_index))
                break
    for fr in frames:
        if not fr['is_empty']:
            continue
        for empty_idx in range(N_EMPTY_PER_FRAME):
            angle = rng.choice(ANGLES); flip = rng.choice(FLIP_OPTIONS)
            cx_c, cy_c = find_random_empty_center(fr['W'], fr['H'], angle, rng)
            specs.append(_spec(fr, 'empty', '', None, None, None, cx_c, cy_c, angle, flip, empty_idx, empty_idx + 1))
    rng.shuffle(specs)
    return specs

def gen_representante_unico(frames, rng):
    specs = []; bbox_r = bbox_radius_for()
    for fr in frames:
        puntos = [('mitosis', wx, wy, c, mid, i + 1) for i, (wx, wy, c, mid) in enumerate(fr['mit_pts'])]
        puntos += [('not_mitosis', wx, wy, c, '', i + 1) for i, (wx, wy, c) in enumerate(fr['not_pts'])]
        for (clase, wx, wy, conf, mid, si) in puntos:
            chosen = None
            for (angle, flip) in ALL_COMBOS:
                center = find_center_for_point(fr['W'], fr['H'], wx, wy, angle, flip, bbox_r, rng)
                if center is not None:
                    chosen = (angle, flip, center); break
            if chosen is None:
                continue
            angle, flip, (cx, cy) = chosen
            specs.append(_spec(fr, clase, mid, wx, wy, conf, cx, cy, angle, flip, 0, si))
    return specs

def gen_sliding_window(frames, overlap):
    specs = []; stride = int(round(PATCH_SIZE * (1 - overlap)))
    for fr in frames:
        def origenes(L):
            if L <= PATCH_SIZE:
                return [0]
            xs = list(range(0, L - PATCH_SIZE + 1, stride))
            if xs[-1] != L - PATCH_SIZE:
                xs.append(L - PATCH_SIZE)
            return xs
        idx = 0
        for ox in origenes(fr['W']):
            for oy in origenes(fr['H']):
                idx += 1
                specs.append(_spec(fr, 'tile', '', ox, oy, None,
                                   ox + PATCH_SIZE / 2, oy + PATCH_SIZE / 2, 0, 'none', 0, idx))
    return specs

def generar_specs(version, frames, seed):
    rng = random.Random(seed)
    if version == 'oversampling':
        return gen_oversampling(frames, rng)
    if version == 'representante_unico':
        return gen_representante_unico(frames, rng)
    if version == 'coverage':
        return gen_coverage(frames, rng)
    if version == 'sliding20':
        return gen_sliding_window(frames, 0.20)
    if version == 'sliding50':
        return gen_sliding_window(frames, 0.50)
    raise ValueError(version)

def extract_and_write(specs, out_dir, caso, version, progreso=True):
    images_dir = out_dir / 'images'
    if images_dir.exists():
        for old in images_dir.glob('*.jpg'):
            old.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)
    images, anns_model, anns_full, manifest_rows = [], [], [], []
    img_id = ann_id_model = ann_id_full = 1
    specs_by_img = defaultdict(list)
    for sp in specs:
        specs_by_img[sp['frame']['img_path']].append(sp)
    for img_path, sp_list in specs_by_img.items():
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        for sp in sp_list:
            fr = sp['frame']; bbox_r = bbox_radius_for()
            patch = extract_patch_img(img, sp['cx_c'], sp['cy_c'], sp['angle'], sp['flip'])
            mit_in, not_in = annotations_in_patch(fr['mit_pts'], fr['not_pts'], sp['cx_c'], sp['cy_c'], sp['angle'], sp['flip'], bbox_r)
            fname = patch_filename(version, sp)
            patch.save(images_dir / fname, quality=IMAGE_QUALITY)
            images.append({'id': img_id, 'file_name': fname, 'width': PATCH_SIZE, 'height': PATCH_SIZE,
                           'dataset': fr['dataset'], 'scanner': fr['scanner'], 'patient': fr['patient'],
                           'frame_stem': fr['frame_stem'], 'final_class': sp['final_class'],
                           'mitosis_id': sp['mitosis_id'], 'source_index': sp['source_index'],
                           'source_world_x': sp['source_world_x'], 'source_world_y': sp['source_world_y'],
                           'angle': sp['angle'], 'flip': sp['flip'], 'representative_index': sp['representative_index'],
                           'patch_center_x': round(sp['cx_c'], 4), 'patch_center_y': round(sp['cy_c'], 4),
                           'n_mit_in_patch': len(mit_in), 'n_not_in_patch': len(not_in),
                           'original_image_width': fr['W'], 'original_image_height': fr['H'], 'mpp_um_per_px': MPP})
            for a in mit_in:
                e = {'id': ann_id_full, 'image_id': img_id, 'category_id': 1, 'bbox': a['bbox'],
                     'area': round(a['bbox'][2] * a['bbox'][3], 2), 'iscrowd': 0,
                     'world_x': a['world_x'], 'world_y': a['world_y'], 'conf': a['conf'], 'mitosis_id': a['mitosis_id']}
                anns_full.append(e); ann_id_full += 1
                e2 = dict(e); e2['id'] = ann_id_model; anns_model.append(e2); ann_id_model += 1
            for a in not_in:
                anns_full.append({'id': ann_id_full, 'image_id': img_id, 'category_id': 2, 'bbox': a['bbox'],
                                  'area': round(a['bbox'][2] * a['bbox'][3], 2), 'iscrowd': 0,
                                  'world_x': a['world_x'], 'world_y': a['world_y'], 'conf': a['conf']})
                ann_id_full += 1
            manifest_rows.append({'image_id': img_id, 'patch_filename': fname, 'scanner': fr['scanner'],
                                  'patient_id': fr['patient'], 'frame_stem': fr['frame_stem'],
                                  'final_class': sp['final_class'], 'mitosis_id': sp['mitosis_id'],
                                  'source_index': sp['source_index'], 'representative_index': sp['representative_index'],
                                  'source_world_x': sp['source_world_x'], 'source_world_y': sp['source_world_y'],
                                  'patch_center_x': round(sp['cx_c'], 4), 'patch_center_y': round(sp['cy_c'], 4),
                                  'angle': sp['angle'], 'flip': sp['flip'], 'original_image_width': fr['W'],
                                  'original_image_height': fr['H'], 'mpp_um_per_px': MPP, 'bbox_radius_px': bbox_r,
                                  'n_mit_in_patch': len(mit_in), 'n_not_in_patch': len(not_in),
                                  'original_image_path': str(img_path)})
            img_id += 1
        img.close()
    cats = [{'id': 1, 'name': 'mitosis'}, {'id': 2, 'name': 'not_mitosis'}]
    (out_dir / '_annotations.coco.json').write_text(json.dumps(
        {'info': {'description': f'{caso}/{version}'}, 'categories': cats[:1], 'images': images, 'annotations': anns_model}))
    (out_dir / '_annotations_full.coco.json').write_text(json.dumps(
        {'info': {'description': f'{caso}/{version} full'}, 'categories': cats, 'images': images, 'annotations': anns_full}))
    if manifest_rows:
        with open(out_dir / '_manifest.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys())); w.writeheader(); w.writerows(manifest_rows)
    return {'n_images': len(images), 'n_mit_ann': len(anns_model), 'n_not_ann': len(anns_full) - len(anns_model)}

ROL = {'train': ['oversampling', 'representante_unico'], 'val': ['coverage'], 'test': ['sliding20', 'sliding50']}

def cmd_generar(args):
    print(f'GENERAR parches TUPAC-23 -> {VERS}\n  split: ' +
          ' | '.join(f'{k}={len(SPLIT[k])}' for k in ('train', 'val', 'test')))
    for split in ('train', 'val', 'test'):
        for version in ROL[split]:
            tot_img = tot_mit = tot_not = 0
            for caso in SPLIT[split]:
                frames = load_case_frames(caso)
                specs = generar_specs(version, frames, SEED)
                out_dir = VERS / caso / version
                out_dir.mkdir(parents=True, exist_ok=True)
                st = extract_and_write(specs, out_dir, caso, version)
                tot_img += st['n_images']; tot_mit += st['n_mit_ann']; tot_not += st['n_not_ann']
            print(f'  [{split:5}] {version:18} {len(SPLIT[split])} casos -> '
                  f'{tot_img:6} parches | {tot_mit:5} ann_mit | {tot_not:5} ann_not')
    print('OK')

def _leer(caso, version):
    d = VERS / caso / version
    coco = json.loads((d / '_annotations.coco.json').read_text())
    anns = defaultdict(list)
    for a in coco['annotations']:
        anns[a['image_id']].append(a)
    return d, coco['images'], anns

def _K_global(casos):
    mit_ids, n_neg = set(), 0
    for caso in casos:
        _, ims, _ = _leer(caso, 'oversampling')
        for im in ims:
            if im['final_class'] == 'mitosis':
                mit_ids.add(im['mitosis_id'])
            else:
                n_neg += 1
    n_mit = len(mit_ids)
    return max(1, math.ceil(n_neg / n_mit)) if n_mit else 1, n_mit, n_neg

def _añadir(out_split, casos, version, filtro=None):
    if out_split.exists():
        shutil.rmtree(out_split)
    out_split.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []; img_id = ann_id = 1
    for caso in casos:
        d, ims, anns = _leer(caso, version)
        src = (d / 'images').resolve()
        for im in ims:
            if filtro and not filtro(im):
                continue
            new = dict(im); new['id'] = img_id
            new['file_name'] = str(src / im['file_name']); new['file_basename'] = im['file_name']
            images.append(new)
            for a in anns.get(im['id'], []):
                na = dict(a); na['id'] = ann_id; na['image_id'] = img_id; annotations.append(na); ann_id += 1
            img_id += 1
    (out_split).mkdir(parents=True, exist_ok=True)
    (out_split / '_annotations.coco.json').write_text(json.dumps(
        {'info': {'description': out_split.name}, 'categories': [{'id': 1, 'name': 'mitosis'}],
         'images': images, 'annotations': annotations}))
    return len(images), len(annotations)

def _yaml(dd):
    (dd / 'data.yaml').write_text(f'path: {dd.resolve()}\ntrain: train\nval: valid\n\nnc: 1\nnames:\n  - mitosis\n')
    def lista(split):
        c = json.loads((dd / split / '_annotations.coco.json').read_text())
        return '\n'.join(im['file_name'] for im in c['images'])
    (dd / 'train_yolo.txt').write_text(lista('train'))
    (dd / 'val_yolo.txt').write_text(lista('valid'))
    (dd / 'dataset_yolo.yaml').write_text(
        f'path: {dd.resolve()}\ntrain: train_yolo.txt\nval: val_yolo.txt\n\nnc: 1\nnames:\n  - mitosis\n')

def cmd_ensamblar(args):
    K, n_mit, n_neg = _K_global(SPLIT['train'])
    fK = lambda im: (im['final_class'] != 'mitosis') or (im['representative_index'] < K)
    out = {}
    for name, version, filt in [('ds_ovs', 'oversampling', fK), ('ds_sinover', 'representante_unico', None)]:
        dd = ROOT / name
        nt = _añadir(dd / 'train', SPLIT['train'], version, filt)
        nv = _añadir(dd / 'valid', SPLIT['val'], 'coverage')
        _añadir(dd / 'test', SPLIT['val'], 'coverage')
        _yaml(dd)
        out[name] = nt
    e20 = _añadir(ROOT / 'eval' / 'test_ov20', SPLIT['test'], 'sliding20')
    e50 = _añadir(ROOT / 'eval' / 'test_ov50', SPLIT['test'], 'sliding50')
    (ROOT / 'dataset_config.json').write_text(json.dumps(
        {'K': K, 'N_mit_train': n_mit, 'N_neg_train': n_neg, 'split': SPLIT,
         'ds_ovs_train': out['ds_ovs'][0], 'ds_sinover_train': out['ds_sinover'][0],
         'test_ov20_tiles': e20[0], 'test_ov50_tiles': e50[0]}, indent=2))
    print(f'ENSAMBLADO  K(train)={K} (N_mit={n_mit}, N_neg={n_neg})')
    print(f'  ds_ovs/     train {out["ds_ovs"][0]:5} img / {out["ds_ovs"][1]:5} mit  (CON oversampling)')
    print(f'  ds_sinover/ train {out["ds_sinover"][0]:5} img / {out["ds_sinover"][1]:5} mit  (SIN oversampling)')
    print(f'  eval/test_ov20 {e20[0]} teselas | eval/test_ov50 {e50[0]} teselas')
    print('  -> YOLO labels: ejecuta  python3 parches.py yolo-labels')

def cmd_yolo_labels(args):
    n_files = n_box = 0
    for vdir in sorted(VERS.glob('*/*')):
        cf = vdir / '_annotations.coco.json'
        if not cf.exists():
            continue
        coco = json.loads(cf.read_text())
        wh = {im['id']: (im['width'], im['height']) for im in coco['images']}
        anns = defaultdict(list)
        for a in coco['annotations']:
            anns[a['image_id']].append(a['bbox'])
        ld = vdir / 'labels'
        if ld.exists():
            shutil.rmtree(ld)
        ld.mkdir()
        for im in coco['images']:
            W, H = wh[im['id']]
            lines = [f'0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}'
                     for (x, y, w, h) in anns.get(im['id'], [])]
            (ld / f'{Path(im["file_name"]).stem}.txt').write_text('\n'.join(lines))
            n_files += 1; n_box += len(lines)
    print(f'YOLO labels: {n_files} ficheros .txt, {n_box} cajas de mitosis')

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('generar').set_defaults(func=cmd_generar)
    sub.add_parser('ensamblar').set_defaults(func=cmd_ensamblar)
    sub.add_parser('yolo-labels').set_defaults(func=cmd_yolo_labels)
    args = ap.parse_args(); args.func(args)

if __name__ == '__main__':
    main()
