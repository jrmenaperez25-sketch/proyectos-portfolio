#!/usr/bin/env python3

from __future__ import annotations

import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

SEED = 42
PATCH_SIZE = 512
HALF = PATCH_SIZE // 2

ANGLES = [0, 36, 72, 108, 144]
FLIP_OPTIONS = ['none', 'h', 'v']
ALL_COMBOS = [(a, f) for a in ANGLES for f in FLIP_OPTIONS]

BBOX_RADIUS_PX_AT_512 = {'aperio': 25, 'hamamatsu': 27}
MPP_BY_SCANNER = {'aperio': 0.2455, 'hamamatsu': 0.2274}

N_EMPTY_PER_FRAME = 2
MAX_TRIES = 5000
IMAGE_QUALITY = 95

BASE_RAW = Path(f'{WORKSPACE}/Dataset_Mitosis14_TUPAC16')

TODOS = [f'{n:02d}' for n in range(3, 19)]
TEST = ['04']
OUTLIERS = ['08', '12', '13', '15', '18']
SCANNERS = {'aperio': 'A', 'hamamatsu': 'H'}

def versiones_de(num: str) -> List[str]:
    if num in TEST:
        return []
    if num in OUTLIERS:
        return ['oversampling']
    return ['coverage', 'oversampling']

def bbox_radius_for(scanner: str) -> int:
    base_r = BBOX_RADIUS_PX_AT_512.get(scanner, 25)
    return int(round(base_r * PATCH_SIZE / 512))

def angle_margin(deg: float) -> int:
    r = math.radians(deg)
    return int(math.ceil(PATCH_SIZE / 2 * (abs(math.cos(r)) + abs(math.sin(r))))) + 1

def extract_patch_img(img: Image.Image, cx_c: float, cy_c: float,
                      angle_deg: float, flip: str) -> Image.Image:
    r = math.radians(angle_deg)
    cos_a, sin_a = math.cos(r), math.sin(r)
    half = PATCH_SIZE / 2
    a = cos_a; b = -sin_a; c = -cos_a * half + sin_a * half + cx_c
    d = sin_a; e = cos_a;  f = -sin_a * half - cos_a * half + cy_c
    patch = img.transform((PATCH_SIZE, PATCH_SIZE), Image.AFFINE, (a, b, c, d, e, f),
                          resample=Image.BICUBIC)
    if flip == 'h':
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == 'v':
        patch = patch.transpose(Image.FLIP_TOP_BOTTOM)
    return patch

def world_to_patch(wx: float, wy: float, cx_c: float, cy_c: float,
                   angle_deg: float, flip: str) -> Tuple[float, float]:
    r = math.radians(angle_deg)
    cos_a, sin_a = math.cos(r), math.sin(r)
    dx, dy = wx - cx_c, wy - cy_c
    pu = cos_a * dx + sin_a * dy + HALF
    pv = -sin_a * dx + cos_a * dy + HALF
    if flip == 'h':
        pu = PATCH_SIZE - pu
    elif flip == 'v':
        pv = PATCH_SIZE - pv
    return pu, pv

def patch_corners_in_frame(cx_c: float, cy_c: float, angle_deg: float) -> List[Tuple[float, float]]:
    r = math.radians(angle_deg)
    cos_a, sin_a = math.cos(r), math.sin(r)
    half = PATCH_SIZE / 2
    a, b, c = cos_a, -sin_a, -cos_a * half + sin_a * half + cx_c
    d, e, f = sin_a, cos_a, -sin_a * half - cos_a * half + cy_c
    P = PATCH_SIZE
    return [(a * u + b * v + c, d * u + e * v + f) for (u, v) in [(0, 0), (P, 0), (P, P), (0, P)]]

def find_center_for_point(W: int, H: int, px: float, py: float,
                          angle_deg: float, flip: str, bbox_r: int,
                          rng: random.Random,
                          max_tries: int = MAX_TRIES) -> Optional[Tuple[float, float]]:
    r = math.radians(angle_deg)
    cos_a, sin_a = math.cos(r), math.sin(r)
    half = PATCH_SIZE / 2
    margin = angle_margin(angle_deg)
    lo, hi = float(bbox_r), float(PATCH_SIZE - bbox_r)
    for _ in range(max_tries):
        mx = rng.uniform(lo, hi)
        my = rng.uniform(lo, hi)
        mx_ = PATCH_SIZE - mx if flip == 'h' else mx
        my_ = PATCH_SIZE - my if flip == 'v' else my
        dx, dy = mx_ - half, my_ - half
        cx_c = px - cos_a * dx + sin_a * dy
        cy_c = py - sin_a * dx - cos_a * dy
        if margin <= cx_c <= W - margin and margin <= cy_c <= H - margin:
            return cx_c, cy_c
    return None

def find_random_empty_center(W: int, H: int, angle_deg: float,
                             rng: random.Random) -> Tuple[float, float]:
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
                out.append({
                    'world_x': round(wx, 2), 'world_y': round(wy, 2), 'conf': conf,
                    'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    'category_id': cat_id,
                    'punto_id': tup[3] if is_mit else f'not_{wx:.2f}_{wy:.2f}',
                    'mitosis_id': tup[3] if is_mit else '',
                })
        return out
    return project(mit_pts, 1, True), project(not_pts, 2, False)

def read_csv_coords(path: Path) -> List[Tuple]:
    coords = []
    try:
        with open(path, newline='') as f:
            for row in csv.reader(f):
                row = [c.strip() for c in row if c.strip()]
                if len(row) >= 2:
                    try:
                        x, y = float(row[0]), float(row[1])
                        conf = float(row[2]) if len(row) >= 3 else None
                        coords.append((x, y, conf))
                    except ValueError:
                        pass
    except Exception:
        pass
    return coords

def make_mitosis_id(scanner, patient, frame_stem, wx, wy) -> str:
    return f'mit_{scanner}_{patient}_{frame_stem}_{wx:.2f}_{wy:.2f}'

def dedup_xy(pts: List[Tuple]) -> List[Tuple]:
    seen, out = set(), []
    for p in pts:
        k = (round(p[0], 2), round(p[1], 2))
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out

def load_patient_frames(scanner: str, patient: str) -> List[Dict[str, Any]]:
    if scanner not in ('aperio', 'hamamatsu'):
        raise ValueError(f'Scanner no soportado: {scanner}')
    scanner_name = 'Aperio' if scanner == 'aperio' else 'Hamamatsu'
    pat_dir = None
    for split_folder in (f'Training {scanner_name}', f'Testing {scanner_name}'):
        d = BASE_RAW / split_folder / patient / patient
        if d.exists():
            pat_dir = d
            break
    if pat_dir is None:
        return []
    frames_x40 = pat_dir / 'frames' / 'x40'
    mit_dir = pat_dir / 'mitosis'
    if not frames_x40.exists():
        return []
    frames = []
    for img_path in sorted(frames_x40.glob('*.tiff')):
        stem = img_path.stem
        mit_raw = dedup_xy(read_csv_coords(mit_dir / f'{stem}_mitosis.csv'))
        not_raw = dedup_xy(read_csv_coords(mit_dir / f'{stem}_not_mitosis.csv'))
        mit_pts = [(wx, wy, conf, make_mitosis_id(scanner, patient, stem, wx, wy))
                   for (wx, wy, conf) in mit_raw]
        not_pts = list(not_raw)
        try:
            with Image.open(img_path) as pil:
                W, H = pil.size
        except Exception:
            continue
        frames.append({
            'img_path': img_path, 'W': W, 'H': H,
            'dataset': 'mitos14', 'scanner': scanner, 'patient': patient,
            'frame_stem': stem, 'mit_pts': mit_pts, 'not_pts': not_pts,
            'is_empty': len(mit_pts) == 0 and len(not_pts) == 0,
        })
    return frames

def _spec(fr, sc, final_class, mitosis_id, wx, wy, conf, cx_c, cy_c,
          angle, flip, rep_idx, source_index) -> Dict:
    return {
        'scanner': sc, 'patient': fr['patient'], 'frame_stem': fr['frame_stem'],
        'frame': fr, 'final_class': final_class, 'mitosis_id': mitosis_id,
        'source_world_x': wx, 'source_world_y': wy, 'source_conf': conf,
        'cx_c': cx_c, 'cy_c': cy_c, 'angle': angle, 'flip': flip,
        'representative_index': rep_idx, 'source_index': source_index,
    }

def patch_filename(version: str, sp: Dict) -> str:
    base = f'{sp["frame_stem"]}_{sp["final_class"]}_{sp["source_index"]}'
    if version == 'oversampling':
        base += f'_a{sp["angle"]}_f{sp["flip"]}'
    return base + '.jpg'

def _clusterizar(items, maxspan):
    clusters = [{'idx': [i], 'minx': it['x'], 'maxx': it['x'],
                 'miny': it['y'], 'maxy': it['y']} for i, it in enumerate(items)]

    def cabe(a, b):
        return (max(a['maxx'], b['maxx']) - min(a['minx'], b['minx']) <= maxspan and
                max(a['maxy'], b['maxy']) - min(a['miny'], b['miny']) <= maxspan)

    def cen(c):
        return ((c['minx'] + c['maxx']) / 2, (c['miny'] + c['maxy']) / 2)

    while True:
        mejor = None
        for i in range(len(clusters)):
            xi, yi = cen(clusters[i])
            for j in range(i + 1, len(clusters)):
                if not cabe(clusters[i], clusters[j]):
                    continue
                xj, yj = cen(clusters[j])
                d = (xi - xj) ** 2 + (yi - yj) ** 2
                if mejor is None or d < mejor[0]:
                    mejor = (d, i, j)
        if mejor is None:
            break
        _, i, j = mejor
        a, b = clusters[i], clusters[j]
        clusters[i] = {'idx': a['idx'] + b['idx'],
                       'minx': min(a['minx'], b['minx']), 'maxx': max(a['maxx'], b['maxx']),
                       'miny': min(a['miny'], b['miny']), 'maxy': max(a['maxy'], b['maxy'])}
        clusters.pop(j)
    return clusters

def gen_coverage(frames, sc, rng) -> List[Dict]:
    specs = []
    bbox_r = bbox_radius_for(sc)
    maxspan = PATCH_SIZE - 2 * bbox_r
    half = PATCH_SIZE / 2
    for fr in frames:
        items = [{'x': wx, 'y': wy, 'conf': conf, 'clase': 'mitosis', 'src': i + 1, 'pid': mid}
                 for i, (wx, wy, conf, mid) in enumerate(fr['mit_pts'])]
        items += [{'x': wx, 'y': wy, 'conf': conf, 'clase': 'not_mitosis', 'src': i + 1,
                   'pid': f'not_{wx:.2f}_{wy:.2f}'}
                  for i, (wx, wy, conf) in enumerate(fr['not_pts'])]
        if not items:
            continue
        for cl in _clusterizar(items, maxspan):
            cx_c = min(max((cl['minx'] + cl['maxx']) / 2, half), fr['W'] - half)
            cy_c = min(max((cl['miny'] + cl['maxy']) / 2, half), fr['H'] - half)
            rep = min((items[k] for k in cl['idx']),
                      key=lambda it: (it['clase'] != 'mitosis', it['src']))
            specs.append(_spec(fr, sc, rep['clase'],
                               rep['pid'] if rep['clase'] == 'mitosis' else '',
                               rep['x'], rep['y'], rep['conf'], cx_c, cy_c, 0, 'none',
                               0, rep['src']))
    return specs

def gen_representante_unico(frames, sc, rng) -> List[Dict]:
    specs = []
    bbox_r = bbox_radius_for(sc)
    for fr in frames:
        puntos = [('mitosis', wx, wy, conf, mid, i + 1)
                  for i, (wx, wy, conf, mid) in enumerate(fr['mit_pts'])]
        puntos += [('not_mitosis', wx, wy, conf, '', i + 1)
                   for i, (wx, wy, conf) in enumerate(fr['not_pts'])]
        for (clase, wx, wy, conf, mid, source_index) in puntos:
            chosen = None
            for (angle, flip) in ALL_COMBOS:
                center = find_center_for_point(fr['W'], fr['H'], wx, wy, angle, flip, bbox_r, rng)
                if center is not None:
                    chosen = (angle, flip, center); break
            if chosen is None:
                continue
            angle, flip, (cx_c, cy_c) = chosen
            specs.append(_spec(fr, sc, clase, mid, wx, wy, conf,
                               cx_c, cy_c, angle, flip, 0, source_index))
    return specs

def gen_oversampling(frames, sc, rng, max_combos) -> List[Dict]:
    specs = []
    bbox_r = bbox_radius_for(sc)

    for fr in frames:
        for source_index, (wx, wy, conf, mid) in enumerate(fr['mit_pts'], start=1):
            combos = list(ALL_COMBOS); rng.shuffle(combos)
            rep_idx = 0
            for (angle, flip) in combos:
                if rep_idx >= max_combos:
                    break
                center = find_center_for_point(fr['W'], fr['H'], wx, wy, angle, flip, bbox_r, rng)
                if center is None:
                    continue
                cx_c, cy_c = center
                specs.append(_spec(fr, sc, 'mitosis', mid, wx, wy, conf,
                                   cx_c, cy_c, angle, flip, rep_idx, source_index))
                rep_idx += 1

    for fr in frames:
        for source_index, (wx, wy, conf) in enumerate(fr['not_pts'], start=1):
            combos = list(ALL_COMBOS); rng.shuffle(combos)
            for (angle, flip) in combos:
                center = find_center_for_point(fr['W'], fr['H'], wx, wy, angle, flip, bbox_r, rng)
                if center is None:
                    continue
                cx_c, cy_c = center
                mit_in, _ = annotations_in_patch(fr['mit_pts'], [], cx_c, cy_c, angle, flip, bbox_r)
                if mit_in:
                    continue
                specs.append(_spec(fr, sc, 'not_mitosis', '', wx, wy, conf,
                                   cx_c, cy_c, angle, flip, 0, source_index))
                break

    for fr in frames:
        if not fr['is_empty']:
            continue
        for empty_idx in range(N_EMPTY_PER_FRAME):
            angle = rng.choice(ANGLES); flip = rng.choice(FLIP_OPTIONS)
            cx_c, cy_c = find_random_empty_center(fr['W'], fr['H'], angle, rng)
            specs.append(_spec(fr, sc, 'empty', '', None, None, None,
                               cx_c, cy_c, angle, flip, empty_idx, empty_idx + 1))

    rng.shuffle(specs)
    return specs

SW_STRIDE = None

def gen_sliding_window(frames, sc, rng=None) -> List[Dict]:
    specs = []
    bbox_r = bbox_radius_for(sc)
    stride = SW_STRIDE if SW_STRIDE else PATCH_SIZE - 2 * bbox_r
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
                specs.append(_spec(fr, sc, 'tile', '', ox, oy, None,
                                   ox + PATCH_SIZE / 2, oy + PATCH_SIZE / 2,
                                   0, 'none', 0, idx))
    return specs

GENERADORES = {'coverage': gen_coverage,
               'representante_unico': gen_representante_unico,
               'oversampling': gen_oversampling,
               'sliding_window': gen_sliding_window}

def generar_specs(version, frames, sc, seed, max_combos) -> List[Dict]:
    rng = random.Random(seed)
    if version == 'oversampling':
        return gen_oversampling(frames, sc, rng, max_combos)
    return GENERADORES[version](frames, sc, rng)

def extract_and_write(specs, out_dir, scanner, patient, version, progreso=True) -> Dict:
    images_dir = out_dir / 'images'
    if images_dir.exists():
        for old in images_dir.glob('*.jpg'):
            old.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    images, anns_model, anns_full, manifest_rows = [], [], [], []
    img_id = ann_id_model = ann_id_full = 1
    usados = set()

    specs_by_img = defaultdict(list)
    for sp in specs:
        specs_by_img[sp['frame']['img_path']].append(sp)

    it = specs_by_img.items()
    if progreso:
        it = tqdm(it, desc=f'  {scanner}/{patient}/{version}', leave=False)
    for img_path, sp_list in it:
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        for sp in sp_list:
            fr = sp['frame']
            bbox_r = bbox_radius_for(fr['scanner'])
            patch = extract_patch_img(img, sp['cx_c'], sp['cy_c'], sp['angle'], sp['flip'])
            mit_in, not_in = annotations_in_patch(
                fr['mit_pts'], fr['not_pts'], sp['cx_c'], sp['cy_c'], sp['angle'], sp['flip'], bbox_r)

            fname = patch_filename(version, sp)
            if fname in usados:
                s, e = fname.rsplit('.', 1); k = 2
                while f'{s}_dup{k}.{e}' in usados:
                    k += 1
                fname = f'{s}_dup{k}.{e}'
            usados.add(fname)
            patch.save(images_dir / fname, quality=IMAGE_QUALITY)

            images.append({
                'id': img_id, 'file_name': fname, 'width': PATCH_SIZE, 'height': PATCH_SIZE,
                'dataset': fr['dataset'], 'scanner': fr['scanner'], 'patient': fr['patient'],
                'frame_stem': fr['frame_stem'], 'final_class': sp['final_class'],
                'mitosis_id': sp['mitosis_id'], 'source_index': sp['source_index'],
                'source_world_x': sp['source_world_x'], 'source_world_y': sp['source_world_y'],
                'angle': sp['angle'], 'flip': sp['flip'],
                'representative_index': sp['representative_index'],
                'patch_center_x': round(sp['cx_c'], 4), 'patch_center_y': round(sp['cy_c'], 4),
                'n_mit_in_patch': len(mit_in), 'n_not_in_patch': len(not_in),
                'original_image_width': fr['W'], 'original_image_height': fr['H'],
                'mpp_um_per_px': MPP_BY_SCANNER.get(fr['scanner']),
            })
            for a in mit_in:
                e = {'id': ann_id_full, 'image_id': img_id, 'category_id': 1,
                     'bbox': a['bbox'], 'area': round(a['bbox'][2] * a['bbox'][3], 2),
                     'iscrowd': 0, 'world_x': a['world_x'], 'world_y': a['world_y'],
                     'conf': a['conf'], 'mitosis_id': a['mitosis_id']}
                anns_full.append(e); ann_id_full += 1
                e2 = dict(e); e2['id'] = ann_id_model
                anns_model.append(e2); ann_id_model += 1
            for a in not_in:
                anns_full.append({'id': ann_id_full, 'image_id': img_id, 'category_id': 2,
                                  'bbox': a['bbox'], 'area': round(a['bbox'][2] * a['bbox'][3], 2),
                                  'iscrowd': 0, 'world_x': a['world_x'], 'world_y': a['world_y'],
                                  'conf': a['conf']})
                ann_id_full += 1
            manifest_rows.append({
                'image_id': img_id, 'patch_filename': fname, 'scanner': fr['scanner'],
                'patient_id': fr['patient'], 'frame_stem': fr['frame_stem'],
                'final_class': sp['final_class'], 'mitosis_id': sp['mitosis_id'],
                'source_index': sp['source_index'], 'representative_index': sp['representative_index'],
                'source_world_x': sp['source_world_x'], 'source_world_y': sp['source_world_y'],
                'patch_center_x': round(sp['cx_c'], 4), 'patch_center_y': round(sp['cy_c'], 4),
                'angle': sp['angle'], 'flip': sp['flip'],
                'original_image_width': fr['W'], 'original_image_height': fr['H'],
                'mpp_um_per_px': MPP_BY_SCANNER.get(fr['scanner']), 'bbox_radius_px': bbox_r,
                'n_mit_in_patch': len(mit_in), 'n_not_in_patch': len(not_in),
                'original_image_path': str(img_path),
            })
            img_id += 1
        img.close()

    coco_model = {'info': {'description': f'{scanner}/{patient}/{version}', 'version': 'tfm'},
                  'categories': [{'id': 1, 'name': 'mitosis', 'supercategory': 'cell'}],
                  'images': images, 'annotations': anns_model}
    coco_full = {'info': {'description': f'{scanner}/{patient}/{version} (full)', 'version': 'tfm'},
                 'categories': [{'id': 1, 'name': 'mitosis', 'supercategory': 'cell'},
                                {'id': 2, 'name': 'not_mitosis', 'supercategory': 'cell'}],
                 'images': images, 'annotations': anns_full}
    (out_dir / '_annotations.coco.json').write_text(json.dumps(coco_model))
    (out_dir / '_annotations_full.coco.json').write_text(json.dumps(coco_full))
    if manifest_rows:
        cols = list(manifest_rows[0].keys())
        with open(out_dir / '_manifest.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(manifest_rows)

    n_mit = len(anns_model); n_not = len(anns_full) - n_mit
    return {'n_images': len(images), 'n_mit_ann': n_mit, 'n_not_ann': n_not}

def construir_version(scanner, patient, version, salida, seed=SEED,
                      max_combos=len(ALL_COMBOS), progreso=True) -> Dict:
    out_dir = salida / 'mitos14' / scanner / patient / version
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_patient_frames(scanner, patient)
    if not frames:
        return {'n_images': 0, 'n_mit_ann': 0, 'n_not_ann': 0, 'error': 'sin frames'}
    specs = generar_specs(version, frames, scanner, seed, max_combos)
    stats = extract_and_write(specs, out_dir, scanner, patient, version, progreso)
    stats['n_frames'] = len(frames)
    stats['n_mit_puntos'] = sum(len(fr['mit_pts']) for fr in frames)
    stats['n_not_puntos'] = sum(len(fr['not_pts']) for fr in frames)
    (out_dir / 'version_info.json').write_text(json.dumps({
        'scanner': scanner, 'paciente': patient, 'version': version,
        'seed': seed, 'patch_size': PATCH_SIZE,
        'max_combos': max_combos if version == 'oversampling' else None,
        'resultado': stats,
    }, indent=2, default=str))
    return stats

VAL_ELEGIBLES = ['03', '05', '06', '07', '09', '10', '11', '14', '16', '17']

def _leer_version(versiones_root, sc, pac, version):
    d = versiones_root / 'mitos14' / sc / pac / version
    coco = json.loads((d / '_annotations.coco.json').read_text())
    anns = defaultdict(list)
    for a in coco['annotations']:
        anns[a['image_id']].append(a)
    return d, coco['images'], anns

def _K_global(train_pac, versiones_root):
    mit_ids, n_neg = set(), 0
    for (sc, pac) in train_pac:
        _, ims, _ = _leer_version(versiones_root, sc, pac, 'oversampling')
        for im in ims:
            if im['final_class'] == 'mitosis':
                mit_ids.add(im['mitosis_id'])
            else:
                n_neg += 1
    n_mit = len(mit_ids)
    K = max(1, math.ceil(n_neg / n_mit)) if n_mit else 1
    return K, n_mit, n_neg

def _añadir_split(out_split, pacientes, version, versiones_root, filtro=None):
    if out_split.exists():
        shutil.rmtree(out_split)
    out_split.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    img_id = ann_id = 1
    for (sc, pac) in pacientes:
        d, ims, anns = _leer_version(versiones_root, sc, pac, version)
        src_dir = (d / 'images').resolve()
        for im in ims:
            if filtro and not filtro(im):
                continue
            new = dict(im); new['id'] = img_id
            new['file_name'] = str(src_dir / im['file_name'])
            new['file_basename'] = im['file_name']
            images.append(new)
            for a in anns.get(im['id'], []):
                na = dict(a); na['id'] = ann_id; na['image_id'] = img_id
                annotations.append(na); ann_id += 1
            img_id += 1
    coco = {'info': {'description': out_split.name, 'version': 'tfm'},
            'categories': [{'id': 1, 'name': 'mitosis', 'supercategory': 'cell'}],
            'images': images, 'annotations': annotations}
    (out_split / '_annotations.coco.json').write_text(json.dumps(coco))
    return len(images), len(annotations)

def ensamblar_fold(versiones_root, salida, val_num, train_version='oversampling') -> Dict:
    val_pac = [('aperio', f'A{val_num}'), ('hamamatsu', f'H{val_num}')]
    train_nums = [n for n in TODOS if n not in TEST and n != val_num]
    train_pac = [(sc, pre + n) for sc, pre in SCANNERS.items() for n in train_nums]

    out = salida / f'fold_{val_num}'
    if out.exists():
        shutil.rmtree(out)
    if train_version == 'oversampling':
        K, n_mit, n_neg = _K_global(train_pac, versiones_root)
        filtro = lambda im: (im['final_class'] != 'mitosis') or (im['representative_index'] < K)
    else:
        K, n_mit, n_neg, filtro = None, None, None, None
    nt_img, nt_ann = _añadir_split(out / 'train', train_pac, train_version, versiones_root, filtro)
    nv_img, nv_ann = _añadir_split(out / 'valid', val_pac, 'coverage', versiones_root)
    _añadir_split(out / 'test', val_pac, 'coverage', versiones_root)

    (out / 'data.yaml').write_text(
        f'path: {out.resolve()}\ntrain: train\nval: valid\n\nnc: 1\nnames:\n  - mitosis\n')
    def _lista(split):
        c = json.loads((out / split / '_annotations.coco.json').read_text())
        return '\n'.join(im['file_name'] for im in c['images'])
    (out / 'train_yolo.txt').write_text(_lista('train'))
    (out / 'val_yolo.txt').write_text(_lista('valid'))
    (out / 'dataset_yolo.yaml').write_text(
        f'path: {out.resolve()}\ntrain: train_yolo.txt\nval: val_yolo.txt\n\nnc: 1\nnames:\n  - mitosis\n')
    (out / 'split_config.json').write_text(json.dumps({
        'fold': val_num, 'test_fijo': 'A04+H04 (aparte)',
        'val_patients': [p for _, p in val_pac],
        'train_patients': [p for _, p in train_pac],
        'K': K, 'N_mit': n_mit, 'N_neg': n_neg,
        'train': {'n_images': nt_img, 'n_ann_mit': nt_ann},
        'valid': {'n_images': nv_img, 'n_ann_mit': nv_ann},
    }, indent=2))
    return {'fold': val_num, 'K': K, 'train_img': nt_img, 'train_ann': nt_ann,
            'val_img': nv_img, 'val_ann': nv_ann}

def cmd_ensamblar(args):
    set_patch_size(args.patch_size)
    nums = [args.fold] if args.fold else VAL_ELEGIBLES
    print(f'ENSAMBLAR {len(nums)} fold(s) [train={args.train_version}]  ->  {args.salida}\n')
    for n in nums:
        r = ensamblar_fold(args.versiones_root, args.salida, n, args.train_version)
        k = f'K={r["K"]:2d}' if r['K'] is not None else 'K=NA'
        print(f'  fold_{n}: {k} | train {r["train_img"]:>6,} img / {r["train_ann"]:>6,} mit'
              f' | val {r["val_img"]:>5,} img / {r["val_ann"]:>4,} mit')

def cmd_yolo_labels(args):
    salida = args.salida
    n_files = n_box = 0
    for vdir in sorted(salida.glob('mitos14/*/*/*')):
        if not (vdir / '_annotations.coco.json').exists():
            continue
        coco = json.loads((vdir / '_annotations.coco.json').read_text())
        wh = {im['id']: (im['width'], im['height']) for im in coco['images']}
        anns = defaultdict(list)
        for a in coco['annotations']:
            anns[a['image_id']].append(a['bbox'])
        labels_dir = vdir / 'labels'
        if labels_dir.exists():
            shutil.rmtree(labels_dir)
        labels_dir.mkdir()
        for im in coco['images']:
            W, H = wh[im['id']]
            lines = []
            for (x, y, w, h) in anns.get(im['id'], []):
                lines.append(f'0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}')
            (labels_dir / f'{Path(im["file_name"]).stem}.txt').write_text('\n'.join(lines))
            n_files += 1; n_box += len(lines)
    print(f'YOLO labels: {n_files:,} ficheros .txt, {n_box:,} cajas de mitosis')

COLORES = [(255, 0, 0), (0, 130, 255), (0, 180, 0), (255, 140, 0), (190, 0, 190),
           (0, 170, 170), (120, 70, 255), (150, 100, 30), (255, 0, 110), (90, 90, 90)]

def _fuente(sz=24):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', sz)
    except Exception:
        return ImageFont.load_default()

def dibujar_frame(scanner, patient, version, frame_stem, salida_png, seed=SEED,
                  solo_rep0=False) -> Dict:
    frames = load_patient_frames(scanner, patient)
    fr = next((f for f in frames if f['frame_stem'] == frame_stem), None)
    if fr is None:
        return {'error': f'frame {frame_stem} no encontrado'}
    specs = [s for s in generar_specs(version, frames, scanner, seed, len(ALL_COMBOS))
             if s['frame_stem'] == frame_stem]
    if solo_rep0:
        specs = [s for s in specs if s['representative_index'] == 0]
    bbox_r = bbox_radius_for(scanner)

    geo = defaultdict(int)
    for sp in specs:
        mit_in, not_in = annotations_in_patch(fr['mit_pts'], fr['not_pts'],
                                               sp['cx_c'], sp['cy_c'], sp['angle'], sp['flip'], bbox_r)
        for a in mit_in + not_in:
            geo[a['punto_id']] += 1
    max_rep = max(geo.values()) if geo else 0

    img = Image.open(fr['img_path']).convert('RGB')
    draw = ImageDraw.Draw(img); font = _fuente(24)
    for i, sp in enumerate(specs):
        color = COLORES[i % len(COLORES)]
        corners = patch_corners_in_frame(sp['cx_c'], sp['cy_c'], sp['angle'])
        draw.line(corners + [corners[0]], fill=color, width=4)
    for j, (wx, wy, *_r) in enumerate(fr['mit_pts'], 1):
        draw.ellipse([wx - 9, wy - 9, wx + 9, wy + 9], fill=(0, 80, 255), outline=(255, 255, 255), width=2)
        draw.text((wx + 11, wy - 11), f'M{j}', fill=(0, 80, 255), font=font)
    for j, (wx, wy, *_r) in enumerate(fr['not_pts'], 1):
        draw.ellipse([wx - 8, wy - 8, wx + 8, wy + 8], fill=(0, 0, 0), outline=(255, 255, 255), width=2)
        draw.text((wx + 10, wy - 10), f'N{j}', fill=(0, 0, 0), font=font)
    Path(salida_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(salida_png)
    return {'frame': frame_stem, 'version': version, 'n_parches': len(specs),
            'n_mit': len(fr['mit_pts']), 'n_not': len(fr['not_pts']),
            'max_veces_en_parche': max_rep, 'png': str(salida_png)}

def cmd_construir(args):
    set_patch_size(args.patch_size)
    if getattr(args, 'sw_stride', None):
        global SW_STRIDE
        SW_STRIDE = args.sw_stride
        print(f'  sliding window stride = {SW_STRIDE} (solape {100*(1-SW_STRIDE/PATCH_SIZE):.0f}%)')
    print(f'CONSTRUIR {args.scanner}/{args.paciente} -> {args.version}')
    st = construir_version(args.scanner, args.paciente, args.version, args.salida,
                           args.seed, args.max_combos)
    print(f'  {st["n_images"]:,} parches | {st["n_mit_ann"]:,} ann mit | {st["n_not_ann"]:,} ann not')

def cmd_generar_todo(args):
    set_patch_size(args.patch_size)
    tareas = [(sc, pre + num, num, v)
              for sc, pre in SCANNERS.items()
              for num in TODOS
              for v in versiones_de(num)]
    print(f'GENERAR TODO -> {args.salida}   ({len(tareas)} construcciones)\n')
    resumen = defaultdict(lambda: [0, 0])
    for sc, pac, num, v in tareas:
        st = construir_version(sc, pac, v, args.salida, args.seed, args.max_combos)
        resumen[v][0] += 1; resumen[v][1] += st['n_images']
        print(f'  {sc:9s} {pac} {v:20s} -> {st["n_images"]:>6,} parches')
    print('\nRESUMEN:')
    for v, (npac, njpg) in resumen.items():
        print(f'  {v:20s}: {npac:2d} carpetas · {njpg:,} parches')

def cmd_dibujar(args):
    set_patch_size(args.patch_size)
    res = dibujar_frame(args.scanner, args.paciente, args.version, args.frame,
                        args.salida, args.seed, args.solo_rep0)
    print(res)

def cmd_muestras(args):
    set_patch_size(args.patch_size)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    print(f'MUESTRAS -> {out}\n')
    filas = []
    for sc, pre in SCANNERS.items():
        for num in TODOS:
            roles = versiones_de(num)
            if not roles:
                continue
            pac = pre + num
            version = roles[0]
            frames = [f for f in load_patient_frames(sc, pac)
                      if f['mit_pts'] or f['not_pts']]
            if not frames:
                continue
            fr = rng.choice(frames)
            png = out / f'{sc}_{pac}_{version}_{fr["frame_stem"]}.png'
            res = dibujar_frame(sc, pac, version, fr['frame_stem'], png, args.seed,
                                solo_rep0=(version == 'oversampling'))
            filas.append((sc, pac, version, res.get('frame'), res.get('n_parches'),
                          res.get('max_veces_en_parche')))
            print(f'  {sc:9s} {pac} {version:20s} {res.get("frame")}: '
                  f'{res.get("n_parches")} parches, máx {res.get("max_veces_en_parche")}/punto')
    print(f'\n{len(filas)} muestras guardadas en {out}')

def cmd_verificar(args):
    set_patch_size(args.patch_size)
    salida = args.salida
    print('=' * 70); print('BARRIDO DE ESTABILIDAD'); print('=' * 70)
    ok = True

    print('\n[1] Reproducibilidad (misma semilla -> mismos specs)')
    for sc, pac, v in [('aperio', 'A03', 'coverage'), ('aperio', 'A05', 'oversampling'),
                       ('hamamatsu', 'H06', 'representante_unico')]:
        frames = load_patient_frames(sc, pac)
        def firma():
            return [(s['frame_stem'], s['final_class'], s['source_index'],
                     round(s['cx_c'], 3), round(s['cy_c'], 3), s['angle'], s['flip'])
                    for s in generar_specs(v, frames, sc, SEED, len(ALL_COMBOS))]
        igual = firma() == firma()
        ok = ok and igual
        print(f'   {sc}/{pac}/{v}: {"OK" if igual else "FALLO"}')

    print('\n[2] Roles por paciente (versiones correctas en disco)')
    roles_ok = True
    for sc, pre in SCANNERS.items():
        for num in TODOS:
            pac = pre + num
            base = salida / 'mitos14' / sc / pac
            hay = sorted([d.name for d in base.iterdir()]) if base.exists() else []
            esp = sorted(versiones_de(num))
            if hay != esp:
                roles_ok = False; print(f'   [FALLO] {sc}/{pac}: {hay} != {esp}')
    ok = ok and roles_ok
    print('   OK' if roles_ok else '   FALLO')

    print('\n[3] Integridad COCO / disco / manifest')
    integ_ok = True
    for vdir in sorted(salida.glob('mitos14/*/*/*')):
        coco = json.loads((vdir / '_annotations.coco.json').read_text())
        njpg = len(list((vdir / 'images').glob('*.jpg')))
        nman = sum(1 for _ in csv.DictReader(open(vdir / '_manifest.csv')))
        if not (len(coco['images']) == njpg == nman):
            integ_ok = False
            print(f'   [FALLO] {vdir}: coco={len(coco["images"])} jpg={njpg} manifest={nman}')
    ok = ok and integ_ok
    print('   OK' if integ_ok else '   FALLO')

    print('\n[4] Coverage: ninguna mitosis/not sin anotar + duplicados (esperado ~2%)')
    T = {'mit': 0, 'not': 0}; SIN = {'mit': 0, 'not': 0}; DUP = {'mit': 0, 'not': 0}
    for covdir in sorted(salida.glob('mitos14/*/*/coverage')):
        sc = covdir.parts[-3]; pac = covdir.parts[-2]
        full = json.loads((covdir / '_annotations_full.coco.json').read_text())
        frame_of = {im['id']: im['frame_stem'] for im in full['images']}
        cnt = defaultdict(int)
        for a in full['annotations']:
            cat = 'mit' if a['category_id'] == 1 else 'not'
            cnt[(frame_of[a['image_id']], cat, round(a['world_x'], 1), round(a['world_y'], 1))] += 1
        for fr in load_patient_frames(sc, pac):
            for (wx, wy, conf, mid) in fr['mit_pts']:
                T['mit'] += 1; k = (fr['frame_stem'], 'mit', round(wx, 1), round(wy, 1))
                if cnt[k] == 0: SIN['mit'] += 1
                elif cnt[k] > 1: DUP['mit'] += 1
            for (wx, wy, conf) in fr['not_pts']:
                T['not'] += 1; k = (fr['frame_stem'], 'not', round(wx, 1), round(wy, 1))
                if cnt[k] == 0: SIN['not'] += 1
                elif cnt[k] > 1: DUP['not'] += 1
    print(f'   mitosis:     total {T["mit"]:>5} | sin anotar {SIN["mit"]:>3} | '
          f'duplicadas {DUP["mit"]:>3} ({100*DUP["mit"]/max(T["mit"],1):.2f}%)')
    print(f'   not_mitosis: total {T["not"]:>5} | sin anotar {SIN["not"]:>3} | '
          f'duplicadas {DUP["not"]:>3} ({100*DUP["not"]/max(T["not"],1):.2f}%)')
    sin_ok = (SIN['mit'] == 0 and SIN['not'] == 0)
    ok = ok and sin_ok
    print('   OK (nada sin anotar)' if sin_ok else '   FALLO (hay puntos sin anotar)')

    print('\n' + '=' * 70)
    print('RESULTADO GLOBAL:', 'TODO OK ✅' if ok else 'HAY FALLOS ❌')
    print('=' * 70)

def set_patch_size(ps: int):
    global PATCH_SIZE, HALF
    PATCH_SIZE = ps; HALF = ps // 2

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    c = sub.add_parser('construir', help='Construye 1 versión de 1 paciente+scanner.')
    c.add_argument('--scanner', required=True, choices=list(SCANNERS))
    c.add_argument('--paciente', required=True)
    c.add_argument('--version', required=True, choices=list(GENERADORES))
    c.add_argument('--salida', type=Path, required=True)
    c.add_argument('--seed', type=int, default=SEED)
    c.add_argument('--patch-size', type=int, default=512)
    c.add_argument('--max-combos', type=int, default=len(ALL_COMBOS))
    c.add_argument('--sw-stride', type=int, default=None,
                   help='solo sliding_window: stride en px (256 = 50%% solape). Default = solape mínimo.')
    c.set_defaults(func=cmd_construir)

    g = sub.add_parser('generar-todo', help='Construye todas las versiones de los 16 pacientes.')
    g.add_argument('--salida', type=Path, required=True)
    g.add_argument('--seed', type=int, default=SEED)
    g.add_argument('--patch-size', type=int, default=512)
    g.add_argument('--max-combos', type=int, default=len(ALL_COMBOS))
    g.set_defaults(func=cmd_generar_todo)

    d = sub.add_parser('dibujar', help='Dibuja un frame con sus parches.')
    d.add_argument('--scanner', required=True, choices=list(SCANNERS))
    d.add_argument('--paciente', required=True)
    d.add_argument('--version', required=True, choices=list(GENERADORES))
    d.add_argument('--frame', required=True)
    d.add_argument('--salida', type=Path, required=True, help='ruta del PNG')
    d.add_argument('--seed', type=int, default=SEED)
    d.add_argument('--patch-size', type=int, default=512)
    d.add_argument('--solo-rep0', action='store_true', help='oversampling: solo representante 0')
    d.set_defaults(func=cmd_dibujar)

    m = sub.add_parser('muestras', help='Un frame aleatorio por paciente+scanner.')
    m.add_argument('--salida', type=Path, required=True, help='raíz de Versiones_Paciente (para roles)')
    m.add_argument('--out-dir', type=Path, required=True, help='carpeta de los PNG')
    m.add_argument('--seed', type=int, default=SEED)
    m.add_argument('--patch-size', type=int, default=512)
    m.set_defaults(func=cmd_muestras)

    v = sub.add_parser('verificar', help='Barrido de estabilidad e invariantes.')
    v.add_argument('--salida', type=Path, required=True)
    v.add_argument('--patch-size', type=int, default=512)
    v.set_defaults(func=cmd_verificar)

    e = sub.add_parser('ensamblar', help='Ensambla los folds LOO (val=coverage, train=oversampling@K).')
    e.add_argument('--versiones-root', type=Path, required=True, help='raíz Versiones_Paciente')
    e.add_argument('--salida', type=Path, required=True, help='carpeta de salida de los folds')
    e.add_argument('--fold', help='num de paciente val (p.ej. 03). Si se omite, los 10 folds.')
    e.add_argument('--train-version', default='oversampling',
                   choices=['oversampling', 'representante_unico'],
                   help='versión de TRAIN: oversampling (K) o representante_unico (sin oversampling).')
    e.add_argument('--patch-size', type=int, default=512)
    e.set_defaults(func=cmd_ensamblar)

    y = sub.add_parser('yolo-labels', help='Escribe labels YOLO (.txt) de cada versión en su pool.')
    y.add_argument('--salida', type=Path, required=True, help='raíz Versiones_Paciente')
    y.set_defaults(func=cmd_yolo_labels)

    args = p.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
