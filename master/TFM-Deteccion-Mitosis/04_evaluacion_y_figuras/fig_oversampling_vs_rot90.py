#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import sys, math, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_REPO)):
    _p = os.path.join(_REPO, _d)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import parches_mitos as PM

OUT = Path(f'{WORKSPACE}/FIGURAS_TFM'); OUT.mkdir(exist_ok=True)
PATCH = PM.PATCH_SIZE
rng = random.Random(7)

SC = 'aperio'
elegido = None
for pac in ['A06', 'A07', 'A09', 'A16', 'A05', 'A11', 'A03', 'A10', 'A14', 'A15', 'A17']:
    try: frames = PM.load_patient_frames(SC, pac)
    except Exception: continue
    for fr in frames:
        W, H = fr['W'], fr['H']
        for (wx, wy, conf, mid) in fr['mit_pts']:
            if 420 <= wx <= W-420 and 420 <= wy <= H-420:
                elegido = (fr, wx, wy); break
        if elegido: break
    if elegido: break
assert elegido, 'no encontré mitosis interior'
fr, wx, wy = elegido
img = Image.open(fr['img_path']).convert('RGB'); W, H = fr['W'], fr['H']
bbox_r = PM.bbox_radius_for(SC)
print(f'mitosis en {fr["frame_stem"]} ({SC}) @ ({wx:.0f},{wy:.0f}) frame {W}x{H}, bbox_r {bbox_r}')

def marca(patch, pu, pv, color=(0, 200, 255)):
    p = patch.copy(); d = ImageDraw.Draw(p)
    for k in range(3):
        d.rectangle([pu-bbox_r-k, pv-bbox_r-k, pu+bbox_r+k, pv+bbox_r+k], outline=color)
    return p

combos = [(0, 'none'), (36, 'h'), (72, 'none'), (108, 'v'),
          (144, 'h'), (36, 'none'), (108, 'none'), (72, 'v'), (144, 'v')]
panelsA = []
for (ang, flip) in combos:
    c = PM.find_center_for_point(W, H, wx, wy, ang, flip, bbox_r, rng)
    if c is None:
        continue
    cx_c, cy_c = c
    patch = PM.extract_patch_img(img, cx_c, cy_c, ang, flip)
    pu, pv = PM.world_to_patch(wx, wy, cx_c, cy_c, ang, flip)
    panelsA.append((marca(patch, pu, pv), f'a{ang}° f:{flip}'))
    if len(panelsA) == 9: break

fig, axes = plt.subplots(3, 3, figsize=(9, 9.6))
for ax, (p, t) in zip(axes.ravel(), panelsA):
    ax.imshow(p); ax.set_title(t, fontsize=10); ax.axis('off')
fig.suptitle('Nuestro oversampling OFFLINE (misma mitosis)\n'
             'centro de recorte ALEATORIO + ángulos {0,36,72,108,144}° + flips  →  '
             'posición y CONTEXTO reales distintos, sin rellenar bordes',
             fontsize=12.5, y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT/'fig_oversampling_A_offline.png', dpi=140, bbox_inches='tight'); plt.close()

half = PATCH / 2
cx0 = cy0 = None
for _ in range(500):
    c = PM.find_center_for_point(W, H, wx, wy, 0, 'none', bbox_r, rng)
    if c is None:
        continue
    pu0, pv0 = PM.world_to_patch(wx, wy, c[0], c[1], 0, 'none')
    if math.hypot(pu0 - half, pv0 - half) > 130:
        cx0, cy0 = c; break
assert cx0 is not None, 'no logré parche descentrado'
base = PM.extract_patch_img(img, cx0, cy0, 0, 'none')
pu0, pv0 = PM.world_to_patch(wx, wy, cx0, cy0, 0, 'none')
print(f'  fig B: mitosis en parche @ ({pu0:.0f},{pv0:.0f}) de 512 (centro=256) -> descentrada')
base_m = np.asarray(marca(base, pu0, pv0, color=(255, 90, 90)))
panelsB = []
for k, deg in enumerate([0, 90, 180, 270]):
    rot = np.rot90(base_m, k)
    panelsB.append((rot, f'{deg}°'))

fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
for ax, (p, t) in zip(axes.ravel(), panelsB):
    ax.imshow(p); ax.set_title(t, fontsize=11); ax.axis('off')
fig.suptitle('Solo rotación en múltiplos de 90° (mismo parche)\n'
             'el campo/fondo es la MISMA imagen girada → apenas añade variedad de contexto; '
             'ángulos no múltiplos de 90° obligarían a rellenar bordes',
             fontsize=12.5, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.86])
plt.savefig(OUT/'fig_oversampling_B_rot90.png', dpi=140, bbox_inches='tight'); plt.close()

print('OK ->', OUT/'fig_oversampling_A_offline.png')
print('OK ->', OUT/'fig_oversampling_B_rot90.png')
