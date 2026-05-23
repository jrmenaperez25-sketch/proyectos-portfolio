import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from scipy.stats import f as f_dist, norm, chi2
import os

OUT = 'figures/modelo_limpio'
os.makedirs(OUT, exist_ok=True)

df        = pd.read_csv('../training.csv')
spec_cols = [c for c in df.columns if c.startswith('m')]
aux_cols  = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN',
             'REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
wn        = np.array([float(c[1:]) for c in spec_cols])

X_raw_all = df[spec_cols].values
n_total   = len(X_raw_all)

# ── Eliminacion iterativa: T² (A=2) > 3×UCL95% ───────────────────────────────
mask_clean  = np.ones(n_total, dtype=bool)
removed_idx = []

print('Eliminacion iterativa — T2 (A=2) > 3 x UCL95%')
print(f'{"Iter":>5}  {"n":>5}  {"UCL95":>7}  {"3xUCL":>8}  {"n>3x":>6}  {"removed":>8}')

iteration = 0
while True:
    iteration += 1
    cur_idx  = np.where(mask_clean)[0]
    X_cur    = X_raw_all[cur_idx]
    n_cur    = len(X_cur)

    sc_it  = StandardScaler().fit(X_cur)
    X_sc_it = sc_it.transform(X_cur)
    pca_it  = PCA(n_components=2).fit(X_sc_it)
    T_it    = pca_it.transform(X_sc_it)
    lam_it  = pca_it.explained_variance_

    T2_it  = np.sum((T_it / np.sqrt(lam_it)) ** 2, axis=1)
    UCL_it = (2*(n_cur-1)/(n_cur-2)) * f_dist.ppf(0.95, 2, n_cur-2)
    above  = np.where(T2_it > 3 * UCL_it)[0]

    if len(above) == 0:
        print(f'{iteration:>5}  {n_cur:>5}  {UCL_it:>7.2f}  {3*UCL_it:>8.2f}  {0:>6}  {"—":>8}')
        print(f'\nConvergido en {iteration} iteraciones.')
        print(f'Eliminadas: {len(removed_idx)} observaciones -> indices: {removed_idx}')
        break

    worst_local = above[np.argmax(T2_it[above])]
    worst_orig  = int(cur_idx[worst_local])
    print(f'{iteration:>5}  {n_cur:>5}  {UCL_it:>7.2f}  {3*UCL_it:>8.2f}  {len(above):>6}  {worst_orig:>8}')
    mask_clean[worst_orig] = False
    removed_idx.append(worst_orig)

# ── Datos limpios ─────────────────────────────────────────────────────────────
df_clean  = df[mask_clean].reset_index(drop=True)
depth_c   = df_clean['Depth'].values
X_clean   = X_raw_all[mask_clean]
n_c       = len(X_clean)

pair_key_c = df_clean[aux_cols].round(6).astype(str).agg('|'.join, axis=1)
groups_c   = pair_key_c.map({k: i for i, k in enumerate(pair_key_c.unique())}).values

sc_final = StandardScaler().fit(X_clean)
X_sc_c   = sc_final.transform(X_clean)

# ── CV en datos limpios ───────────────────────────────────────────────────────
gkf   = GroupKFold(n_splits=10)
MAX_A = 10
evr_folds_c, q2_folds_c = [], []

for tr, te in gkf.split(X_clean, groups_c, groups_c):
    sc_cv  = StandardScaler().fit(X_clean[tr])
    X_tr   = sc_cv.transform(X_clean[tr])
    X_te   = sc_cv.transform(X_clean[te])
    pca_cv = PCA(n_components=MAX_A).fit(X_tr)
    evr_folds_c.append(pca_cv.explained_variance_ratio_ * 100)
    P_cv = pca_cv.components_
    TSS  = np.sum(X_te ** 2)
    q2_nc = [1 - np.sum((X_te - X_te @ P_cv[:nc].T @ P_cv[:nc]) ** 2) / TSS
             for nc in range(1, MAX_A + 1)]
    q2_folds_c.append(q2_nc)

evr_arr_c = np.array(evr_folds_c)
q2_arr_c  = np.array(q2_folds_c)

print(f'\n10-fold CV — modelo limpio (n={n_c}):')
print(f'{"A":>3}  {"R2X_med%":>9}  {"Q2_med":>8}')
for a in range(MAX_A):
    print(f'{a+1:>3}  {evr_arr_c[:,a].mean():>9.2f}  {q2_arr_c[:,a].mean():>8.4f}')

# ── PCA modelo completo limpio ────────────────────────────────────────────────
pca_c  = PCA(n_components=50).fit(X_sc_c)
T_c    = pca_c.transform(X_sc_c)
P_c    = pca_c.components_
lam_c  = pca_c.explained_variance_
evr_c  = pca_c.explained_variance_ratio_ * 100

p1c, p2c = P_c[0], P_c[1]

X_aux_sc_c = StandardScaler().fit_transform(df_clean[aux_cols].values)
corr_c     = np.corrcoef(X_aux_sc_c.T, T_c[:, :2].T)[:len(aux_cols), len(aux_cols):]
scale_c    = max(np.abs(p1c).max(), np.abs(p2c).max()) / max(np.abs(corr_c).max(), 1e-9)
aux_p1c    = corr_c[:, 0] * scale_c
aux_p2c    = corr_c[:, 1] * scale_c

def region_color(w):
    if   w >= 4000: return '#4878CF'
    elif w >= 2000: return '#D65F5F'
    else:           return '#6ACC65'

reg_colors_c = np.array([region_color(w) for w in wn])
colors_d     = {'Topsoil': 'steelblue', 'Subsoil': 'tomato'}

block_aux_colors = {
    'Topografia':   ('#FF7F0E', ['ELEV','RELI','CTI']),
    'Clima':        ('#2CA02C', ['TMAP','TMFI']),
    'LST':          ('#9467BD', ['LSTD','LSTN']),
    'Reflectancia': ('#8C564B', ['REF1','REF2','REF3','REF7']),
    'Albedo':       ('#E377C2', ['BSAN','BSAS','BSAV']),
    'Vegetacion':   ('#17BECF', ['EVI']),
}
aux_col_map = {c: col for _, (col, cols) in block_aux_colors.items() for c in cols}

legend_spec = [
    mpatches.Patch(color='#4878CF', label='$>$4000 cm$^{-1}$'),
    mpatches.Patch(color='#D65F5F', label='2000--4000 cm$^{-1}$'),
    mpatches.Patch(color='#6ACC65', label='$<$2000 cm$^{-1}$'),
]

# ── Fig 1: R²X + Q² CV (limpio) ───────────────────────────────────────────────
cum_r2x_c = np.cumsum(evr_c[:MAX_A])
q2_m_c    = q2_arr_c.mean(axis=0) * 100
x, width  = np.arange(MAX_A), 0.38

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width/2, cum_r2x_c, width, color='steelblue', alpha=0.85,
            label='$R^2_X$ acumulado — modelo limpio (%)')
b2 = ax.bar(x + width/2, q2_m_c,    width, color='tomato',    alpha=0.85,
            label='$Q^2$ CV medio (%)')
for bar, v in zip(b1, cum_r2x_c):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}',
            ha='center', va='bottom', fontsize=7.5, color='steelblue', fontweight='bold')
for bar, v in zip(b2, q2_m_c):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}',
            ha='center', va='bottom', fontsize=7.5, color='tomato', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'A={a}' for a in range(1, MAX_A+1)])
ax.set_ylabel('(%)', fontsize=12)
ax.set_title(f'$R^2_X$ acumulado y $Q^2$ CV medio — Modelo limpio (n={n_c})\n'
             f'Eliminadas: {len(removed_idx)} obs', fontweight='bold')
ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.14),
          ncol=2, framealpha=0.9)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/pca_cv.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 2: Score plot (limpio) ────────────────────────────────────────────────
lam1c, lam2c = lam_c[0], lam_c[1]
UCL_sc_c     = (2*(n_c-1)/(n_c-2)) * f_dist.ppf(0.95, 2, n_c-2)

fig, ax = plt.subplots(figsize=(10, 8))
for grp, col in colors_d.items():
    mask = depth_c == grp
    ax.scatter(T_c[mask, 0], T_c[mask, 1], c=col, s=18, alpha=0.55,
               label=grp, edgecolors='none')
ell = mpatches.Ellipse((0, 0),
                        2*np.sqrt(UCL_sc_c*lam1c), 2*np.sqrt(UCL_sc_c*lam2c),
                        angle=0, edgecolor='black', facecolor='none', lw=2, ls='--')
ax.add_patch(ell)
ax.plot([], [], 'k--', lw=2, label='$T^2$ UCL 95% (F)')
ax.axhline(0, color='k', lw=0.6, ls='--'); ax.axvline(0, color='k', lw=0.6, ls='--')
ax.set_xlabel(f'PC1 ({evr_c[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({evr_c[1]:.1f}%)', fontsize=12)
ax.set_title(f'Score Plot — Modelo limpio (n={n_c})\n'
             'Elipse: $T^2 = \\frac{2(n-1)}{n-2}\\,F_{{0.95,\\,2,\\,n-2}}$',
             fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 3: Loading plot espectral (limpio) ────────────────────────────────────
order_c  = np.argsort(wn)[::-1]
key_wn50 = np.arange(600, 7600, 50)

fig, ax = plt.subplots(figsize=(13, 9))
ax.plot(p1c[order_c], p2c[order_c], color='gray', lw=0.5, alpha=0.3, zorder=1)
for i in range(0, len(order_c)-1, 350):
    ax.annotate('', xy=(p1c[order_c[i+1]], p2c[order_c[i+1]]),
                xytext=(p1c[order_c[i]], p2c[order_c[i]]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2), zorder=3)
ax.scatter(p1c, p2c, c=reg_colors_c, s=5, alpha=0.5, zorder=2)
for kw in key_wn50:
    idx = np.argmin(np.abs(wn - kw))
    ax.scatter(p1c[idx], p2c[idx], c=region_color(wn[idx]), s=25, zorder=5,
               edgecolors='black', linewidths=0.4)
    ax.annotate(f'{int(round(wn[idx]))}', (p1c[idx], p2c[idx]),
                fontsize=5, color='black', xytext=(3, 2), textcoords='offset points', zorder=6)
ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.set_xlabel(f'p$_1$ (PC1={evr_c[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'p$_2$ (PC2={evr_c[1]:.1f}%)', fontsize=12)
ax.set_title(f'Loading Plot — Modelo limpio (n={n_c})', fontweight='bold')
ax.legend(handles=legend_spec, fontsize=9, loc='best')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_loading1.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 4: Loading + auxiliares (limpio) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
ax.plot(p1c[order_c], p2c[order_c], color='gray', lw=0.5, alpha=0.3, zorder=1)
ax.scatter(p1c, p2c, c=reg_colors_c, s=5, alpha=0.5, zorder=2)
for kw in key_wn50:
    idx = np.argmin(np.abs(wn - kw))
    ax.scatter(p1c[idx], p2c[idx], c=region_color(wn[idx]), s=25, zorder=5,
               edgecolors='black', linewidths=0.4)
    ax.annotate(f'{int(round(wn[idx]))}', (p1c[idx], p2c[idx]),
                fontsize=5, color='black', xytext=(3, 2), textcoords='offset points', zorder=6)
for j, aux in enumerate(aux_cols):
    col = aux_col_map[aux]
    ax.scatter(aux_p1c[j], aux_p2c[j], c=col, s=80, zorder=7, marker='D',
               edgecolors='black', linewidths=0.5)
    ax.annotate(aux, (aux_p1c[j], aux_p2c[j]), fontsize=8, fontweight='bold',
                color=col, xytext=(5, 4), textcoords='offset points')
ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.set_xlabel(f'p$_1$ (PC1={evr_c[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'p$_2$ (PC2={evr_c[1]:.1f}%)', fontsize=12)
ax.set_title(f'Loading Plot + Auxiliares — Modelo limpio (n={n_c})', fontweight='bold')
legend_aux_p = [mpatches.Patch(color=col, label=name)
                for name, (col, _) in block_aux_colors.items()]
ax.legend(handles=legend_spec + legend_aux_p, fontsize=8, loc='best')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_loading2.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 5: Zoom — 3 esquinas + TMAP/ELEV ─────────────────────────────────────
tmap_i = aux_cols.index('TMAP')
elev_i = aux_cols.index('ELEV')
txc    = (aux_p1c[tmap_i] + aux_p1c[elev_i]) / 2
tyc    = (aux_p2c[tmap_i] + aux_p2c[elev_i]) / 2
tr_aux = max(abs(aux_p1c[tmap_i] - aux_p1c[elev_i]),
             abs(aux_p2c[tmap_i] - aux_p2c[elev_i]), 0.005) * 4.0

n_corner = max(int(len(p1c) * 0.20), 50)

def dir_box(score, p1, p2, n_top, pad=0.10):
    top = np.argsort(score)[-n_top:]
    px, py = p1[top], p2[top]
    dx = max((px.max() - px.min()) * pad, 1e-4)
    dy = max((py.max() - py.min()) * pad, 1e-4)
    return px.min()-dx, px.max()+dx, py.min()-dy, py.max()+dy

x0_tr, x1_tr, y0_tr, y1_tr = dir_box( p1c + p2c, p1c, p2c, n_corner)
x0_br, x1_br, y0_br, y1_br = dir_box( p1c - p2c, p1c, p2c, n_corner)
x0_bl, x1_bl, y0_bl, y1_bl = dir_box(-p1c - p2c, p1c, p2c, n_corner)

zoom_c = [
    (x0_tr, x1_tr, y0_tr, y1_tr, 'Superior derecha'),
    (x0_br, x1_br, y0_br, y1_br, 'Inferior derecha'),
    (x0_bl, x1_bl, y0_bl, y1_bl, 'Inferior izquierda'),
    (txc-tr_aux, txc+tr_aux, tyc-tr_aux, tyc+tr_aux, 'Zona TMAP / ELEV'),
]
key_wn10 = np.arange(600, 7600, 50)

fig_z, axes_z = plt.subplots(2, 2, figsize=(14, 12))
for (x0, x1, y0, y1, ztitle), ax_z in zip(zoom_c, axes_z.flat):
    dx = max((x1-x0)*0.10, 1e-4)
    dy = max((y1-y0)*0.10, 1e-4)
    xlim_z = (x0-dx, x1+dx); ylim_z = (y0-dy, y1+dy)

    in_box = ((p1c >= xlim_z[0]) & (p1c <= xlim_z[1]) &
              (p2c >= ylim_z[0]) & (p2c <= ylim_z[1]))
    ax_z.scatter(p1c[in_box], p2c[in_box],
                 c=[region_color(w) for w in wn[in_box]], s=8, alpha=0.55, zorder=2)

    for kw in key_wn10:
        idx = np.argmin(np.abs(wn - kw))
        if xlim_z[0] <= p1c[idx] <= xlim_z[1] and ylim_z[0] <= p2c[idx] <= ylim_z[1]:
            col = region_color(wn[idx])
            ax_z.scatter(p1c[idx], p2c[idx], c=col, s=45, zorder=5,
                         edgecolors='black', linewidths=0.5)
            ax_z.annotate(f'{int(round(wn[idx]))}', (p1c[idx], p2c[idx]),
                          fontsize=7, color='black', xytext=(3, 2),
                          textcoords='offset points')

    for j, aux in enumerate(aux_cols):
        if xlim_z[0] <= aux_p1c[j] <= xlim_z[1] and ylim_z[0] <= aux_p2c[j] <= ylim_z[1]:
            col = aux_col_map[aux]
            ax_z.scatter(aux_p1c[j], aux_p2c[j], c=col, s=130, zorder=7, marker='D',
                         edgecolors='black', linewidths=0.7)
            ax_z.annotate(aux, (aux_p1c[j], aux_p2c[j]), fontsize=9, fontweight='bold',
                         color=col, xytext=(6, 4), textcoords='offset points')

    ax_z.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax_z.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax_z.set_xlim(xlim_z); ax_z.set_ylim(ylim_z)
    ax_z.set_xlabel('p$_1$', fontsize=10); ax_z.set_ylabel('p$_2$', fontsize=10)
    ax_z.set_title(ztitle, fontweight='bold')
    ax_z.grid(alpha=0.3)

fig_z.suptitle(f'Zoom — Modelo limpio (n={n_c})  |  Etiquetas cada 10 cm$^{{-1}}$',
               fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_loading_zoom.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 6: Mapa de influencia (limpio — T² con A=2) ──────────────────────────
lam_res_c = lam_c[2:]
T2_c      = np.sum((T_c[:, :2] / np.sqrt(lam_c[:2])) ** 2, axis=1)
UCL_T2_c  = (2*(n_c-1)/(n_c-2)) * f_dist.ppf(0.95, 2, n_c-2)

X_hat_c = T_c[:, :2] @ P_c[:2]
E_c     = X_sc_c - X_hat_c
SCR_c   = np.sum(E_c ** 2, axis=1)

total_var_c = np.sum(X_sc_c ** 2) / (n_c - 1)
th1 = total_var_c - lam_c[:2].sum()   # exacto via traza
th2 = np.sum(lam_res_c ** 2)
th3 = np.sum(lam_res_c ** 3)
h0  = 1 - 2*th1*th3 / (3*th2**2)
z95 = norm.ppf(0.95)
if h0 > 0:
    UCL_SCR_c = th1 * (z95*np.sqrt(2*th2*h0**2)/th1 + 1 + th2*h0*(h0-1)/th1**2)**(1/h0)
else:
    g_sc = th2 / th1
    h_sc = th1 ** 2 / th2
    UCL_SCR_c = g_sc * chi2.ppf(0.95, h_sc)

above_t2_c = np.where(T2_c > 3*UCL_T2_c)[0]

fig, ax = plt.subplots(figsize=(10, 8))
for grp, col in colors_d.items():
    mask = depth_c == grp
    ax.scatter(T2_c[mask], SCR_c[mask], c=col, s=15, alpha=0.5,
               label=grp, edgecolors='none')
ax.axvline(UCL_T2_c,    color='orange', lw=1.5, ls='--',
           label=f'$T^2$ UCL 95% = {UCL_T2_c:.1f}')
ax.axhline(UCL_SCR_c,   color='tomato', lw=1.5, ls='--',
           label=f'SCR UCL 95% = {UCL_SCR_c:.2f}')
ax.axvline(3*UCL_T2_c,  color='darkred', lw=1.2, ls=':', label='$3\\times$ UCL')
ax.axhline(3*UCL_SCR_c, color='darkred', lw=1.2, ls=':')
for i in above_t2_c:
    ax.scatter(T2_c[i], SCR_c[i], c='black', s=80, zorder=6, marker='*')
    ax.annotate(str(i), (T2_c[i], SCR_c[i]), fontsize=8, fontweight='bold',
                xytext=(5, 4), textcoords='offset points')
ax.set_xlabel('$T^2$ de Hotelling  (A=2)', fontsize=12)
ax.set_ylabel('SCR  (A=2)', fontsize=12)
ax.set_title(f'Mapa de influencia — Modelo limpio (n={n_c})\n'
             f'T2 > 3xUCL: {len(above_t2_c)} obs',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_outlier.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\nModelo limpio completado. Figuras en {OUT}/')
print(f'n_original={n_total}  n_limpio={n_c}  eliminadas={len(removed_idx)}')
