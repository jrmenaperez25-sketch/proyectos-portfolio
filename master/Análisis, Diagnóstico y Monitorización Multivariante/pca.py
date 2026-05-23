import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from scipy.stats import f as f_dist, norm, chi2
import os

OUT = 'figures/preprocesado'
os.makedirs(OUT, exist_ok=True)

df        = pd.read_csv('../training.csv')
spec_cols = [c for c in df.columns if c.startswith('m')]
aux_cols  = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN',
             'REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
wn        = np.array([float(c[1:]) for c in spec_cols])
depth     = df['Depth'].values
n         = len(df)

pair_key = df[aux_cols].round(6).astype(str).agg('|'.join, axis=1)
groups   = pair_key.map({k: i for i, k in enumerate(pair_key.unique())}).values

X_raw  = df[spec_cols].values
scaler = StandardScaler().fit(X_raw)
X_sc   = scaler.transform(X_raw)

# ── 10-fold GroupKFold CV ─────────────────────────────────────────────────────
gkf   = GroupKFold(n_splits=10)
MAX_A = 10
evr_folds, q2_folds = [], []

for tr, te in gkf.split(X_raw, groups, groups):
    sc_cv  = StandardScaler().fit(X_raw[tr])
    X_tr   = sc_cv.transform(X_raw[tr])
    X_te   = sc_cv.transform(X_raw[te])
    pca_cv = PCA(n_components=MAX_A).fit(X_tr)
    evr_folds.append(pca_cv.explained_variance_ratio_ * 100)
    P_cv = pca_cv.components_
    TSS  = np.sum(X_te ** 2)
    q2_nc = [1 - np.sum((X_te - X_te @ P_cv[:nc].T @ P_cv[:nc]) ** 2) / TSS
             for nc in range(1, MAX_A + 1)]
    q2_folds.append(q2_nc)

evr_arr = np.array(evr_folds)
q2_arr  = np.array(q2_folds)

print('10-fold CV (pares geograficos):')
print(f'{"A":>3}  {"R2X_med%":>9}  {"Q2_med":>8}  {"Q2_std":>8}')
for a in range(MAX_A):
    print(f'{a+1:>3}  {evr_arr[:,a].mean():>9.2f}  {q2_arr[:,a].mean():>8.4f}  {q2_arr[:,a].std():>8.4f}')

# ── PCA modelo completo ───────────────────────────────────────────────────────
pca_full = PCA(n_components=50).fit(X_sc)
T_full   = pca_full.transform(X_sc)
P_full   = pca_full.components_
lam_all  = pca_full.explained_variance_
evr_full = pca_full.explained_variance_ratio_ * 100

p1 = P_full[0]
p2 = P_full[1]

# Proyeccion auxiliares: r(aux_j, t_a) escalada al espacio de loadings
X_aux_sc = StandardScaler().fit_transform(df[aux_cols].values)
corr_aux = np.corrcoef(X_aux_sc.T, T_full[:, :2].T)[:len(aux_cols), len(aux_cols):]
scale    = max(np.abs(p1).max(), np.abs(p2).max()) / max(np.abs(corr_aux).max(), 1e-9)
aux_p1   = corr_aux[:, 0] * scale
aux_p2   = corr_aux[:, 1] * scale

def region_color(w):
    if   w >= 4000: return '#4878CF'
    elif w >= 2000: return '#D65F5F'
    else:           return '#6ACC65'

reg_colors = np.array([region_color(w) for w in wn])
colors_d   = {'Topsoil': 'steelblue', 'Subsoil': 'tomato'}

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

# ── Fig 1: R²X acumulado + Q² CV — barras agrupadas ──────────────────────────
cum_r2x = np.cumsum(evr_full[:MAX_A])
q2_m    = q2_arr.mean(axis=0) * 100
x, width = np.arange(MAX_A), 0.38

fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - width/2, cum_r2x, width, color='steelblue', alpha=0.85,
            label='$R^2_X$ acumulado — modelo completo (%)')
b2 = ax.bar(x + width/2, q2_m,    width, color='tomato',    alpha=0.85,
            label='$Q^2$ CV medio (%)')
for bar, v in zip(b1, cum_r2x):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}',
            ha='center', va='bottom', fontsize=7.5, color='steelblue', fontweight='bold')
for bar, v in zip(b2, q2_m):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}',
            ha='center', va='bottom', fontsize=7.5, color='tomato', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'A={a}' for a in range(1, MAX_A+1)])
ax.set_ylabel('(%)', fontsize=12)
ax.set_title('$R^2_X$ acumulado (modelo completo) y $Q^2$ CV medio\n'
             'GroupKFold(10) por pares geograficos', fontweight='bold')
ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.14),
          ncol=2, framealpha=0.9)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/pca_cv.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 2: Score plot — elipse F (A=2) ───────────────────────────────────────
lam1, lam2 = lam_all[0], lam_all[1]
UCL_score  = (2*(n-1)/(n-2)) * f_dist.ppf(0.95, 2, n-2)

fig, ax = plt.subplots(figsize=(10, 8))
for grp, col in colors_d.items():
    mask = depth == grp
    ax.scatter(T_full[mask, 0], T_full[mask, 1],
               c=col, s=18, alpha=0.55, label=grp, edgecolors='none')
ell = mpatches.Ellipse((0, 0),
                        2*np.sqrt(UCL_score * lam1),
                        2*np.sqrt(UCL_score * lam2),
                        angle=0, edgecolor='black', facecolor='none', lw=2, ls='--')
ax.add_patch(ell)
ax.plot([], [], 'k--', lw=2, label='$T^2$ UCL 95% (F)')
ax.axhline(0, color='k', lw=0.6, ls='--'); ax.axvline(0, color='k', lw=0.6, ls='--')
ax.set_xlabel(f'PC1 ({evr_full[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({evr_full[1]:.1f}%)', fontsize=12)
ax.set_title('Score Plot PC1 vs PC2 — Topsoil vs Subsoil\n'
             'Elipse: $T^2 = \\frac{2(n-1)}{n-2}\\,F_{0.95,\\,2,\\,n-2}$',
             fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Parametros comunes para loading plots ─────────────────────────────────────
order    = np.argsort(wn)[::-1]
key_wn50 = np.arange(600, 7600, 50)

# ── Fig 3: Loading plot espectral (marcadores c/50, sin etiquetas) ────────────
fig, ax = plt.subplots(figsize=(13, 9))
ax.plot(p1[order], p2[order], color='gray', lw=0.5, alpha=0.3, zorder=1)
for i in range(0, len(order)-1, 350):
    ax.annotate('', xy=(p1[order[i+1]], p2[order[i+1]]),
                xytext=(p1[order[i]], p2[order[i]]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2), zorder=3)
ax.scatter(p1, p2, c=reg_colors, s=5, alpha=0.5, zorder=2)
for kw in key_wn50:
    idx = np.argmin(np.abs(wn - kw))
    ax.scatter(p1[idx], p2[idx], c=region_color(wn[idx]), s=25, zorder=5,
               edgecolors='black', linewidths=0.4)
    ax.annotate(f'{int(round(wn[idx]))}', (p1[idx], p2[idx]),
                fontsize=5, color='black', xytext=(3, 2), textcoords='offset points', zorder=6)
ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.set_xlabel(f'p$_1$  (PC1 = {evr_full[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'p$_2$  (PC2 = {evr_full[1]:.1f}%)', fontsize=12)
ax.set_title('Loading Plot — Espectro MIR\nMarcadores cada 50 cm$^{-1}$',
             fontweight='bold')
ax.legend(handles=legend_spec, fontsize=9, loc='best')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_loading1.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 4: Loading plot + auxiliares proyectadas ──────────────────────────────
fig, ax = plt.subplots(figsize=(13, 9))
ax.plot(p1[order], p2[order], color='gray', lw=0.5, alpha=0.3, zorder=1)
ax.scatter(p1, p2, c=reg_colors, s=5, alpha=0.5, zorder=2)
for kw in key_wn50:
    idx = np.argmin(np.abs(wn - kw))
    ax.scatter(p1[idx], p2[idx], c=region_color(wn[idx]), s=25, zorder=5,
               edgecolors='black', linewidths=0.4)
    ax.annotate(f'{int(round(wn[idx]))}', (p1[idx], p2[idx]),
                fontsize=5, color='black', xytext=(3, 2), textcoords='offset points', zorder=6)
for j, aux in enumerate(aux_cols):
    col = aux_col_map[aux]
    ax.scatter(aux_p1[j], aux_p2[j], c=col, s=80, zorder=7, marker='D',
               edgecolors='black', linewidths=0.5)
    ax.annotate(aux, (aux_p1[j], aux_p2[j]), fontsize=8, fontweight='bold',
                color=col, xytext=(5, 4), textcoords='offset points')
ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
ax.set_xlabel(f'p$_1$  (PC1 = {evr_full[0]:.1f}%)', fontsize=12)
ax.set_ylabel(f'p$_2$  (PC2 = {evr_full[1]:.1f}%)', fontsize=12)
ax.set_title('Loading Plot — Espectro MIR + Variables Auxiliares\n'
             'Auxiliares: r(aux, $t_a$) escalada', fontweight='bold')
legend_aux_p = [mpatches.Patch(color=col, label=name)
                for name, (col, _) in block_aux_colors.items()]
ax.legend(handles=legend_spec + legend_aux_p, fontsize=8, loc='best')
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_loading2.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 5: Zoom — 3 esquinas + zona TMAP/ELEV, etiquetas c/10 ────────────────
tmap_i = aux_cols.index('TMAP')
elev_i = aux_cols.index('ELEV')
txc    = (aux_p1[tmap_i] + aux_p1[elev_i]) / 2
tyc    = (aux_p2[tmap_i] + aux_p2[elev_i]) / 2
tr_aux = max(abs(aux_p1[tmap_i] - aux_p1[elev_i]),
             abs(aux_p2[tmap_i] - aux_p2[elev_i]), 0.005) * 4.0

# Zooms por direccion: top 20% de puntos en cada direccion diagonal
n_corner = max(int(len(p1) * 0.20), 50)

def dir_box(score, p1, p2, n_top, pad=0.10):
    top = np.argsort(score)[-n_top:]
    px, py = p1[top], p2[top]
    dx = max((px.max() - px.min()) * pad, 1e-4)
    dy = max((py.max() - py.min()) * pad, 1e-4)
    return px.min()-dx, px.max()+dx, py.min()-dy, py.max()+dy

x0_tr, x1_tr, y0_tr, y1_tr = dir_box( p1 + p2, p1, p2, n_corner)
x0_br, x1_br, y0_br, y1_br = dir_box( p1 - p2, p1, p2, n_corner)
x0_bl, x1_bl, y0_bl, y1_bl = dir_box(-p1 - p2, p1, p2, n_corner)

zoom_configs = [
    (x0_tr, x1_tr, y0_tr, y1_tr, 'Superior derecha'),
    (x0_br, x1_br, y0_br, y1_br, 'Inferior derecha'),
    (x0_bl, x1_bl, y0_bl, y1_bl, 'Inferior izquierda'),
    (txc-tr_aux, txc+tr_aux, tyc-tr_aux, tyc+tr_aux, 'Zona TMAP / ELEV'),
]
key_wn10 = np.arange(600, 7600, 50)

fig_z, axes_z = plt.subplots(2, 2, figsize=(14, 12))
for (x0, x1, y0, y1, ztitle), ax_z in zip(zoom_configs, axes_z.flat):
    dx = max((x1 - x0) * 0.10, 1e-4)
    dy = max((y1 - y0) * 0.10, 1e-4)
    xlim_z = (x0 - dx, x1 + dx)
    ylim_z = (y0 - dy, y1 + dy)

    in_box = ((p1 >= xlim_z[0]) & (p1 <= xlim_z[1]) &
              (p2 >= ylim_z[0]) & (p2 <= ylim_z[1]))
    ax_z.scatter(p1[in_box], p2[in_box],
                 c=[region_color(w) for w in wn[in_box]], s=8, alpha=0.55, zorder=2)

    for kw in key_wn10:
        idx = np.argmin(np.abs(wn - kw))
        if xlim_z[0] <= p1[idx] <= xlim_z[1] and ylim_z[0] <= p2[idx] <= ylim_z[1]:
            col = region_color(wn[idx])
            ax_z.scatter(p1[idx], p2[idx], c=col, s=45, zorder=5,
                         edgecolors='black', linewidths=0.5)
            ax_z.annotate(f'{int(round(wn[idx]))}', (p1[idx], p2[idx]),
                          fontsize=7, color='black', xytext=(3, 2),
                          textcoords='offset points')

    for j, aux in enumerate(aux_cols):
        if xlim_z[0] <= aux_p1[j] <= xlim_z[1] and ylim_z[0] <= aux_p2[j] <= ylim_z[1]:
            col = aux_col_map[aux]
            ax_z.scatter(aux_p1[j], aux_p2[j], c=col, s=130, zorder=7, marker='D',
                         edgecolors='black', linewidths=0.7)
            ax_z.annotate(aux, (aux_p1[j], aux_p2[j]), fontsize=9, fontweight='bold',
                         color=col, xytext=(6, 4), textcoords='offset points')

    ax_z.axhline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax_z.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
    ax_z.set_xlim(xlim_z); ax_z.set_ylim(ylim_z)
    ax_z.set_xlabel('p$_1$', fontsize=10); ax_z.set_ylabel('p$_2$', fontsize=10)
    ax_z.set_title(ztitle, fontweight='bold')
    ax_z.grid(alpha=0.3)

fig_z.suptitle('Zoom — 3 esquinas + Zona TMAP/ELEV  |  Etiquetas cada 10 cm$^{-1}$',
               fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_loading_zoom.png', dpi=150, bbox_inches='tight')
plt.close()

# ── T² (A=2) y SCR ───────────────────────────────────────────────────────────
A_ctrl   = 2
T_ctrl   = T_full[:, :A_ctrl]
lam_ctrl = lam_all[:A_ctrl]
lam_res  = lam_all[A_ctrl:]

T2        = np.sum((T_ctrl / np.sqrt(lam_ctrl)) ** 2, axis=1)
UCL_T2_95 = (A_ctrl*(n-1)/(n-A_ctrl)) * f_dist.ppf(0.95, A_ctrl, n-A_ctrl)

X_hat = T_ctrl @ P_full[:A_ctrl]
E     = X_sc - X_hat
SCR   = np.sum(E ** 2, axis=1)

# theta1 exacto via traza: total_var - suma primeros A autovalores
total_var  = np.sum(X_sc ** 2) / (n - 1)
theta1     = total_var - lam_all[:A_ctrl].sum()
theta2     = np.sum(lam_res ** 2)
theta3     = np.sum(lam_res ** 3)
h0         = 1 - 2*theta1*theta3 / (3*theta2**2)
z95        = norm.ppf(0.95)

print(f'  theta1={theta1:.2f}  theta2={theta2:.2f}  h0={h0:.4f}')
if h0 > 0:
    UCL_SCR_95 = theta1 * (z95*np.sqrt(2*theta2*h0**2)/theta1 + 1
                            + theta2*h0*(h0-1)/theta1**2) ** (1/h0)
    print('  Limite SCR: Jackson-Mudholkar')
else:
    # h0<=0 -> aprox. chi-cuadrado de Box (1954): siempre valida
    g_s = theta2 / theta1
    h_s = theta1 ** 2 / theta2
    UCL_SCR_95 = g_s * chi2.ppf(0.95, h_s)
    print(f'  Limite SCR: chi2 (Box 1954)  g={g_s:.2f}  h={h_s:.3f}')

out_t2  = np.where(T2  > 3 * UCL_T2_95)[0]
out_scr = np.where(SCR > 3 * UCL_SCR_95)[0]
out_all = np.union1d(out_t2, out_scr)

print(f'\nControl (A={A_ctrl}): T2_UCL95={UCL_T2_95:.2f}  SCR_UCL95={UCL_SCR_95:.4f}')
print(f'T2 > 3xUCL95 : {sorted(out_t2.tolist())}')
print(f'SCR > 3xUCL95: {sorted(out_scr.tolist())}')
print(f'Union        : {sorted(out_all.tolist())}')

# ── Fig 6: Mapa de influencia ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
for grp, col in colors_d.items():
    mask = depth == grp
    ax.scatter(T2[mask], SCR[mask], c=col, s=15, alpha=0.5, label=grp, edgecolors='none')
ax.axvline(UCL_T2_95,    color='orange',  lw=1.5, ls='--',
           label=f'$T^2$ UCL 95% = {UCL_T2_95:.1f}')
ax.axhline(UCL_SCR_95,   color='tomato',  lw=1.5, ls='--',
           label=f'SCR UCL 95% = {UCL_SCR_95:.2f}')
ax.axvline(3*UCL_T2_95,  color='darkred', lw=1.2, ls=':', label='$3\\times$ UCL 95%')
ax.axhline(3*UCL_SCR_95, color='darkred', lw=1.2, ls=':')
for i in out_all:
    ax.scatter(T2[i], SCR[i], c='black', s=80, zorder=6, marker='*')
    ax.annotate(str(i), (T2[i], SCR[i]), fontsize=8, fontweight='bold',
                xytext=(5, 4), textcoords='offset points')
ax.set_xlabel(f'$T^2$ de Hotelling  (A={A_ctrl})', fontsize=12)
ax.set_ylabel(f'SCR  (A={A_ctrl})', fontsize=12)
ax.set_title(f'Mapa de influencia — $T^2$ vs SCR  (A={A_ctrl})\n'
             f'Anomalias > $3\\times$UCL 95%: {len(out_all)} obs',
             fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/pca_outlier.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\nPCA completado. Figuras en {OUT}/')
