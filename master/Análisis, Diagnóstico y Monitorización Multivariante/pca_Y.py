import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from scipy.stats import f as f_dist
import os

OUT = 'figures/pca_Y'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv('../training.csv')
aux_cols = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN',
            'REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
depth  = df['Depth'].values
colors_d = {'Topsoil': 'steelblue', 'Subsoil': 'tomato'}

pair_key = df[aux_cols].round(6).astype(str).agg('|'.join, axis=1)
groups   = pair_key.map({k: i for i, k in enumerate(pair_key.unique())}).values

configs = [
    (['Ca', 'P', 'pH', 'SOC', 'Sand'], 'Todas las respuestas (con P)',      'conP'),
    (['Ca', 'pH', 'SOC', 'Sand'],       'Sin Fósforo (Ca, pH, SOC, Sand)',   'sinP'),
]

VAR_COLORS = {
    'Ca':   '#E6194B',
    'P':    '#3CB44B',
    'pH':   '#4363D8',
    'SOC':  '#F58231',
    'Sand': '#911EB4',
}

# ── Ajuste + CV ───────────────────────────────────────────────────────────────
results = {}

for targets, label, tag in configs:
    Y_raw = df[targets].values
    n     = len(Y_raw)
    MAX_A = len(targets)

    Y_sc = StandardScaler().fit_transform(Y_raw)

    gkf = GroupKFold(n_splits=10)
    evr_folds, q2_folds = [], []

    for tr, te in gkf.split(Y_raw, groups, groups):
        sc_cv  = StandardScaler().fit(Y_raw[tr])
        Y_tr   = sc_cv.transform(Y_raw[tr])
        Y_te   = sc_cv.transform(Y_raw[te])
        pca_cv = PCA(n_components=MAX_A).fit(Y_tr)
        evr_folds.append(pca_cv.explained_variance_ratio_ * 100)
        P_cv = pca_cv.components_
        TSS  = np.sum(Y_te ** 2)
        q2_nc = [1 - np.sum((Y_te - Y_te @ P_cv[:nc].T @ P_cv[:nc]) ** 2) / TSS
                 for nc in range(1, MAX_A + 1)]
        q2_folds.append(q2_nc)

    evr_arr = np.array(evr_folds)
    q2_arr  = np.array(q2_folds)

    pca   = PCA(n_components=MAX_A).fit(Y_sc)
    T     = pca.transform(Y_sc)
    P_mat = pca.components_          # (MAX_A, len(targets))
    lam   = pca.explained_variance_
    evr   = pca.explained_variance_ratio_ * 100

    # Correlation loadings: r(y_j, t_a) = sqrt(lambda_a) * p_aj
    corr_load = np.sqrt(lam[:, None]) * P_mat   # (MAX_A, len(targets))

    results[tag] = dict(T=T, P_mat=P_mat, corr_load=corr_load, lam=lam, evr=evr,
                        targets=targets, q2_arr=q2_arr, label=label, n=n)

    print(f'\n=== PCA Y — {label} ===')
    cum = np.cumsum(evr)
    print(f'{"A":>3}  {"R²Y%":>8}  {"R²Ycum%":>10}  {"Q²_med":>8}')
    for a in range(MAX_A):
        print(f'{a+1:>3}  {evr[a]:>8.2f}  {cum[a]:>10.2f}  {q2_arr[:,a].mean():>8.4f}')

# ── Fig 1: R²Y acumulado + Q² — 2 paneles ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (targets, label, tag) in zip(axes, configs):
    r  = results[tag]
    MA = len(targets)
    cum_r2 = np.cumsum(r['evr'])
    q2_m   = r['q2_arr'].mean(axis=0) * 100
    x, width = np.arange(MA), 0.38
    b1 = ax.bar(x - width/2, cum_r2, width, color='steelblue', alpha=0.85,
                label='$R^2_Y$ acum. (%)')
    b2 = ax.bar(x + width/2, q2_m,   width, color='tomato',    alpha=0.85,
                label='$Q^2$ CV (%)')
    for bar, v in zip(b1, cum_r2):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}',
                ha='center', va='bottom', fontsize=7.5, color='steelblue', fontweight='bold')
    for bar, v in zip(b2, q2_m):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}',
                ha='center', va='bottom', fontsize=7.5, color='tomato', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'A={i}' for i in range(1, MA + 1)])
    ax.set_ylabel('(%)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, axis='y')
fig.suptitle('PCA sobre variables respuesta — $R^2_Y$ acumulado y $Q^2$ CV (GroupKFold 10)',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/pcaY_cv.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 2: Score plots — 2 paneles ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, (targets, label, tag) in zip(axes, configs):
    r   = results[tag]
    T, lam, evr, n = r['T'], r['lam'], r['evr'], r['n']
    UCL = (2*(n-1)/(n-2)) * f_dist.ppf(0.95, 2, n-2)
    for grp, col in colors_d.items():
        mask = depth == grp
        ax.scatter(T[mask, 0], T[mask, 1], c=col, s=15, alpha=0.5,
                   label=grp, edgecolors='none')
    ell = mpatches.Ellipse((0, 0),
                            2*np.sqrt(UCL*lam[0]), 2*np.sqrt(UCL*lam[1]),
                            angle=0, edgecolor='black', facecolor='none', lw=2, ls='--')
    ax.add_patch(ell)
    ax.plot([], [], 'k--', lw=2, label='$T^2$ UCL 95%')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.set_xlabel(f'PC1 ({evr[0]:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({evr[1]:.1f}%)', fontsize=11)
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
fig.suptitle('Score Plot PCA sobre variables respuesta — Topsoil vs Subsoil',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/pcaY_scores.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 3: Correlation loading plots — 2 paneles ─────────────────────────────
# r(y_j, t_a) = sqrt(lambda_a) * p_aj  ∈ [-1, 1] → todas las flechas dentro del círculo unidad
theta = np.linspace(0, 2*np.pi, 300)
fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, (targets, label, tag) in zip(axes, configs):
    r   = results[tag]
    cl, evr = r['corr_load'], r['evr']
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, alpha=0.4)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    for j, var in enumerate(targets):
        col = VAR_COLORS[var]
        lx, ly = cl[0, j], cl[1, j]
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=2.5))
        sign_x = np.sign(lx) if abs(lx) > 0.02 else 1
        sign_y = np.sign(ly) if abs(ly) > 0.02 else 1
        ax.text(lx + 0.07*sign_x, ly + 0.06*sign_y,
                var, fontsize=13, fontweight='bold', color=col)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel(f'$r(y_j,\\ t_1)$  — PC1 = {evr[0]:.1f}%', fontsize=11)
    ax.set_ylabel(f'$r(y_j,\\ t_2)$  — PC2 = {evr[1]:.1f}%', fontsize=11)
    ax.set_title(label, fontweight='bold')
    ax.grid(alpha=0.2)
    ax.set_aspect('equal')
fig.suptitle('Correlation Loading Plot — $r(y_j,\\ t_a) = \\sqrt{\\lambda_a}\\,p_{aj}$',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/pcaY_loadings.png', dpi=150, bbox_inches='tight')
plt.close()

print(f'\nFiguras guardadas en {OUT}/')
print('  pcaY_cv.png | pcaY_scores.png | pcaY_loadings.png')
