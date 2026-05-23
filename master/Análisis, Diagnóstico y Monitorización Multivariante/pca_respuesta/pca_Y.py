import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from scipy.stats import f as f_dist
import os

os.makedirs('figures', exist_ok=True)

df = pd.read_csv('../../training.csv')
aux_cols = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN',
            'REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
depth    = df['Depth'].values
colors_d = {'Topsoil': 'steelblue', 'Subsoil': 'tomato'}

pair_key = df[aux_cols].round(6).astype(str).agg('|'.join, axis=1)
groups   = pair_key.map({k: i for i, k in enumerate(pair_key.unique())}).values

configs = [
    (['Ca', 'P', 'pH', 'SOC', 'Sand'], 'Todas las respuestas (con P)',    'conP'),
    (['Ca', 'pH', 'SOC', 'Sand'],       'Sin Fósforo (Ca, pH, SOC, Sand)', 'sinP'),
]

VAR_COLORS = {
    'Ca':   '#E6194B',
    'P':    '#3CB44B',
    'pH':   '#4363D8',
    'SOC':  '#F58231',
    'Sand': '#911EB4',
}

# ── Ajuste ────────────────────────────────────────────────────────────────────
results = {}

for targets, label, tag in configs:
    Y_raw = df[targets].values
    n     = len(Y_raw)
    MAX_A = len(targets)

    Y_sc  = StandardScaler().fit_transform(Y_raw)
    pca   = PCA(n_components=MAX_A).fit(Y_sc)
    T     = pca.transform(Y_sc)
    P_mat = pca.components_          # (MAX_A, nvars)
    lam   = pca.explained_variance_
    evr   = pca.explained_variance_ratio_ * 100

    # r(y_j, t_a) = sqrt(lambda_a) * p_aj  ∈ [-1,1]
    corr_load = np.sqrt(lam[:, None]) * P_mat

    results[tag] = dict(T=T, corr_load=corr_load, lam=lam, evr=evr,
                        targets=targets, label=label, n=n)

    print(f'\n=== PCA Y — {label} ===')
    cum = np.cumsum(evr)
    print(f'{"A":>3}  {"R²Y%":>8}  {"R²Ycum%":>10}')
    for a in range(MAX_A):
        print(f'{a+1:>3}  {evr[a]:>8.2f}  {cum[a]:>10.2f}')

# ── Fig 1: Score plots ────────────────────────────────────────────────────────
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
fig.suptitle('Score Plot PCA variables respuesta — Topsoil vs Subsoil',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('figures/pcaY_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print('\nGuardado: figures/pcaY_scores.png')

# ── Fig 2: Correlation loading plots ─────────────────────────────────────────
theta = np.linspace(0, 2*np.pi, 300)
fig, axes = plt.subplots(1, 2, figsize=(13, 7))
for ax, (targets, label, tag) in zip(axes, configs):
    r        = results[tag]
    cl, evr  = r['corr_load'], r['evr']
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, alpha=0.4)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.5)
    for j, var in enumerate(targets):
        col    = VAR_COLORS[var]
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
plt.savefig('figures/pcaY_loadings.png', dpi=150, bbox_inches='tight')
plt.close()
print('Guardado: figures/pcaY_loadings.png')
