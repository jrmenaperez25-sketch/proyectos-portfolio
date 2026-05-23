import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

os.makedirs('figures/preprocesado', exist_ok=True)

df = pd.read_csv('../training.csv')

spec_cols = [c for c in df.columns if c.startswith('m')]
aux_cols  = ['BSAN','BSAS','BSAV','CTI','ELEV','EVI','LSTD','LSTN',
             'REF1','REF2','REF3','REF7','RELI','TMAP','TMFI']
targets   = ['Ca', 'P', 'pH', 'SOC', 'Sand']
wn        = np.array([float(c[1:]) for c in spec_cols])
depth     = df['Depth'].values

# Info geografica de parejas
pair_key    = df[aux_cols].round(6).astype(str).agg('|'.join, axis=1)
pair_counts = Counter(pair_key)
n_pairs     = sum(1 for v in pair_counts.values() if v == 2)
n_sing      = sum(1 for v in pair_counts.values() if v == 1)
n_gt2       = sum(1 for v in pair_counts.values() if v > 2)

print('=== RESUMEN EDA ===')
print(f'Muestras totales : {len(df)}')
print(f'  Topsoil        : {np.sum(depth=="Topsoil")}')
print(f'  Subsoil        : {np.sum(depth=="Subsoil")}')
print(f'Variables espectrales : {len(spec_cols)}')
print(f'Rango espectral       : {wn.min():.0f} - {wn.max():.0f} cm-1')
print(f'Ubicaciones unicas    : {len(pair_counts)}')
print(f'  Pares completos (n=2): {n_pairs}  -> {2*n_pairs} obs')
print(f'  Singletons (n=1)     : {n_sing}')
print(f'  Grupos (n>2)         : {n_gt2}')

print('\nVariables respuesta:')
for t in targets:
    print(f'  {t:5s}  media={df[t].mean():.4f}  std={df[t].std():.4f}  '
          f'min={df[t].min():.4f}  max={df[t].max():.4f}  asim={df[t].skew():.2f}')

# ── Fig 1: Todos los espectros individuales + media por profundidad ───────────
X     = df[spec_cols].values
depth = df['Depth'].values

colors_depth = {'Topsoil': ('#4878CF', 0.06), 'Subsoil': ('#D65F5F', 0.06)}

fig, ax = plt.subplots(figsize=(13, 5))

# Regiones espectrales de fondo
regions = [(4000, wn.max(), '#4878CF', 0.07),
           (2000, 4000,     '#D65F5F', 0.07),
           (wn.min(), 2000, '#6ACC65', 0.07)]
for lo, hi, col, alp in regions:
    mask = (wn >= lo) & (wn <= hi)
    ax.axvspan(wn[mask].min(), wn[mask].max(), color=col, alpha=alp)

# Espectros individuales (todos, muy finos y transparentes)
for grp, (col, alp) in colors_depth.items():
    idx_g = np.where(depth == grp)[0]
    for i in idx_g:
        ax.plot(wn, X[i], color=col, lw=0.3, alpha=alp)

# Media por grupo encima
mu_top = X[depth == 'Topsoil'].mean(axis=0)
mu_sub = X[depth == 'Subsoil'].mean(axis=0)
ax.plot(wn, mu_top, color='#2255AA', lw=1.6, label='Media Topsoil')
ax.plot(wn, mu_sub, color='#AA2222', lw=1.6, label='Media Subsoil')

ax.set_xlim(wn.max(), wn.min())
ax.set_xlabel('Numero de onda (cm$^{-1}$)', fontsize=12)
ax.set_ylabel('Absorbancia (log$_{10}$ I$_0$/I)', fontsize=12)
ax.set_title('Espectros MIR brutos — 1157 muestras de suelo africano\n'
             'Datos originales antes de cualquier preprocesado', fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/preprocesado/eda_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print('Guardado: eda_spectrum.png')


print('\nEDA completado.')
