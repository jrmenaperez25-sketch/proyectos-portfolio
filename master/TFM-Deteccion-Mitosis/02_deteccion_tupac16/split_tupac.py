#!/usr/bin/env python3
import os
WORKSPACE = os.environ.get("TFM_WORKSPACE", "/workspace")
import csv
import json
from pathlib import Path

GT = Path(f'{WORKSPACE}/EXPERIMENTO_TUPAC73/gt/resumen_gt.csv')
OUT = Path(f'{WORKSPACE}/EXPERIMENTO_TUPAC73/split'); OUT.mkdir(parents=True, exist_ok=True)
QUOTAS = {'train': 16, 'val': 3, 'test': 4}

def main():
    rows = [r for r in csv.DictReader(open(GT)) if r['caso'] != 'TOTAL']
    info = {r['caso']: {'mitosis': int(r['mitosis']), 'hard': int(r['hard_negatives']),
                        'hpf': int(r['hpf_con_anotacion'])} for r in rows}
    order = sorted(info, key=lambda c: (-info[c]['mitosis'], -info[c]['hard'], c))

    counts = {k: 0 for k in QUOTAS}
    assign = {}
    for caso in order:
        cand = [k for k in QUOTAS if counts[k] < QUOTAS[k]]
        k = min(cand, key=lambda k: (counts[k] / QUOTAS[k], -QUOTAS[k]))
        assign[caso] = k; counts[k] += 1

    split = {k: sorted(c for c in assign if assign[c] == k) for k in QUOTAS}
    (OUT / 'split.json').write_text(json.dumps(split, indent=2))

    with open(OUT / 'split_stats.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['split', 'n_pac', 'pacientes', 'mitosis', 'hard_negatives', 'hpf'])
        for k in ('train', 'val', 'test'):
            cs = split[k]
            m = sum(info[c]['mitosis'] for c in cs)
            h = sum(info[c]['hard'] for c in cs)
            hp = sum(info[c]['hpf'] for c in cs)
            w.writerow([k, len(cs), '|'.join(cs), m, h, hp])
            print(f'  {k:5}: {len(cs):2} pac | mit={m:4} hard={h:4} hpf={hp:3} | {cs}')
    print(f'-> {OUT/"split.json"}')

if __name__ == '__main__':
    main()
