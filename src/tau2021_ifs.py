"""IFS transferability using only 2021 forecasts (decision information available by end of 2021).
A = 2021 starts 1..12 (Jan 1 - Jun 5), B = starts 13..25 (Jun 18 - Oct 13); same statistic and bootstrap as work/tau_boot.py."""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import json, numpy as np
z = np.load(DATA_D + '/dou_scs/tau_cache/ifs.npz'); res = {}
nA = z['A_Qnet'].shape[0]; print('n_A total', nA, 'n_B', z['B_Qnet'].shape[0])
for split in ('chrono', 'alt'):
    for k, comps in (('Qnet', ('Qnet',)), ('momentum', ('tau_u', 'tau_v')), ('EmP', ('EmP',))):
        x = np.concatenate([z[f'A_{c}'][:25] for c in comps], 2).astype(np.float64)
        ia, ib = (np.arange(12), np.arange(12, 25)) if split == 'chrono' else (np.arange(0, 25, 2), np.arange(1, 25, 2))
        a, b = x[ia], x[ib]; sb2 = (b**2).sum((1, 2)); rng = np.random.default_rng(0)
        tau = lambda ja, jb: float(1 - ((b[jb] - a[ja].mean(0))**2).sum()/sb2[jb].sum())
        pt = tau(np.arange(len(a)), np.arange(len(b)))
        bs = [tau(rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))) for _ in range(4000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        corr = float(np.corrcoef(a.mean(0).ravel(), b.mean(0).ravel())[0, 1])
        res[f'{split}_{k}'] = dict(tau=round(pt, 4), ci95=[round(float(lo), 4), round(float(hi), 4)], on=bool(lo > 0), corr=round(corr, 3))
        print(split, k, res[f'{split}_{k}'], flush=True)
json.dump(res, open('tau2021_ifs.json', 'w'), indent=1)
