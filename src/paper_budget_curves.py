"""OceanBench 2024 at native 1/12 deg (CPU only): lead-by-lead RMSE (days 1..10) of the external systems, G track (truth =
GLO12 nowcast daily series) and R track (truth = GLORYS 2024 from the Dou cache), plus persistence. 46 starts
F = 2024-01-31..2024-12-11 weekly; cos-lat weights.
Revised after the internal review 2026-10-13 (F6-F8):
  * frozen common support = Dou ocean mask & every system (GLO12, WenHai, XiHe, LangYa, GLONET) finite at EVERY lead of
    EVERY start & both truths finite on every day used (t-1 .. F+9); saved to support.npy and used by ob12_eval.py;
    any non-finite value inside it raises (no silent zero-error points);
  * GLONET (1/4-deg nodes) bilinear to 1/12 deg (protocol section 2);
  * per-origin weighted MSE [start, lead, variable] saved (ext_per_origin.npz) for the paired month-block bootstrap;
  * d1-7: both the pooled RMSE (sqrt of the mean MSE over leads 1-7) and the mean of the per-lead RMSEs.
Output: <DATA_E>/experiments/paper_budget/{curves.json, support.npy, ext_per_origin.npz}"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

OB = Path(DATA_D + '/oceanbench_scs')
ROOT = Path(DATA_D + '/dou_scs')
OUT = Path(DATA_E + '/experiments/paper_budget')
SYS = ('glo12', 'wenhai', 'xihe', 'langya', 'glonet')
VARS = ('uo', 'vo', 'thetao', 'so', 'zos')
F_LIST = [date(2024, 1, 31) + timedelta(7*k) for k in range(46)]
NL = 10


def load_system(s, F, lat, lon):
    """(dates, [n, 5, 301, 301]) on the Dou grid, or None."""
    p = OB/s/f'{F:%Y%m%d}.npz'
    if not p.exists():
        return None
    z = np.load(p); times = [date.fromisoformat(str(t)[:10]) for t in z['times']]
    x = np.stack([z[v] for v in VARS], 1).astype(np.float32)
    if x.shape[-2:] != (301, 301):                                   # GLONET 1/4-deg nodes -> bilinear to 1/12
        la, lo = np.asarray(z['lat'], np.float64), np.asarray(z['lon'], np.float64)
        o = np.argsort(la); pts = np.stack(np.meshgrid(lat, lon, indexing='ij'), -1)
        y = np.empty((len(times), 5, 301, 301), np.float32)
        for i in range(len(times)):
            for k in range(5):
                y[i, k] = RegularGridInterpolator((la[o], lo), x[i, k][o], bounds_error=False, fill_value=np.nan)(pts)
        x = y
    return times, x


def truths():
    meta = json.loads((ROOT/'meta.json').read_text()); D0 = date.fromisoformat(meta['first_day'])
    oc = np.memmap(ROOT/'ocean.f32', np.float32, 'r', shape=(meta['N'], 5, 301, 301))
    now = {}
    for f in sorted(OB.glob('glo12_nowcast/*.npz')):
        z = np.load(f)
        for j, t in enumerate(z['times']):
            now[date.fromisoformat(str(t)[:10])] = np.stack([z[v][j] for v in VARS])
    return {'G': lambda d: now.get(d), 'R': lambda d: np.asarray(oc[(d - D0).days])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mask = np.load(ROOT/'mask.npy'); lat = np.load(ROOT/'lat.npy'); lon = np.load(ROOT/'lon.npy')
    tr = truths()
    fc = {s: {F: load_system(s, F, lat, lon) for F in F_LIST} for s in SYS}
    missing = {s: [str(F) for F in F_LIST if fc[s][F] is None or not all(F + timedelta(L) in fc[s][F][0] for L in range(NL))] for s in SYS}
    support = mask.copy()
    for s in SYS:
        for F, v in fc[s].items():
            if v is not None:
                times, x = v
                for L in range(NL):
                    if F + timedelta(L) in times:
                        support &= np.isfinite(x[times.index(F + timedelta(L))]).all(0)
    days = sorted({F + timedelta(L) for F in F_LIST for L in range(-2, NL)})
    no_truth = {}
    for track in ('G', 'R'):
        for d in days:
            y = tr[track](d)
            if y is None:
                no_truth.setdefault(track, []).append(str(d))
            else:
                support &= np.isfinite(y).all(0)
    np.save(OUT/'support.npy', support)
    w = np.cos(np.deg2rad(lat))[:, None]*support; w = w/w.sum()
    res = {'n_points': int(support.sum()), 'n_points_dou_mask': int(mask.sum()), 'missing_forecasts': missing,
           'missing_truth_days': no_truth, 'G': {}, 'R': {}}
    per = {}
    for track in ('G', 'R'):
        for s in SYS + ('persistence',):
            mse = np.full((len(F_LIST), NL, 5), np.nan)
            for si, F in enumerate(F_LIST):
                for L in range(NL):
                    d = F + timedelta(L); y = tr[track](d)
                    if s == 'persistence':
                        x = tr[track](F - timedelta(1))
                    else:
                        v = fc[s][F]
                        x = None if v is None or d not in v[0] else v[1][v[0].index(d)]
                    if x is None or y is None:
                        continue
                    e = (x - y)[:, support]
                    if not np.isfinite(e).all():
                        raise ValueError(f'{s} {track} {F} lead {L + 1}: {int((~np.isfinite(e)).sum())} non-finite values inside the support')
                    mse[si, L] = (w[support]*e**2).sum(1)
            per[f'{track}_{s}'] = mse
            n = np.isfinite(mse[:, :, 0]).sum(0)
            r = {v: [round(float(np.sqrt(np.nanmean(mse[:, L, k]))), 4) if n[L] else None for L in range(NL)] for k, v in enumerate(VARS)}
            r['n_starts_per_lead'] = n.tolist()
            r['d1_7_pooled'] = {v: round(float(np.sqrt(np.nanmean(mse[:, :7, k]))), 4) for k, v in enumerate(VARS)}
            r['d1_7_mean_of_leads'] = {v: round(float(np.mean(r[v][:7])), 4) if None not in r[v][:7] else None for v in VARS}
            res[track][s] = r
    np.savez(OUT/'ext_per_origin.npz', starts=np.array([str(F) for F in F_LIST]), vars=np.array(VARS), **per)
    (OUT/'curves.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'n_points': res['n_points'], 'n_points_dou_mask': res['n_points_dou_mask'],
                      'missing': {s: len(v) for s, v in missing.items()}, 'missing_truth': {k: len(v) for k, v in no_truth.items()}}))
    for track in ('G', 'R'):
        print('==', track)
        for s, r in res[track].items():
            print(s, {v: [r[v][i] for i in (0, 2, 6, 8)] for v in ('thetao', 'uo', 'zos')})


if __name__ == '__main__':
    main()
