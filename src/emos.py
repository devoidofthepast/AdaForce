"""EMOS spread calibration (docs/emos_registration_20261023.md). Predictive N(mu, sigma^2), mu = final-system product (mean of
the three seeds' s2_ut05, unchanged), sigma^2 = c + d * s^2 with s^2 the unbiased variance of the 21 members (3 seeds x centre
+ 6 ZFE). c, d >= 0 per variable x lead, fitted on 2025 H1 (23 IFS ENS starts, GLO12 truth) by minimising the closed-form
normal CRPS on every 4th grid point (both directions) of the period's common support. Evaluated on the 2025 H2 holdout
(25 starts, 14 leads) and OceanBench 2024 (46 starts, 9 leads, GLO12-analysis truth, + WenHai MAE).
Metrics (d1-7, d5-9, d8-14): spread-skill (EMOS sigma rms / mu RMSE; raw: debiased member spread), 80%/90% interval coverage
and mean width (raw: member quantiles), CRPS (EMOS closed form vs raw fair CRPS); month-block paired bootstrap 4000.
Output: <DATA_E>/experiments/paper_budget/emos.json"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from paper_budget_curves import F_LIST, OB, VARS

FC = Path(DATA_D + '/dou_scs/fc_fields'); SEEDS = ('c44', 'c42', 'c43'); V = ('u', 'v', 'SST', 'SSS', 'SSH')
mask = np.load(DATA_D + '/dou_scs/mask.npy'); SUB4 = np.zeros_like(mask); SUB4[::4, ::4] = True
QZ = {80: norm.ppf(0.9), 90: norm.ppf(0.95)}


def crps_norm(mu, sig, y):
    sig = np.maximum(sig, 1e-6)
    z = (y - mu)/sig
    return sig*(z*(2*norm.cdf(z) - 1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi))


def fair_crps(X, y):
    M = X.shape[0]; Xs = np.sort(X, 0); k = np.arange(1, M + 1)[:, None]
    return np.abs(X - y[None]).mean(0) - ((2*k - M - 1)*Xs).sum(0)/(M*(M - 1))


def load_period(kind):
    """yields (start_date F, mu [NL,5,H,W], members [21,NL,5,H,W], truth [NL,5,H,W])"""
    if kind == '2024':
        now = {}
        for f in sorted(OB.glob('glo12_nowcast/*.npz')):
            z = np.load(f)
            for j, t in enumerate(z['times']):
                now[date.fromisoformat(str(t)[:10])] = np.stack([z[v][j] for v in VARS]).astype(np.float64)
        for F in F_LIST:
            z = np.load(FC/'fullsys/new'/f'{F:%Y%m%d}.npz'); mu = np.stack([z[v] for v in VARS], 1).astype(np.float64)
            mem = np.concatenate([np.load(FC/'fullsys_mem'/s/f'{F:%Y%m%d}.npz')['members'].astype(np.float64) for s in SEEDS])
            yield F, mu, mem, np.stack([now[F + timedelta(L)] for L in range(mu.shape[0])])
    else:
        import h2025_infer as H  # GLO12 analysis loader of the full evaluation code (not part of the core release)
        gl = H.Glo12(); sd, md = ('h2025_h1', 'h2025_h1_mem') if kind == 'h1' else ('h2025', 'h2025_mem')
        for p in sorted((FC/md/SEEDS[0]).glob('*.npz')):
            F = date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:]))
            mu = np.mean([np.stack([np.load(FC/sd/s/'s2_ut05'/p.name)[v] for v in VARS], 1) for s in SEEDS], 0).astype(np.float64)
            mem = np.concatenate([np.load(FC/md/s/p.name)['members'].astype(np.float64) for s in SEEDS])
            yield F, mu, mem, np.stack([gl.surf(F + timedelta(L)) for L in range(mu.shape[0])]).astype(np.float64)


def fit():
    rows = {}
    for F, mu, mem, y in load_period('h1'):
        s2 = mem.var(0, ddof=1)
        for L in range(mu.shape[0]):
            for k in range(5):
                ok = SUB4 & mask & np.isfinite(y[L, k]) & np.isfinite(mu[L, k]) & np.isfinite(s2[L, k])
                rows.setdefault((L, k), []).append(np.stack([mu[L, k][ok], s2[L, k][ok], y[L, k][ok]]))
        print('fit data', F, flush=True)
    par = np.zeros((14, 5, 2))
    for (L, k), r in rows.items():
        m, s2, y = np.concatenate(r, 1)
        f = lambda th: crps_norm(m, np.sqrt(np.exp(th[0]) + np.exp(th[1])*s2), y).mean()
        e2 = np.mean((m - y)**2); th0 = np.log([max(e2*0.5, 1e-8), 1.0])
        par[L, k] = np.exp(minimize(f, th0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-8, 'maxiter': 2000}).x)
    return par


def evaluate(kind, par, wh=False):
    NL = 9 if kind == '2024' else 14; agg = []                       # per start: [NL, 5, 11] sums
    starts = []
    import full_benchmark as B
    for F, mu, mem, y in load_period(kind):
        mu, mem, y = mu[:NL], mem[:, :NL], y[:NL]; s2 = mem.var(0, ddof=1); M = mem.shape[0]
        whf = B.load_sys(OB/'wenhai', F) if wh else None
        a = np.zeros((NL, 5, 11))
        for L in range(NL):
            for k in range(5):
                ok = mask & np.isfinite(y[L, k]) & np.isfinite(mu[L, k]) & np.isfinite(s2[L, k]) & (np.isfinite(whf[L, k]) if wh else True)
                m, v, t = mu[L, k][ok], s2[L, k][ok], y[L, k][ok]; X = mem[:, L, k][:, ok]
                sig = np.sqrt(par[L, k, 0] + par[L, k, 1]*v); e = t - m
                lo80, hi80 = np.quantile(X, [0.1, 0.9], 0); lo90, hi90 = np.quantile(X, [0.05, 0.95], 0)
                a[L, k] = [len(t), (e**2).sum(), (sig**2).sum(), (M + 1)/M*v.sum(), crps_norm(m, sig, t).sum(), fair_crps(X, t).sum(),
                           (np.abs(e) <= QZ[80]*sig).sum(), (np.abs(e) <= QZ[90]*sig).sum(), ((t >= lo90) & (t <= hi90)).sum(),
                           (2*QZ[90]*sig).sum(), np.abs(whf[L, k][ok] - t).sum() if wh else np.nan]
        agg.append(a); starts.append(F); print(kind, F, flush=True)
    A = np.stack(agg); months = np.array([F.month for F in starts]); um = np.unique(months); rng = np.random.default_rng(0)
    draws = [np.concatenate([np.where(months == mm)[0] for mm in rng.choice(um, len(um))]) for _ in range(4000)]
    res = {'n_starts': len(starts)}
    wins = {'d1_7': (0, 7), 'd5_9': (4, 9)} | ({'d8_14': (7, 14)} if NL == 14 else {})
    for wn, (l0, l1) in wins.items():
        S = A[:, l0:l1].sum((0, 1)); n = S[:, 0]
        res[wn] = {v: {'spread_skill_emos': float(np.sqrt(S[k, 2]/S[k, 1])), 'spread_skill_raw': float(np.sqrt(S[k, 3]/S[k, 1])),
                       'cover80_emos': float(S[k, 6]/n[k]), 'cover90_emos': float(S[k, 7]/n[k]), 'cover90_raw': float(S[k, 8]/n[k]),
                       'width90_emos': float(S[k, 9]/n[k]), 'crps_emos': float(S[k, 4]/n[k]), 'crps_raw': float(S[k, 5]/n[k]),
                       'mae_wenhai': float(S[k, 10]/n[k]) if wh else None} for k, v in enumerate(V)}
        for tag, (num, den) in (('crps_emos_vs_raw', (4, 5)),) + ((('crps_emos_vs_wenhai_mae', (4, 10)),) if wh else ()):
            f = lambda ix: 100*(A[ix, l0:l1, :, num].sum((0, 1))/A[ix, l0:l1, :, den].sum((0, 1)) - 1)
            pt = f(np.arange(len(starts))); bs = np.array([f(ix) for ix in draws]); lo, hi = np.percentile(bs, [2.5, 97.5], 0)
            res[wn][tag] = {v: [float(pt[k]), float(lo[k]), float(hi[k])] for k, v in enumerate(V)}
    return res


def main():
    par = fit()
    res = {'params_c_d_by_lead_var': par.tolist(), 'holdout_2025h2': evaluate('h2', par), 'oceanbench_2024': evaluate('2024', par, wh=True)}
    Path(DATA_E + '/experiments/paper_budget/emos.json').write_text(json.dumps(res, indent=1))
    for per in ('holdout_2025h2', 'oceanbench_2024'):
        r = res[per]['d5_9']; print(per, {v: (round(r[v]['spread_skill_raw'], 2), round(r[v]['spread_skill_emos'], 2), round(r[v]['cover90_emos'], 3)) for v in V}, flush=True)


if __name__ == '__main__':
    main()
