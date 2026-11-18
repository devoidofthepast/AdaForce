"""OceanBench 2024 at native 1/12 deg (docs/oceanbench_1_12_protocol_draft_20261012.md) for Dou-backbone models.
Starts: F = 2024-01-31 + 7k (46); t = F - 1. Initial ocean (t-1, t): G track = GLO12 nowcast daily fields, R track = GLORYS
(Dou cache). Atmosphere: ERA5 (0.25-deg cache, SepInterp to 1/12) for days <= t; for days t+1..t+9 the forecast forcing
--atm ifs (OceanBench file {F}.npz, index i = day t + i; bilinear to 1/12; sowinu10/sowinv10/sotemair/sotemhum (dew
point)/sosudosw/sosudolw/sowaprec) or era5 (reference forcing). Live COARE fluxes each step (coare_torch), 9-day rollout.
Optional --shrink BETA_JSON (per-lead per-channel beta, forecast days only).
Revised after the internal review 2026-10-13:
  * F6: scores on the frozen common support of paper_budget_curves.py (support.npy; run that first); a non-finite
    prediction or truth inside it raises (no silent zero-error points);
  * F7: per-origin weighted MSE [start, lead, variable] saved (ob12_{tag}_per_origin.npz) for the paired month-block
    bootstrap; d1-7 reported both pooled and as the mean of the per-lead RMSEs;
  * F4: --atm both runs IFS and ERA5 forcing from the same start and saves, per origin/lead/variable, E[e_era5^2],
    E[e_ifs^2], E[delta^2] and E[e_era5*delta] (delta = ifs - era5 prediction): MSE_ifs - MSE_era5 = E[delta^2] + 2E[e0 delta],
    i.e. a reference-forcing SENSITIVITY decomposition, not an upper bound on what forcing correction can gain.
Usage: python ob12_eval.py --run seed42 [--atm ifs|era5|both] [--lt k|one] [--shrink path] [--tag name]
Output: <DATA_E>/experiments/paper_budget/ob12_{tag}.json (+ _per_origin.npz)"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import coare_torch as CT
import dou_scs_feet as FE
from dou_scs_train import D0, N, ROOT, DouScs, ScsDou, norm_stats
from paper_budget_curves import F_LIST, VARS

OB = Path(DATA_D + '/oceanbench_scs')
OUT = Path(DATA_E + '/experiments/paper_budget')
IFSV = ('sowinu10', 'sowinv10', 'sotemair', 'sotemhum', 'sosudosw', 'sosudolw', 'sowaprec')
LEADS = 9


def ifs_on_grid(F, lat, lon):
    z = np.load(OB/'ifs'/f'{F:%Y%m%d}.npz')
    la, lo = np.asarray(z['lat'], np.float64), np.asarray(z['lon'], np.float64)
    o = np.argsort(la)
    pts = np.stack(np.meshgrid(lat, lon, indexing='ij'), -1)
    out = np.zeros((10, 7, 301, 301), np.float32)
    for c, v in enumerate(IFSV):
        x = np.asarray(z[v], np.float64)[:, o]
        for i in range(10):
            out[i, c] = RegularGridInterpolator((la[o], lo), x[i], bounds_error=False, fill_value=None)(pts)
    out[:, 6] = np.maximum(out[:, 6], 0.)
    return out                                                     # [day index i = t + i, 7 SI, H, W]


def summary(mse):
    """mse [S, L, 5] -> per-lead RMSE, pooled d1-7 and mean-of-leads d1-7."""
    r = {v: [round(float(np.sqrt(np.mean(mse[:, L, k]))), 4) for L in range(LEADS)] for k, v in enumerate(VARS)}
    r['d1_7_pooled'] = {v: round(float(np.sqrt(np.mean(mse[:, :7, k]))), 4) for k, v in enumerate(VARS)}
    r['d1_7_mean_of_leads'] = {v: round(float(np.mean(r[v][:7])), 4) for v in VARS}
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', default='seed42'); ap.add_argument('--lt', default='one', choices=('one', 'k'))
    ap.add_argument('--shrink', default=None); ap.add_argument('--shrink-channels', nargs='*', type=int, default=None); ap.add_argument('--tag', default=None)
    ap.add_argument('--max-starts', type=int, default=0)
    ap.add_argument('--atm', default='ifs', choices=('ifs', 'era5', 'both'))
    a = ap.parse_args()
    tag = a.tag or a.run
    modes = ('era5', 'ifs') if a.atm == 'both' else (a.atm,)
    st = norm_stats()
    lat, lon = np.load(ROOT/'lat.npy'), np.load(ROOT/'lon.npy'); mask_np = np.load(ROOT/'mask.npy')
    support = np.load(OUT/'support.npy')                            # frozen by paper_budget_curves.py
    oc = np.memmap(ROOT/'ocean.f32', np.float32, 'r', shape=(N, 5, 301, 301))
    era = FE.era5_025(); interp = FE.SepInterp()
    mask = torch.tensor(mask_np, device='cuda'); latg = torch.tensor(np.repeat(lat[:, None], 301, 1), device='cuda')
    sup = torch.tensor(support, device='cuda')
    w = np.cos(np.deg2rad(lat))[:, None]*support; w = torch.tensor(w/w.sum(), device='cuda')[sup]
    now = {}
    for f in sorted(OB.glob('glo12_nowcast/*.npz')):
        z = np.load(f)
        for j, t in enumerate(z['times']):
            now[date.fromisoformat(str(t)[:10])] = np.stack([z[v][j] for v in VARS]).astype(np.float32)
    starts = F_LIST[:a.max_starts] if a.max_starts else F_LIST
    g = lambda k: torch.tensor(st[k], device='cuda', dtype=torch.float32).view(-1, 1, 1)
    om, osd, fm, fsd, dm, dsd = g('ocean_mean'), g('ocean_std'), g('flux_mean'), g('flux_std'), g('diff_mean'), g('diff_std')
    cst = torch.tensor(ScsDou(D0 + timedelta(1), D0 + timedelta(1), st).const, device='cuda')
    if a.shrink:
        clim = np.memmap(ROOT/'flux_clim_doy.f32', np.float32, 'r', shape=(366, 8, 301, 301))
        beta = torch.tensor(json.loads(Path(a.shrink).read_text())['beta'], device='cuda', dtype=torch.float32)
    model = DouScs().cuda()
    model.load_state_dict(torch.load(ROOT/'runs'/a.run/'best.pt', map_location='cuda', weights_only=True)['model']); model.eval()

    def rollout(o_prev, o_curr, it, ifs, mode):
        def atm(d):
            k = d - it
            return interp(torch.tensor(np.asarray(era[d]), device='cuda')) if (k <= 0 or mode == 'era5') else ifs[k]

        def flux(o, d):
            f = CT.bulk_flux(o, FE.to_iface(atm(d)), latg, mask)
            k = d - it
            if a.shrink and k >= 1:
                c = torch.tensor(np.asarray(clim[(D0 + timedelta(d)).timetuple().tm_yday - 1]), device='cuda')
                b = beta[k - 1].clone()
                if a.shrink_channels is not None:
                    keep = torch.ones(8, dtype=torch.bool, device='cuda'); keep[a.shrink_channels] = False; b[keep] = 1.
                f = c + b.view(8, 1, 1)*(f - c)
            return torch.nan_to_num((f - fm)/fsd)
        preds = []
        f_prev = flux(o_prev, it - 1)
        with torch.no_grad():
            for k in range(1, LEADS + 1):
                cur, tgt = it + k - 1, it + k
                f_curr, f_futu = flux(o_curr, cur), flux(o_curr, tgt)
                dt = D0 + timedelta(tgt); nd = 366 if dt.year % 4 == 0 else 365; doy = dt.timetuple().tm_yday - 1
                tch = torch.stack([torch.full((301, 301), float(np.sin(doy*2*np.pi/nd)), device='cuda'), torch.full((301, 301), float(np.cos(doy*2*np.pi/nd)), device='cuda')])
                ctx = torch.cat([f_prev, torch.nan_to_num((o_prev - om)/osd), f_curr, torch.nan_to_num((o_curr - om)/osd), f_futu, cst, tch]).unsqueeze(0)
                lt = float(k) if a.lt == 'k' else 1.
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model({'y_context': ctx.float(), 'lt': torch.tensor([[lt]], device='cuda')})[0].float()
                o_next = torch.where(mask, o_curr + out.permute(2, 1, 0)*dsd + dm, torch.nan)
                preds.append(o_next[:, sup])
                f_prev, o_prev, o_curr = f_curr, o_curr, o_next
        return preds                                                # LEADS x [5, n_support]

    res, per = {'n_points': int(support.sum()), 'atm': a.atm, 'shrink': a.shrink, 'run': a.run, 'lt': a.lt}, {}
    for track in ('G', 'R'):
        truth = (lambda d: now.get(d)) if track == 'G' else (lambda d: np.asarray(oc[(d - D0).days]))
        mse = {m: np.zeros((len(starts), LEADS, 5)) for m in modes}
        if a.atm == 'both':
            d2, cross = np.zeros((len(starts), LEADS, 5)), np.zeros((len(starts), LEADS, 5))
        for si, F in enumerate(starts):
            t = F - timedelta(1); it = (t - D0).days
            o_prev, o_curr = truth(t - timedelta(1)), truth(t)
            if o_prev is None or o_curr is None:
                raise ValueError(f'{track} {F}: missing initial state')
            o_prev = torch.where(mask, torch.tensor(o_prev, device='cuda'), torch.nan); o_curr = torch.where(mask, torch.tensor(o_curr, device='cuda'), torch.nan)
            ifs = torch.tensor(ifs_on_grid(F, lat, lon), device='cuda') if 'ifs' in modes else None
            P = {m: rollout(o_prev, o_curr, it, ifs, m) for m in modes}
            for L in range(LEADS):
                tr = truth(D0 + timedelta(it + L + 1))
                if tr is None:
                    raise ValueError(f'{track} {F} lead {L + 1}: missing truth')
                tr = torch.tensor(tr, device='cuda')[:, sup]
                E = {}
                for m in modes:
                    e = P[m][L] - tr
                    if not torch.isfinite(e).all():
                        raise ValueError(f'{m} {track} {F} lead {L + 1}: {int((~torch.isfinite(e)).sum())} non-finite values inside the support')
                    E[m] = e; mse[m][si, L] = (w*e**2).sum(1).cpu().numpy()
                if a.atm == 'both':
                    dl = E['ifs'] - E['era5']
                    d2[si, L] = (w*dl**2).sum(1).cpu().numpy(); cross[si, L] = (w*E['era5']*dl).sum(1).cpu().numpy()
        for m in modes:
            per[f'{track}_{m}'] = mse[m]
        res[track] = summary(mse[modes[0]]) if len(modes) == 1 else {m: summary(mse[m]) for m in modes}
        if a.atm == 'both':
            per[f'{track}_delta2'], per[f'{track}_cross'] = d2, cross
            res[track]['decomposition'] = {v: {'mse_ifs_minus_era5': [round(float((mse['ifs'] - mse['era5'])[:, L, k].mean()), 6) for L in range(LEADS)],
                                               'E_delta2': [round(float(d2[:, L, k].mean()), 6) for L in range(LEADS)],
                                               'two_E_e0_delta': [round(float(2*cross[:, L, k].mean()), 6) for L in range(LEADS)]} for k, v in enumerate(VARS)}
        print(track, json.dumps({m: {v: summary(mse[m])[v][i] for v in ('thetao', 'uo', 'so', 'zos') for i in (0,)} for m in modes}), flush=True)
    np.savez(OUT/f'ob12_{tag}_per_origin.npz', starts=np.array([str(F) for F in starts]), vars=np.array(VARS), **per)
    (OUT/f'ob12_{tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
