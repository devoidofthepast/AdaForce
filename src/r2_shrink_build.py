"""R2 (docs/method_dou_feet_pilot_registration_20261011.md): per-lead, per-channel shrinkage coefficients beta[k, c] for the 8
COARE flux channels and the day-of-year flux climatology used as the shrinkage target.
clim: flux0 cache 2000-2020, every 3rd day, day of year +-15 days -> <DATA_D>/dou_scs/flux_clim_doy.f32 [366, 8, 301, 301]
beta: training-period CFSv2 pool (every 3rd start = 129 starts, 2018-10..2020), flux of (ERA5 + error) vs flux of ERA5 at the
error's valid day with the true ocean of that day, anomalies w.r.t. clim, beta = Cov(F', F~')/Var(F~') pooled over ocean
points and starts -> <DATA_D>/dou_scs/r2_beta.json"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import json
from datetime import date, timedelta

import numpy as np
import torch

import coare_torch as CT
import dou_scs_feet as FE
from dou_scs_train import D0, N, ROOT


def main():
    mask = np.load(ROOT/'mask.npy')
    f0 = np.memmap(ROOT/'flux0.f32', np.float32, 'r', shape=(N, 8, 301, 301))
    oc = np.memmap(ROOT/'ocean.f32', np.float32, 'r', shape=(N, 5, 301, 301))
    last_train = (date(2020, 12, 31) - D0).days
    days = np.arange(0, last_train + 1, 3)
    acc = np.zeros((366, 8, 301, 301)); cnt = np.zeros(366)
    for i in days:
        d = (D0 + timedelta(int(i))).timetuple().tm_yday - 1
        acc[d] += np.nan_to_num(np.asarray(f0[i], np.float64)); cnt[d] += 1
    clim = np.memmap(ROOT/'flux_clim_doy.f32', np.float32, 'w+', shape=(366, 8, 301, 301))
    for d in range(366):
        w = [(d + s) % 366 for s in range(-15, 16)]
        clim[d] = (acc[w].sum(0)/cnt[w].sum()).astype(np.float32)
    clim.flush(); del acc
    pool = FE.Pool(); era = FE.era5_025(); interp = FE.SepInterp()
    m = torch.tensor(mask, device='cuda'); latg = torch.tensor(np.repeat(np.load(ROOT/'lat.npy')[:, None], 301, 1), device='cuda')
    starts = json.loads(open(FE.POOL + 'meta.json').read())['starts']
    s_tt = np.zeros((14, 8)); s_ft = np.zeros((14, 8))
    for si in range(0, len(starts), 3):
        s0 = date.fromisoformat(starts[si])
        for L in range(1, 15):
            vd = s0 + timedelta(L); i = (vd - D0).days; dy = vd.timetuple().tm_yday - 1
            a_era = torch.tensor(np.asarray(era[i]), device='cuda')[None]
            a_fc = a_era + torch.tensor(pool.field(si, L), device='cuda')[None]
            o = torch.tensor(np.asarray(oc[i]), device='cuda')[None]
            c = np.asarray(clim[dy], np.float64)[:, mask]
            fe = CT.bulk_flux(o, FE.to_iface(interp(a_era)), latg, m)[0].cpu().numpy()[:, mask] - c
            ff = CT.bulk_flux(o, FE.to_iface(interp(a_fc)), latg, m)[0].cpu().numpy()[:, mask] - c
            ok = np.isfinite(fe) & np.isfinite(ff)
            s_tt[L - 1] += np.where(ok, ff*ff, 0).sum(1); s_ft[L - 1] += np.where(ok, fe*ff, 0).sum(1)
        print('start', starts[si], flush=True)
    beta = np.clip(s_ft/s_tt, 0., 1.)
    names = ['tau_u', 'tau_v', 'sensible', 'latent', 'net_short', 'net_long', 'evap', 'rain']
    json.dump({'beta': beta.round(4).tolist(), 'channels': names, 'leads': list(range(1, 15)), 'pool_starts_used': len(range(0, len(starts), 3))},
              open(ROOT/'r2_beta.json', 'w'), indent=1)
    for c, n in enumerate(names):
        print(n, beta[[0, 2, 6, 9, 13], c].round(2))


if __name__ == '__main__':
    main()
