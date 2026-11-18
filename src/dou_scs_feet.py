"""FEET on the Dou backbone (docs/method_dou_feet_pilot_registration_20261011.md).
- era5_025.f32 [N, 7, 100, 100]: ERA5 formal_daily_v2 SI daily fields u10 v10 t2m d2m ssrd strd tp (0.25-deg cell centres),
  N = the Dou cache days (2000-01-01 .. 2024-12-31).
- SepInterp: separable linear interpolation with edge extrapolation, 0.25-deg centres -> GLORYS12 1/12 grid; identical to
  scipy RegularGridInterpolator(linear, fill_value=None) used for the cache (checked in test_dou_scs_feet.py).
- Pool: the 0.25-deg FEET pool aug_pool_cfs_2018_2020 (CFSv2 member 01 - ERA5, 14 leads x 8 primitives, normalised by
  raw_stats); here de-normalised to SI and the 7 non-pressure channels used.
- perturb_context(): for the perturbed samples, recompute the three flux blocks of y_context on the GPU (coare_torch) from
  (ERA5 + error) and the true ocean, and normalise them with the run's flux statistics."""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import json
from datetime import date, timedelta

import netCDF4
import numpy as np
import torch

import coare_torch as CT
from dou_scs_train import D0, N, ROOT, ScsDou

POOL = DATA_E + '/experiments/multiyear_inputs/aug_pool_cfs_2018_2020/'
V2 = DATA_E + '/era5_scs_full_hourly/formal_daily_v2/'
SI = ('u10', 'v10', 't2m', 'd2m', 'ssrd', 'strd', 'tp')
POOL_IDX = [0, 1, 2, 3, 5, 6, 7]          # PRIMITIVE_CHANNELS minus sp
P_AUG = .5
FLUX_SLOTS = (slice(0, 8), slice(13, 21), slice(26, 34))   # F(prev), F(curr), F(futu) inside the 38 channels


def build_era5_025():
    path = ROOT/'era5_025.f32'
    done = ROOT/'era5_025.done'
    if done.exists():
        return
    a = np.memmap(path, np.float32, 'w+', shape=(N, 7, 100, 100))
    d, i = D0, 0
    while i < N:
        with netCDF4.Dataset(f'{V2}forcing_{d.year}{d.month:02d}.nc') as f:
            n = len(f['time'])
            a[i:i + n] = np.stack([np.asarray(f[v][:], np.float32) for v in SI], 1)
        i += n; d = D0 + timedelta(i)
    a.flush(); done.write_text('ok')


def era5_025():
    return np.memmap(ROOT/'era5_025.f32', np.float32, 'r', shape=(N, 7, 100, 100))


def _w1d(src, dst):
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    i = np.clip(np.searchsorted(src, dst, side='right') - 1, 0, len(src) - 2)
    w = (dst - src[i])/(src[i + 1] - src[i])
    W = np.zeros((len(dst), len(src)))
    W[np.arange(len(dst)), i] = 1 - w; W[np.arange(len(dst)), i + 1] = w
    return W


class SepInterp:
    def __init__(self, device='cuda'):
        with netCDF4.Dataset(f'{V2}forcing_200001.nc') as f:
            la, lo = np.asarray(f['latitude'][:]), np.asarray(f['longitude'][:])
        lat, lon = np.load(ROOT/'lat.npy'), np.load(ROOT/'lon.npy')
        self.Wy = torch.tensor(_w1d(la, lat), device=device)
        self.Wx = torch.tensor(_w1d(lo, lon), device=device)

    def __call__(self, x):                      # [..., 100, 100] -> [..., 301, 301] (float64)
        return self.Wy @ x.double() @ self.Wx.T


def to_iface(x):
    """SI [..., 7, H, W] (u10 v10 t2m d2m ssrd strd tp) -> the authors' COARE interface dict."""
    td = x[..., 3, :, :] - 273.15
    e = 6.1121*torch.exp(17.502*td/(td + 240.97))*(1.0007 + 3.46e-6*1015.)
    return {'u10': x[..., 0, :, :], 'v10': x[..., 1, :, :], 't2m': x[..., 2, :, :], 'sh': 0.622*e/(1015. - 0.378*e),
            'ssrd': x[..., 4, :, :]*3600., 'strd': x[..., 5, :, :]*3600., 'tp': x[..., 6, :, :]*3.6}


class Pool:
    def __init__(self):
        meta = json.loads(open(POOL + 'meta.json').read())
        self.err = np.load(POOL + 'raw_err.npy', mmap_mode='r')                 # [387, 14, 8, 100, 100] normalised
        self.std = np.array(meta['raw_stats'])[POOL_IDX, 1].astype(np.float32)[:, None, None]
        months = np.array([int(s[5:7]) for s in meta['starts']])
        self.by_month = {m: np.where(np.minimum(np.abs(months - m), 12 - np.abs(months - m)) <= 1)[0] for m in range(1, 13)}

    def field(self, s, lead):                   # SI error [7, 100, 100] of pool start s at lead (1..14); 0 for lead <= 0
        if lead <= 0:
            return np.zeros((7, 100, 100), np.float32)
        return np.asarray(self.err[s, lead - 1, POOL_IDX], np.float32)*self.std


class ScsDouFeet(ScsDou):
    """ScsDou plus, per sample, a FEET draw: lead k ~ U{1..14} (always; fed as lt), with prob 0.5 a pool start s."""
    def __init__(self, *a, seed=42, **k):
        super().__init__(*a, **k)
        self.seed, self.epoch, self.pool, self.era = seed, 0, None, None

    def __getitem__(self, j):
        item = super().__getitem__(j)
        if self.pool is None:
            self.pool, self.era = Pool(), era5_025()
        i = int(self.ids[j])
        rng = np.random.default_rng([self.seed, i, self.epoch])
        lead = int(rng.integers(1, 15))
        pert = rng.random() < P_AUG
        month = (D0 + timedelta(i)).month
        s = int(rng.choice(self.pool.by_month[month])) if pert else 0
        days = (i - 1, i, i + 1)                                     # prev, curr, futu
        leads = (lead - 2, lead - 1, lead)
        atm = np.stack([np.asarray(self.era[d]) for d in days]) if pert else np.zeros((3, 7, 100, 100), np.float32)
        err = np.stack([self.pool.field(s, l) for l in leads]) if pert else np.zeros((3, 7, 100, 100), np.float32)
        oc = self.mm[0]
        ocean = np.stack([np.asarray(oc[i - 1]), np.asarray(oc[i])]) if pert else np.zeros((2, 5, 301, 301), np.float32)
        item.update({'lt': torch.tensor([float(lead)]), 'pert': torch.tensor(pert), 'atm025': torch.from_numpy(atm + err),
                     'ocean_raw': torch.from_numpy(ocean)})
        return item


def perturb_context(task, interp, latg, mask, st):
    """In place on the GPU batch: recompute F(prev), F(curr), F(futu) of the perturbed samples."""
    p = task['pert']
    if not bool(p.any()):
        return task
    fm = torch.as_tensor(st['flux_mean'], device=latg.device).view(8, 1, 1)
    fs = torch.as_tensor(st['flux_std'], device=latg.device).view(8, 1, 1)
    atm = interp(task['atm025'][p])                                     # [b, 3, 7, 301, 301]
    oc = task['ocean_raw'][p]                                           # [b, 2, 5, 301, 301]
    ocean_for = (oc[:, 0], oc[:, 1], oc[:, 1])                          # prev block <- O_prev; curr, futu <- O_curr
    ctx = task['y_context']
    for blk, sl in enumerate(FLUX_SLOTS):
        f = CT.bulk_flux(ocean_for[blk], to_iface(atm[:, blk]), latg, mask)
        ctx[p, sl] = torch.nan_to_num((f - fm)/fs).to(ctx.dtype)
    return task
