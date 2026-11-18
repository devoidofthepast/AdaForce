"""Step 2 of docs/method_dou_zero_arm_scs_registration_20261011.md: SCS 1/12-deg caches for the Dou zero-arm (D:, float32).
Days 2000-01-01 .. 2024-12-31 (index 0 = 2000-01-01), grid = GLORYS12 native 301 x 301 (0-25N, 100-125E).
  ocean.f32  [N, 5, 301, 301]  uo, vo, thetao, so, zos (surface; land NaN)
  flux0.f32  [N, 8, 301, 301]  F(d)       = COARE(ERA5 day d,   ocean day d)
  flux1.f32  [N, 8, 301, 301]  F_lead1(d) = COARE(ERA5 day d+1, ocean day d)   (last day NaN until ERA5 2025 is converted)
Flux order and code = the authors' compute_bulk_flux_np (tau_u, tau_v, sensible, latent, net_short, net_long, evap, rain),
P = 1015 hPa. Atmosphere: ERA5 formal_daily_v2 (0.25-deg cell centres) bilinear to the 1/12 grid (edge rows extrapolated),
converted to the authors' interface; humidity = 2 m specific humidity from d2m (registration change 8, option A).
Resumable: progress.json records the last finished day of each array. Usage: python dou_scs_build_cache.py"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import json
import time
from datetime import date, timedelta
from pathlib import Path

import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dou_bulk_adapter import author_module, specific_humidity_from_dewpoint

ROOT = Path(DATA_D + '/dou_scs')
V2 = DATA_E + '/era5_scs_full_hourly/formal_daily_v2/'
STATIC = DATA_E + '/glorys12_static_scs/glorys12_static_bathy_scs.nc'
D0, D1 = date(2000, 1, 1), date(2024, 12, 31)
N = (D1 - D0).days + 1
H = W = 301
OCEAN_SRC = [  # (first day, last day, {var: (file, var)})
    (date(2000, 1, 1), date(2016, 12, 31), {'uo': (DATA_E + '/glorys_scs_2000_2016/glorys_uo_vo_2000_2016.nc', 'uo'),
                                            'vo': (DATA_E + '/glorys_scs_2000_2016/glorys_uo_vo_2000_2016.nc', 'vo'),
                                            'thetao': (DATA_E + '/glorys_scs_2000_2016/glorys_thetao_so_2000_2016.nc', 'thetao'),
                                            'so': (DATA_E + '/glorys_scs_2000_2016/glorys_thetao_so_2000_2016.nc', 'so'),
                                            'zos': (DATA_E + '/glorys_scs_2000_2016/glorys_zos_2000_2016.nc', 'zos')}),
    (date(2017, 1, 1), date(2024, 12, 31), {'uo': (DATA_E + '/glorys_scs_subsurface/glorys_uo_0.4m_2017_2024.nc', 'uo'),
                                            'vo': (DATA_E + '/glorys_scs_subsurface/glorys_vo_0.4m_2017_2024.nc', 'vo'),
                                            'thetao': (DATA_E + '/south_china_sea/surface_thermohaline/glorys_thetao_so_2017_2024.nc', 'thetao'),
                                            'so': (DATA_E + '/south_china_sea/surface_thermohaline/glorys_thetao_so_2017_2024.nc', 'so'),
                                            'zos': (DATA_E + '/glorys_scs_subsurface/glorys_zos_surface_2017_2024.nc', 'zos')})]
VARS = ('uo', 'vo', 'thetao', 'so', 'zos')


def arr(name, ch):
    path = ROOT/f'{name}.f32'
    mode = 'r+' if path.exists() else 'w+'
    return np.memmap(path, np.float32, mode, shape=(N, ch, H, W))


def progress(update=None):
    p = ROOT/'progress.json'
    st = json.loads(p.read_text()) if p.exists() else {'ocean': -1, 'flux0': -1, 'flux1': -1}
    if update:
        st.update(update); p.write_text(json.dumps(st))
    return st


def build_ocean(oc):
    done = progress()['ocean']
    for a, b, src in OCEAN_SRC:
        i0, i1 = (a - D0).days, (b - D0).days
        if done >= i1:
            continue
        files = {v: netCDF4.Dataset(f) for v, (f, _) in src.items()}
        try:
            for v, (f, n) in src.items():
                t = files[v]['time']
                first = netCDF4.num2date(t[0], t.units)
                assert (first.year, first.month, first.day) == (a.year, a.month, a.day) and len(t) == i1 - i0 + 1, (f, first, len(t))
            for s in range(max(i0, done + 1), i1 + 1, 60):
                e = min(s + 60, i1 + 1)
                for k, v in enumerate(VARS):
                    x = files[v][src[v][1]][s - i0:e - i0]
                    x = np.ma.filled(x, np.nan).astype(np.float32)
                    oc[s:e, k] = x[:, 0] if x.ndim == 4 else x
                oc.flush(); progress({'ocean': e - 1})
                print('ocean', (D0 + timedelta(e - 1)).isoformat(), flush=True)
        finally:
            for f in files.values():
                f.close()


class Era5:
    def __init__(self, lat, lon):
        self.pts = np.stack(np.meshgrid(lat, lon, indexing='ij'), -1)
        self.cache = {}

    def day(self, d):
        key = (d.year, d.month)
        if key not in self.cache:
            self.cache = {}
            fn = f'{V2}forcing_{d.year}{d.month:02d}.nc'
            if not Path(fn).exists():
                return None
            with netCDF4.Dataset(fn) as f:
                self.cache[key] = (f['latitude'][:].astype(np.float64), f['longitude'][:].astype(np.float64),
                                   {n: np.asarray(f[n][:], np.float64) for n in ('u10', 'v10', 't2m', 'd2m', 'ssrd', 'strd', 'tp')})
        la, lo, data = self.cache[key]
        it = lambda v: RegularGridInterpolator((la, lo), v[d.day - 1], bounds_error=False, fill_value=None)(self.pts)
        x = {n: it(v) for n, v in data.items()}
        return {'u10': x['u10'], 'v10': x['v10'], 't2m': x['t2m'], 'sh': specific_humidity_from_dewpoint(x['d2m'], 1015.),
                'ssrd': x['ssrd']*3600., 'strd': x['strd']*3600., 'tp': x['tp']*3.6}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(STATIC) as s:
        lat, lon = np.asarray(s['latitude'][:]), np.asarray(s['longitude'][:])
        mask = np.squeeze(np.ma.filled(s['mask'][:], 0)).astype(bool)
    np.save(ROOT/'mask.npy', mask); np.save(ROOT/'lat.npy', lat); np.save(ROOT/'lon.npy', lon)
    (ROOT/'meta.json').write_text(json.dumps({'first_day': D0.isoformat(), 'last_day': D1.isoformat(), 'N': N, 'shape': [H, W],
                                              'ocean_vars': VARS, 'flux_vars': ['tau_u', 'tau_v', 'sensible', 'latent', 'net_short', 'net_long', 'evap', 'rain'],
                                              'humidity': '2 m from d2m (Buck, 1015 hPa)', 'pressure_hpa': 1015.}, indent=1))
    oc = arr('ocean', 5)
    build_ocean(oc)
    author = author_module()
    era = Era5(lat, lon)
    latg = np.repeat(lat[:, None], W, 1).astype(np.float32)
    f0, f1 = arr('flux0', 8), arr('flux1', 8)
    st = progress()
    start = min(st['flux0'], st['flux1']) + 1
    t0 = time.time()
    atm_next = era.day(D0 + timedelta(start))
    for i in range(start, N):
        d = D0 + timedelta(i)
        atm, atm_next = atm_next, era.day(d + timedelta(1))
        ocean = np.asarray(oc[i])
        f0[i] = author.compute_bulk_flux_np(ocean, {k: v.astype(np.float32) for k, v in atm.items()}, latg, mask)
        f1[i] = (author.compute_bulk_flux_np(ocean, {k: v.astype(np.float32) for k, v in atm_next.items()}, latg, mask)
                 if atm_next is not None else np.nan)
        if i % 30 == 29 or i == N - 1:
            f0.flush(); f1.flush(); progress({'flux0': i, 'flux1': i})
            print('flux', d.isoformat(), f'{(time.time() - t0)/(i - start + 1):.2f} s/day', flush=True)
    print('ALL_DONE', flush=True)


if __name__ == '__main__':
    main()
