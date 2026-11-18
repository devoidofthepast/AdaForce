"""(2026-10-24) Faster COARE 3.5 for the online fluxes of rollout training, evaluation and inference.
Same arithmetic as coare_torch.bulk_flux (identical function calls, same dtype), but
  (1) graph-capturable: no host-to-device tensor creation inside (NaN fill via full_like),
  (2) CUDA Graph replay: the ~600 element-wise kernels of one call are captured once per (shape, dtype) and replayed with a
      single launch, which removes the launch overhead that dominates on a 301 x 301 grid.
FastBulkFlux(lat, mask, dtype=torch.float64)(ocean, atm) returns exactly what coare_torch.bulk_flux returns (checked by
coare_bench.py); dtype=torch.float32 is offered for speed only after its error has been measured."""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import torch

import coare_torch as CT


def bulk_flux_g(ocean, atm, lat, mask, dtype=torch.float64, pressure_hpa=1015.0):
    o = ocean.to(dtype)
    a = {k: v.to(dtype) for k, v in atm.items()}
    du, dv = a['u10'] - o[..., 0, :, :], a['v10'] - o[..., 1, :, :]
    sst = o[..., 2, :, :]
    spd = torch.sqrt(du**2 + dv**2)
    t2m_c = a['t2m'] - 273.15
    P = torch.full_like(spd, pressure_hpa)
    rh = CT.rh_calc(t2m_c, P, a['sh'])
    valid = mask & torch.isfinite(spd) & torch.isfinite(t2m_c) & torch.isfinite(rh) & torch.isfinite(sst) \
        & torch.isfinite(a['ssrd']) & torch.isfinite(a['strd']) & torch.isfinite(a['tp'])
    fill = lambda x, v: torch.where(valid, x, torch.full_like(x, v))
    latb = torch.broadcast_to(lat.to(dtype), spd.shape)
    tau, hsb, hlb, rsn, rln, evap, rain = CT.coare35vn(fill(spd, 5.), fill(t2m_c, 25.), fill(rh, 80.), fill(sst, 26.), P,
                                                       fill(a['ssrd']/3600., 200.), fill(a['strd']/3600., 400.),
                                                       fill(a['tp']*1000., 0.), latb)
    tau_u, tau_v = tau*du/spd, tau*dv/spd
    out = torch.stack([tau_u, tau_v, hsb, hlb, rsn, rln, evap, rain], -3)
    nan = torch.full_like(out, float('nan'))
    out = torch.where(valid.unsqueeze(-3), out, nan)
    return torch.where(torch.isfinite(out), out, nan).float()


class FastBulkFlux:
    KEYS = ('u10', 'v10', 't2m', 'sh', 'ssrd', 'strd', 'tp')

    def __init__(self, lat, mask, dtype=torch.float64, graph=True, ocean_only=True):
        self.full_lat, self.full_mask, self.dtype, self.graph, self.ocean_only = lat, mask, dtype, graph, ocean_only; self.g = {}
        if ocean_only:                                   # element-wise arithmetic on the ocean points only (identical values)
            self.idx = torch.nonzero(mask.flatten()).squeeze(1); n = self.idx.numel()
            self.lat = lat.flatten()[self.idx].view(n, 1); self.mask = torch.ones(n, 1, dtype=torch.bool, device=mask.device)
        else:
            self.lat, self.mask = lat, mask

    def _gather(self, ocean, atm):
        n = self.idx.numel()
        o = ocean.flatten(-2)[..., self.idx].view(*ocean.shape[:-2], n, 1)
        return o, {k: atm[k].flatten(-2)[..., self.idx].view(*atm[k].shape[:-2], n, 1) for k in self.KEYS}

    def _scatter(self, out, like):
        full = torch.full((*out.shape[:-2], like.shape[-2]*like.shape[-1]), float('nan'), device=out.device, dtype=out.dtype)
        full[..., self.idx] = out.squeeze(-1)
        return full.view(*out.shape[:-2], like.shape[-2], like.shape[-1])

    def _capture(self, shape_o, dt_o, shape_a, dt_a):
        so = torch.zeros(shape_o, device='cuda', dtype=dt_o); sa = {k: torch.zeros(shape_a, device='cuda', dtype=dt_a) for k in self.KEYS}   # same dtypes as the inputs: no truncation on copy
        sa['t2m'] += 290.; sa['sh'] += 0.01; sa['u10'] += 5.
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):                                            # warm-up on a side stream before capture
                bulk_flux_g(so, sa, self.lat, self.mask, self.dtype)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = bulk_flux_g(so, sa, self.lat, self.mask, self.dtype)
        return g, so, sa, out

    def __call__(self, ocean, atm):
        if self.ocean_only:
            o, a = self._gather(ocean, atm)
            return self._scatter(self._call(o, a), ocean)
        return self._call(ocean, atm)

    def _call(self, ocean, atm):
        if not self.graph:
            return bulk_flux_g(ocean, atm, self.lat, self.mask, self.dtype)
        key = (tuple(ocean.shape), ocean.dtype, tuple(atm['u10'].shape), atm['u10'].dtype)
        if key not in self.g:
            self.g[key] = self._capture(*key)
        g, so, sa, out = self.g[key]
        so.copy_(ocean)
        for k in self.KEYS:
            assert atm[k].dtype == sa[k].dtype, (k, atm[k].dtype)
            sa[k].copy_(atm[k])
        g.replay()
        return out.clone()
