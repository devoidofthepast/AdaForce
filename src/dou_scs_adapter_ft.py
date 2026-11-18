"""New-information adapter pilot (docs/new_info_adapter_registration_20261019.md; draft confirmed by default on 2026-10-19).
Same checkpoint (seed44_ro/best.pt), equal budget, curriculum (R, epochs, origins) = (2,1,3000), (4,1,2000), (8,1,1500),
backbone lr 5e-5 (cosine), adapter lr 5e-4. Arms:
  c0: continued training, original 38-channel input (no adapter);
  c1: + ocean states of days t-2, t-3 (10 ch); at the origin from the analysis, during the rollout from the model's own
      predictions (deployment-consistent);
  c2: + origin-time subsurface memory: T, S at 65.8/109.7/222.5 m, u, v at 65.8/109.7 m (10 ch, GLORYS12) + days since the
      origin / 14 (1 ch), constant through the rollout.
Adapter: conv3x3(E->64) - GELU - conv1x1(64->38, zero-initialised) added to the 38-channel input; at initialisation the
output is bit-identical to the base model (checked before training; stop otherwise).
After training: evaluation on the 101 dev-2021 ERA5 origins (a2_ro_beta/starts.json), 14 days, fixed 58,601-point support,
per-origin MSE [origin, lead, var] + KE / SSH band correlations r and transfer T at lead 5 (central box, 50-100/100-200 km).
Usage: python dou_scs_adapter_ft.py --arm c1 --out <DATA_D>/dou_scs/runs/adapter_c1 [--smoke-steps 2]"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import argparse
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from dou_scs_rollout_ft import ACC, Roller, step_loss
from dou_scs_train import D0, ROOT, DouScs, idx, norm_stats

STAGES = ((2, 1, 3000), (4, 1, 2000), (8, 1, 1500))
G3 = Path(DATA_E + '/glorys3d_1_12'); D3 = date(2000, 1, 1)
SUB = [('thetao', '65.81'), ('thetao', '109.73'), ('thetao', '222.48'), ('so', '65.81'), ('so', '109.73'), ('so', '222.48'),
       ('uo', '65.81'), ('uo', '109.73'), ('vo', '65.81'), ('vo', '109.73')]
N_EXTRA = {'c0': 0, 'c1': 10, 'c2': 11}


class Adapter(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.a = nn.Conv2d(n_in, 64, 3, padding=1); self.b = nn.Conv2d(64, 38, 1)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return self.b(torch.nn.functional.gelu(self.a(x)))


class Subsurface:
    def __init__(self, mask):
        self.nc = {k: netCDF4.Dataset(G3/f'{k[0]}_{k[1]}m.nc') for k in SUB}; self.mask = mask
        rng = np.random.default_rng(0); days = rng.integers(0, (date(2020, 12, 31) - D3).days, 120)
        x = np.stack([self.raw(int(d)) for d in days]); m = mask.cpu().numpy()
        self.mu = np.array([np.nanmean(x[:, c][:, m]) for c in range(len(SUB))], np.float32)
        self.sd = np.array([np.nanstd(x[:, c][:, m]) for c in range(len(SUB))], np.float32)

    def raw(self, t3):
        return np.stack([np.ma.filled(np.ma.asarray(self.nc[k][k[0]][t3], dtype=np.float32), np.nan).reshape(301, 301) for k in SUB])

    def __call__(self, day_index):                                 # day_index in the ocean.f32 calendar (D0)
        t3 = (D0 + timedelta(day_index) - D3).days
        z = (self.raw(t3) - self.mu[:, None, None])/self.sd[:, None, None]
        return torch.nan_to_num(torch.tensor(z, device='cuda')*self.mask)


class AdapterRoller(Roller):
    def __init__(self, st, arm):
        super().__init__(st); self.arm = arm
        self.sub = Subsurface(self.mask) if arm == 'c2' else None

    def run(self, model, adapter, i, R, train=True, collect=False):
        """origin i; returns (mean loss, per-step losses, states if collect)."""
        hist = [self.truth(i - 3), self.truth(i - 2), self.truth(i - 1), self.truth(i)]   # [..., o_prev, o_curr]
        sub0 = self.sub(i) if self.arm == 'c2' else None
        f_prev = self.flux(hist[-2], i - 1); lt = torch.tensor([[1.]], device='cuda')
        total, per, states = 0., [], []
        for j in range(1, R + 1):
            o_prev, o_curr = hist[-2], hist[-1]
            cur, tgt = i + j - 1, i + j
            f_curr, f_futu = self.flux(o_curr, cur), self.flux(o_curr, tgt)
            ctx = torch.cat([f_prev, torch.nan_to_num((o_prev - self.om)/self.osd), f_curr, torch.nan_to_num((o_curr - self.om)/self.osd),
                             f_futu, self.cst, self.tch(tgt)]).unsqueeze(0).float()
            if self.arm == 'c1':
                extra = torch.cat([torch.nan_to_num((hist[-3] - self.om)/self.osd), torch.nan_to_num((hist[-4] - self.om)/self.osd)]).unsqueeze(0).float()
            elif self.arm == 'c2':
                extra = torch.cat([sub0, torch.full((1, 301, 301), (j - 1)/14., device='cuda')]).unsqueeze(0).float()
            else:
                extra = None

            def step(c, e):
                if e is not None:
                    c = c + adapter(e)*self.mask
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return model({'y_context': c, 'lt': lt})[0].float()
            out = checkpoint(step, ctx, extra, use_reentrant=False) if train else step(ctx, extra)
            o_tr = self.truth(tgt)
            loss = step_loss(self.lf, o_tr, o_curr, out, self.dm, self.dsd)
            total = total + loss/R; per.append(float(loss))
            o_next = torch.where(self.mask, o_curr + out.permute(2, 1, 0)*self.dsd + self.dm, torch.nan)
            if collect:
                states.append(o_next.detach())
            f_prev = f_curr; hist = hist[1:] + [o_next]
        return total, per, states


def band_stats(fc, tr, box, wl, bands, win):
    a, b, h = box; out = {}
    for var, ch in (('KE', (0, 1)), ('SSH', (4,))):
        pp = pt = cr = 0.
        for c in ch:
            P = np.fft.fft2((fc[c, a:a + h, b:b + h] - fc[c, a:a + h, b:b + h].mean())*win)
            T = np.fft.fft2((tr[c, a:a + h, b:b + h] - tr[c, a:a + h, b:b + h].mean())*win)
            pp = pp + abs(P)**2; pt = pt + abs(T)**2; cr = cr + (P*np.conj(T)).real
        out[var] = [[float(pp[m].sum()), float(pt[m].sum()), float(cr[m].sum())] for m in [(wl >= lo) & (wl < hi) for lo, hi in bands]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('c0', 'c1', 'c2')); ap.add_argument('--out', required=True)
    ap.add_argument('--init-from', default=str(ROOT/'runs/seed44_ro/best.pt')); ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--smoke-steps', type=int, default=0); ap.add_argument('--eval-only-n', type=int, default=0, help='test the evaluation on n origins, no training')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); torch.set_float32_matmul_precision('medium')
    st = norm_stats(); ro = AdapterRoller(st, a.arm)
    model = DouScs().cuda(); model.load_state_dict(torch.load(a.init_from, map_location='cuda', weights_only=True)['model'])
    adapter = Adapter(N_EXTRA[a.arm]).cuda() if N_EXTRA[a.arm] else None
    dev = [idx(date(2021, m, d)) for m in range(1, 13) for d in (5, 20)]
    # identity check at initialisation (adapter arms): outputs bit-identical to the base model
    if adapter is not None:
        model.eval(); adapter.eval()
        with torch.no_grad():
            for i in dev[:3]:
                _, _, s1 = ro.run(model, adapter, i, 3, train=False, collect=True)
                ro_arm = ro.arm; ro.arm = 'c0'; _, _, s0 = ro.run(model, None, i, 3, train=False, collect=True); ro.arm = ro_arm
                d = max(float(torch.nan_to_num(x - y).abs().max()) for x, y in zip(s1, s0))
                if d > 1e-5:
                    raise RuntimeError(f'identity check failed at origin {i}: max |diff| {d}')
        print(json.dumps({'identity_check': 'passed'}), flush=True)
    groups = [{'params': model.parameters(), 'lr': 5e-5}] + ([{'params': adapter.parameters(), 'lr': 5e-4}] if adapter else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-5)
    total_updates = sum(e*n//ACC for _, e, n in STAGES)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: .5*(1 + math.cos(math.pi*min(u, total_updates)/total_updates)))
    first, last = idx(date(2000, 1, 5)), idx(date(2020, 12, 31))
    (out/'config.json').write_text(json.dumps(vars(a) | {'stages': STAGES, 'acc': ACC, 'total_updates': total_updates}, indent=1))
    hist = []
    for R, n_ep, n_or in (STAGES if not a.eval_only_n else ()):
        for _ in range(n_ep):
            model.train(); adapter is not None and adapter.train(); t0 = time.time(); losses = []
            origins = rng.integers(first, last - R + 1, n_or); opt.zero_grad(set_to_none=True)
            for k, i in enumerate(origins):
                loss, _, _ = ro.run(model, adapter, int(i), R)
                (loss/ACC).backward(); losses.append(float(loss))
                if not math.isfinite(losses[-1]):
                    raise FloatingPointError(f'non-finite loss at origin {int(i)}, R {R}')
                if (k + 1) % ACC == 0:
                    opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                    if a.smoke_steps and (k + 1)//ACC >= a.smoke_steps:
                        dt = time.time() - t0
                        print(json.dumps({'smoke': True, 'R': R, 'sec_per_origin': round(dt/(k + 1), 2), 'gpu_peak_gb': round(torch.cuda.max_memory_allocated()/1e9, 2),
                                          'est_train_hours': round(dt/(k + 1)*sum(e*n*r for r, e, n in STAGES)/R/3600, 2)}), flush=True)
                        return
            hist.append({'R': R, 'train_loss': round(float(np.mean(losses)), 4), 'sec': round(time.time() - t0)})
            print(json.dumps(hist[-1]), flush=True)
    if not a.eval_only_n:
        torch.save({'model': model.state_dict(), 'adapter': adapter.state_dict() if adapter else None, 'arm': a.arm}, out/'final.pt')
        (out/'history.json').write_text(json.dumps(hist, indent=1))
    # evaluation: 101 dev-2021 ERA5 origins, 14 days
    starts = json.loads(Path(DATA_E + '/experiments/a2_ro_beta/starts.json').read_text())['era5']
    starts = [idx(date.fromisoformat(s)) if isinstance(s, str) else int(s) for s in starts]
    starts = starts[:a.eval_only_n] if a.eval_only_n else starts
    sup = np.load(ROOT/'mask.npy').copy()
    for ii in range(idx(date(2020, 12, 30)), idx(date(2021, 11, 15)) + 1):
        sup &= np.isfinite(np.asarray(ro.oc[ii])).all(0)
    tsup = torch.tensor(sup, device='cuda')
    lat = np.load(ROOT/'lat.npy'); a0, b0, h = 152, 132, 106                    # central SCS all-ocean box (ke_spectra.box(), as spectral_decomp)
    lo = np.load(ROOT/'lon.npy'); dx = float(np.mean(np.diff(lo)))*111.*np.cos(np.deg2rad(lat[a0:a0 + h].mean())); dy = float(np.mean(np.diff(lat)))*111.
    KX, KY = np.meshgrid(np.fft.fftfreq(h, dx), np.fft.fftfreq(h, dy)); K = np.hypot(KX, KY); wl = np.where(K > 0, 1/np.maximum(K, 1e-12), np.inf)
    win = np.outer(np.hanning(h), np.hanning(h)); bands = ((50, 100), (100, 200))
    mse = np.zeros((len(starts), 14, 5)); spec = {'KE': np.zeros((2, 3)), 'SSH': np.zeros((2, 3))}
    model.eval(); adapter is not None and adapter.eval()
    with torch.no_grad():
        for n, i in enumerate(starts):
            _, _, s = ro.run(model, adapter, i, 14, train=False, collect=True)
            for L in range(14):
                e = (s[L] - ro.truth(i + L + 1))[:, tsup].double()
                mse[n, L] = (e**2).mean(1).cpu().numpy()
            fc5 = np.nan_to_num(s[4].cpu().numpy()); tr5 = np.nan_to_num(ro.truth(i + 5).cpu().numpy())   # land = 0
            if True:
                bs = band_stats(fc5, tr5, (a0, b0, h), wl, bands, win)
                for v in spec:
                    spec[v] += np.array(bs[v])
    np.savez(out/'dev_eval.npz', starts=np.array(starts), mse=mse)
    rm = np.sqrt(mse.mean(0))
    res = {'arm': a.arm, 'rmse_d1_3': rm[:3].mean(0).round(5).tolist(), 'rmse_d5_9': rm[4:9].mean(0).round(5).tolist(),
           'rmse_d8_14': rm[7:14].mean(0).round(5).tolist(),
           'lead5_bands': {v: {'T': np.sqrt(spec[v][:, 0]/spec[v][:, 1]).round(3).tolist(), 'r': (spec[v][:, 2]/np.sqrt(spec[v][:, 0]*spec[v][:, 1])).round(3).tolist()} for v in spec}}
    (out/'dev_eval.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
