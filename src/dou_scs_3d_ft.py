"""Three-dimensional state extension (B) and its matched control (C0'), docs/state3d_registration_20261019.md.
--arm c0p: surface model only, curriculum (1,1,4000),(2,1,3000),(4,1,2000),(8,1,1500) from seed44_ro/best.pt.
--arm b3d: same curriculum and origins; + 18 subsurface channels (T, S at 29/66/110/156/222 m; u, v at 29/66/110/156 m) as
  state: prev+curr subsurface (36 ch) enter through a zero-initialised 4x4 patch embedding added to the tokens; a second
  CNN prediction head (shared trunk features) outputs the 18 subsurface tendencies; in the rollout the subsurface is the
  model's own prediction. Loss = surface step_loss + 0.5 * mean masked per-channel RMSE of the normalised subsurface tendency.
Identity check: at initialisation the surface output equals the base model (< 1e-5). Evaluation as dou_scs_adapter_ft.py
(101 dev-2021 ERA5 origins, 14 d) + subsurface RMSE vs persistence.
Usage: python dou_scs_3d_ft.py --arm b3d --out <DATA_D>/dou_scs/runs/state3d_b3d [--smoke-steps 2] [--eval-only-n 3]"""
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

from dou_scs_adapter_ft import band_stats
from dou_scs_rollout_ft import ACC, Roller, step_loss
from dou_scs_train import D0, ROOT, DouScs, idx, norm_stats
from timm.models.vision_transformer import PatchEmbed
from vit_new import CNNPredictionHead

STAGES = ((1, 1, 4000), (2, 1, 3000), (4, 1, 2000), (8, 1, 1500))
G3 = Path(DATA_E + '/glorys3d_1_12'); D3 = date(2000, 1, 1)
SUB = [('thetao', d) for d in ('29.44', '65.81', '109.73', '155.85', '222.48')] + [('so', d) for d in ('29.44', '65.81', '109.73', '155.85', '222.48')] + \
      [('uo', d) for d in ('29.44', '65.81', '109.73', '155.85')] + [('vo', d) for d in ('29.44', '65.81', '109.73', '155.85')]
# (2026-10-22, docs/subsurface_boost_registration_20261022.md) SUB3D_DEEP=1 appends 14 channels (T, S at 318/454/644 m;
# u, v at 222/318/454/644 m) AFTER the original 18, so original channel indices and statistics are unchanged; files of the
# deeper levels live in <DATA_D>/glorys3d_1_12. SUB3D_WSUB overrides the subsurface loss weight. Defaults = original.
import os as _os
if _os.environ.get('SUB3D_DEEP') == '1':
    SUB = SUB + [('thetao', d) for d in ('318.13', '453.94', '643.57')] + [('so', d) for d in ('318.13', '453.94', '643.57')] + \
        [('uo', d) for d in ('222.48', '318.13', '453.94', '643.57')] + [('vo', d) for d in ('222.48', '318.13', '453.94', '643.57')]
G3_ALT = Path(DATA_D + '/glorys3d_1_12')
NS = len(SUB); W_SUB = float(_os.environ.get('SUB3D_WSUB', 0.5))
# (2026-10-23, docs/e2_subsurface_capacity_registration_20261023.md) SUB3D_SUBBLOCKS=K adds K subsurface-only Swin V2 blocks
# (copies of the last K backbone blocks, post-norm affine set to zero -> identity at initialisation) between the shared
# features and the subsurface head; SUB3D_GSCALE scales the gradient that the subsurface branch sends back into the shared
# features (forward unchanged). Defaults (0, 1.0) = original model.
SUBBLOCKS = int(_os.environ.get('SUB3D_SUBBLOCKS', 0)); GSCALE = float(_os.environ.get('SUB3D_GSCALE', 1.0))
# (2026-10-24, docs/m1_subsurface_anomaly_registration_20261024.md) SUB3D_ANOM=1 adds a zero-initialised embedding of the
# subsurface anomaly relative to the 2000-2020 harmonic daily climatology (work/m1_clim.py; <DATA_D>/dou_scs/
# sub_clim_harm.npy, 3 harmonics, standardised units, float32), scaled by the per-channel anomaly std. The phase comes from the
# target-day sin/cos input channels (36, 37) and is stepped back by one and two days, so no date has to be passed around.
ANOM = _os.environ.get('SUB3D_ANOM') == '1'
_CLIM = {}
_DAY = 2*np.pi/365.2425                                                    # one day of phase; nd = 365/366 differs by < 2e-5 rad


def _clim():
    if not _CLIM:
        import json as _json
        _CLIM['c'] = torch.tensor(np.load(DATA_D + '/dou_scs/sub_clim_harm.npy'), device='cuda')   # float32 [NS, 7, H, W]
        meta = _json.loads(Path(DATA_D + '/dou_scs/sub_clim_meta.json').read_text())
        _CLIM['sd'] = torch.tensor(meta['anom_sd'], device='cuda').view(1, -1, 1, 1)
        _CLIM['k'] = torch.arange(1, int(meta['harmonics']) + 1, device='cuda', dtype=torch.float32)
    return _CLIM['c'], _CLIM['sd'], _CLIM['k']


def clim_at(theta):
    """climatology [B, NS, H, W] (standardised units) at phases theta [B] (rad)."""
    c, _, k = _clim()
    kt = theta[:, None]*k[None]                                                                       # [B, K]
    b = torch.cat([torch.ones_like(theta)[:, None], torch.stack([torch.cos(kt), torch.sin(kt)], 2).flatten(1)], 1)   # [B, 2K+1]
    return torch.einsum('bp,cphw->bchw', b, c)


def sub_anomaly(ctx, sctx):
    """[B, 2*NS, H, W] scaled anomalies of (s_prev, s_curr); the target day (phase from ctx[:, 36:38]) follows s_curr."""
    _, sd, _ = _clim()
    with torch.autocast('cuda', enabled=False):                              # keep the climatology arithmetic in float32
        th = torch.atan2(ctx[:, 36, 0, 0].float(), ctx[:, 37, 0, 0].float())
        a_prev = (sctx[:, :NS].float() - clim_at(th - 2*_DAY))/sd; a_curr = (sctx[:, NS:].float() - clim_at(th - _DAY))/sd
    return torch.cat([a_prev, a_curr], 1)


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s):
        ctx.s = s
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g*ctx.s, None


class Sub3D:
    """standardised subsurface state z = (x - mu)/sd [18, 301, 301] (0 where no water) + per-depth water masks."""
    def __init__(self):
        self.nc = {k: netCDF4.Dataset((G3 if (G3/f'{k[0]}_{k[1]}m.nc').exists() else G3_ALT)/f'{k[0]}_{k[1]}m.nc') for k in SUB}
        rng = np.random.default_rng(0); days = np.sort(rng.integers(1, (date(2020, 12, 31) - D3).days, 120))
        x0 = np.stack([self.raw(int(d) - 1) for d in days]); x1 = np.stack([self.raw(int(d)) for d in days])
        self.water = np.isfinite(x1).all(0)                                  # [18, H, W]
        self.mu = np.nanmean(x1, (0, 2, 3)).astype(np.float32); self.sd = np.nanstd(x1, (0, 2, 3)).astype(np.float32)
        dz = (x1 - x0)/self.sd[None, :, None, None]
        self.dm = np.nanmean(dz, (0, 2, 3)).astype(np.float32); self.dsd = np.nanstd(dz, (0, 2, 3)).astype(np.float32)
        t = lambda a: torch.tensor(a, device='cuda').view(-1, 1, 1)
        self.tdm, self.tdsd = t(self.dm), t(self.dsd); self.twater = torch.tensor(self.water, device='cuda')

    def raw(self, t3):
        return np.stack([np.ma.filled(np.ma.asarray(self.nc[k][k[0]][t3], dtype=np.float32), np.nan).reshape(301, 301) for k in SUB])

    def __call__(self, day_index):
        t3 = (D0 + timedelta(day_index) - D3).days
        z = (self.raw(t3) - self.mu[:, None, None])/self.sd[:, None, None]
        return torch.nan_to_num(torch.tensor(z, device='cuda'))*self.twater


class Extra(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.embed = PatchEmbed(vit.padded_img_size, vit.patch_size, 2*NS, vit.head.proj.in_channels)
        nn.init.zeros_(self.embed.proj.weight); nn.init.zeros_(self.embed.proj.bias)
        self.head = CNNPredictionHead(embed_dim=vit.head.proj.in_channels, out_channels=NS, patch_size=vit.patch_size, padded_img_size=vit.padded_img_size)
        if _os.environ.get('SUB3D_FOLDBN') == '1':          # (2026-10-24) exact BN folding in the subsurface head, see head_fold.py
            from head_fold import fold_head
            fold_head(self.head)
        if ANOM:
            self.embed_anom = PatchEmbed(vit.padded_img_size, vit.patch_size, 2*NS, vit.head.proj.in_channels)
            nn.init.zeros_(self.embed_anom.proj.weight); nn.init.zeros_(self.embed_anom.proj.bias)
        if SUBBLOCKS > 0:
            import copy
            self.blocks = nn.ModuleList(copy.deepcopy(vit.blocks[-SUBBLOCKS:]))
            for blk in self.blocks:
                for nm in (blk.norm1, blk.norm2):
                    nn.init.zeros_(nm.weight); nn.init.zeros_(nm.bias)


def forward3d(model, extra, ctx, sctx, lt):
    """surface out [lon, lat, 5] (as DouScs) and subsurface out [lon, lat, 18]."""
    vit = model.decoder_lr; pw, ph = vit.pad_w_total, vit.pad_h_total
    x = nn.functional.pad(ctx, (0, pw, 0, ph), mode='replicate'); xs = nn.functional.pad(sctx, (0, pw, 0, ph), mode='replicate')
    x = vit.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    tok = vit.token_embeds[0](x) + vit.pos_embed
    if extra is not None:
        tok = tok + extra.embed(xs)
        if getattr(extra, 'embed_anom', None) is not None:
            tok = tok + extra.embed_anom(nn.functional.pad(sub_anomaly(ctx, sctx), (0, pw, 0, ph), mode='replicate'))
    tok = tok + vit.lead_time_embed(lt[:, 0].unsqueeze(-1)).unsqueeze(1)
    B, N, C = tok.shape; h = tok.view(B, vit.grid_h, vit.grid_w, C)
    for blk in vit.blocks:
        h = blk(h)
    h = vit.pre_head(vit.norm(h))
    H, Wd = vit.ori_img_size
    surf = vit.head(h)[:, :, :H, :Wd].permute(0, 2, 3, 1).permute(0, 2, 1, 3)*model.mask.permute(0, 3, 2, 1)
    sub = None
    if extra is not None:
        hs = h
        if getattr(extra, 'blocks', None) is not None:
            hs = _GradScale.apply(h, GSCALE)
            for blk in extra.blocks:
                hs = blk(hs)
        sub = extra.head(hs)[:, :, :H, :Wd].permute(0, 2, 3, 1).permute(0, 2, 1, 3)
    return surf, sub


class Roller3D(Roller):
    def __init__(self, st, arm):
        super().__init__(st); self.arm = arm; self.sub = Sub3D() if arm == 'b3d' else None

    def run(self, model, extra, i, R, train=True, collect=False):
        o_prev, o_curr = self.truth(i - 1), self.truth(i)
        if self.arm == 'b3d':
            s_prev, s_curr = self.sub(i - 1), self.sub(i)
        f_prev = self.flux(o_prev, i - 1); lt = torch.tensor([[1.]], device='cuda')
        total, per, states, sstates = 0., [], [], []
        for j in range(1, R + 1):
            cur, tgt = i + j - 1, i + j
            f_curr, f_futu = self.flux(o_curr, cur), self.flux(o_curr, tgt)
            ctx = torch.cat([f_prev, torch.nan_to_num((o_prev - self.om)/self.osd), f_curr, torch.nan_to_num((o_curr - self.om)/self.osd),
                             f_futu, self.cst, self.tch(tgt)]).unsqueeze(0).float()
            sctx = torch.cat([s_prev, s_curr]).unsqueeze(0).float() if self.arm == 'b3d' else torch.zeros((1, 2*NS, 301, 301), device='cuda')

            def step(c, s):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    a, b = forward3d(model, extra, c, s, lt)
                return a[0].float(), (b[0].float() if b is not None else None)
            out, sout = checkpoint(step, ctx, sctx, use_reentrant=False) if train else step(ctx, sctx)
            o_tr = self.truth(tgt)
            loss = step_loss(self.lf, o_tr, o_curr, out, self.dm, self.dsd)
            if self.arm == 'b3d':
                s_tr = self.sub(tgt); sp = sout.permute(2, 1, 0)                          # [18, lat, lon]
                tgt_n = ((s_tr - s_curr) - self.sub.tdm)/self.sub.tdsd
                wm = self.sub.twater.float()
                lsub = torch.sqrt(((sp - tgt_n)**2*wm).sum((1, 2))/wm.sum((1, 2))).mean()
                loss = loss + W_SUB*lsub
                s_next = (s_curr + sp*self.sub.tdsd + self.sub.tdm)*self.sub.twater
                if collect:
                    sstates.append((s_next.detach(), s_tr))
                s_prev, s_curr = s_curr, s_next
            total = total + loss/R; per.append(float(loss))
            o_next = torch.where(self.mask, o_curr + out.permute(2, 1, 0)*self.dsd + self.dm, torch.nan)
            if collect:
                states.append(o_next.detach())
            f_prev, o_prev, o_curr = f_curr, o_curr, o_next
        return total, per, states, sstates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('c0p', 'b3d')); ap.add_argument('--out', required=True)
    ap.add_argument('--init-from', default=str(ROOT/'runs/seed44_ro/best.pt')); ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--smoke-steps', type=int, default=0); ap.add_argument('--eval-only-n', type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); torch.set_float32_matmul_precision('medium')
    st = norm_stats(); ro = Roller3D(st, a.arm)
    model = DouScs().cuda(); model.load_state_dict(torch.load(a.init_from, map_location='cuda', weights_only=True)['model'])
    extra = Extra(model.decoder_lr).cuda() if a.arm == 'b3d' else None
    dev = [idx(date(2021, m, d)) for m in range(1, 13) for d in (5, 20)]
    if extra is not None:                                          # identity check of the surface output at initialisation
        model.eval(); extra.eval()
        with torch.no_grad():
            for i in dev[:3]:
                _, _, s1, _ = ro.run(model, extra, i, 3, train=False, collect=True)
                ro.arm = 'c0p'; _, _, s0, _ = ro.run(model, None, i, 3, train=False, collect=True); ro.arm = 'b3d'
                d = max(float(torch.nan_to_num(x - y).abs().max()) for x, y in zip(s1, s0))
                if d > 1e-5:
                    raise RuntimeError(f'identity check failed at origin {i}: max |diff| {d}')
        print(json.dumps({'identity_check': 'passed'}), flush=True)
    groups = [{'params': model.parameters(), 'lr': 5e-5}] + ([{'params': extra.parameters(), 'lr': 5e-4}] if extra else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-5)
    total_updates = sum(e*n//ACC for _, e, n in STAGES)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: .5*(1 + math.cos(math.pi*min(u, total_updates)/total_updates)))
    first, last = idx(date(2000, 1, 5)), idx(date(2020, 12, 31))
    (out/'config.json').write_text(json.dumps(vars(a) | {'stages': STAGES, 'acc': ACC, 'w_sub': W_SUB}, indent=1))
    hist = []
    for R, n_ep, n_or in (STAGES if not a.eval_only_n else ()):
        for _ in range(n_ep):
            model.train(); extra is not None and extra.train(); t0 = time.time(); losses = []
            origins = rng.integers(first, last - R + 1, n_or); opt.zero_grad(set_to_none=True)
            for k, i in enumerate(origins):
                loss, _, _, _ = ro.run(model, extra, int(i), R)
                (loss/ACC).backward(); losses.append(float(loss))
                if not math.isfinite(losses[-1]):
                    raise FloatingPointError(f'non-finite loss at origin {int(i)}, R {R}')
                if (k + 1) % ACC == 0:
                    opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                    if a.smoke_steps and (k + 1)//ACC >= a.smoke_steps:
                        dt = time.time() - t0
                        print(json.dumps({'smoke': True, 'R': R, 'sec_per_origin': round(dt/(k + 1), 2), 'gpu_peak_gb': round(torch.cuda.max_memory_allocated()/1e9, 2),
                                          'est_train_hours': round(dt/(k + 1)*sum(e*n*max(r, 1) for r, e, n in STAGES)/R/3600, 2)}), flush=True)
                        return
            hist.append({'R': R, 'train_loss': round(float(np.mean(losses)), 4), 'sec': round(time.time() - t0)})
            print(json.dumps(hist[-1]), flush=True)
    if not a.eval_only_n:
        torch.save({'model': model.state_dict(), 'extra': extra.state_dict() if extra else None, 'arm': a.arm}, out/'final.pt')
        (out/'history.json').write_text(json.dumps(hist, indent=1))
    starts = json.loads(Path(DATA_E + '/experiments/a2_ro_beta/starts.json').read_text())['era5']
    starts = [idx(date.fromisoformat(s)) for s in starts]; starts = starts[:a.eval_only_n] if a.eval_only_n else starts
    sup = np.load(ROOT/'mask.npy').copy()
    for ii in range(idx(date(2020, 12, 30)), idx(date(2021, 11, 15)) + 1):
        sup &= np.isfinite(np.asarray(ro.oc[ii])).all(0)
    tsup = torch.tensor(sup, device='cuda')
    lat = np.load(ROOT/'lat.npy'); a0, b0, h = 152, 132, 106
    lo = np.load(ROOT/'lon.npy'); dx = float(np.mean(np.diff(lo)))*111.*np.cos(np.deg2rad(lat[a0:a0 + h].mean())); dy = float(np.mean(np.diff(lat)))*111.
    KX, KY = np.meshgrid(np.fft.fftfreq(h, dx), np.fft.fftfreq(h, dy)); K = np.hypot(KX, KY); wl = np.where(K > 0, 1/np.maximum(K, 1e-12), np.inf)
    win = np.outer(np.hanning(h), np.hanning(h)); bands = ((50, 100), (100, 200))
    mse = np.zeros((len(starts), 14, 5)); spec = {'KE': np.zeros((2, 3)), 'SSH': np.zeros((2, 3))}
    smse = np.zeros((len(starts), 14, NS)); spers = np.zeros((len(starts), 14, NS))
    model.eval(); extra is not None and extra.eval()
    with torch.no_grad():
        for n, i in enumerate(starts):
            _, _, s, ss = ro.run(model, extra, i, 14, train=False, collect=True)
            for L in range(14):
                e = (s[L] - ro.truth(i + L + 1))[:, tsup].double(); mse[n, L] = (e**2).mean(1).cpu().numpy()
                if ss:
                    wm = ro.sub.twater; s0 = ro.sub(i)
                    smse[n, L] = [float(((ss[L][0][c] - ss[L][1][c])[wm[c]]**2).mean()) for c in range(NS)]
                    spers[n, L] = [float(((s0[c] - ss[L][1][c])[wm[c]]**2).mean()) for c in range(NS)]
            fc5 = np.nan_to_num(s[4].cpu().numpy()); tr5 = np.nan_to_num(ro.truth(i + 5).cpu().numpy())
            bs = band_stats(fc5, tr5, (a0, b0, h), wl, bands, win)
            for v in spec:
                spec[v] += np.array(bs[v])
    np.savez(out/'dev_eval.npz', starts=np.array(starts), mse=mse, smse=smse, spers=spers)
    rm = np.sqrt(mse.mean(0))
    res = {'arm': a.arm, 'rmse_d1_3': rm[:3].mean(0).round(5).tolist(), 'rmse_d5_9': rm[4:9].mean(0).round(5).tolist(),
           'rmse_d8_14': rm[7:14].mean(0).round(5).tolist(),
           'lead5_bands': {v: {'T': np.sqrt(spec[v][:, 0]/spec[v][:, 1]).round(3).tolist(), 'r': (spec[v][:, 2]/np.sqrt(spec[v][:, 0]*spec[v][:, 1])).round(3).tolist()} for v in spec}}
    if a.arm == 'b3d':
        res['sub_rmse_over_persistence_lead1_5_9'] = [(np.sqrt(smse[:, L].mean(0))/np.sqrt(spers[:, L].mean(0))).round(3).tolist() for L in (0, 4, 8)]
    (out/'dev_eval.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
