"""Multi-step rollout fine-tune (RO) of the Dou backbone, docs/method_dou_rollout_finetune_registration_20261013.md.
Per sample: origin day i (curr) drawn from the training period, true O(i-1), O(i) as start, R autoregressive steps exactly
as at inference (ob12_eval.py / dou_scs_eval_pilot.py): each step's fluxes are computed live on the GPU (coare_torch) from
the CURRENT model ocean state and ERA5 (0.25 deg cache, SepInterp to 1/12); lt = 1.
Loss: mean over the R steps of WeightedRmseLoss on the normalised tendency w.r.t. the model's own current state,
target = (O_true(i+j) - O_model(i+j-1) - diff_mean)/diff_std, i.e. the state error in tendency units.
Gradients flow through the ocean-state inputs of all steps and through the residual (identity) path of the state update
(step_loss; the target is NOT detached -- internal review F1); the live fluxes are computed from the detached state (they
are forcing, not a learned path). Activation checkpointing per step; micro-batch 1 origin, 16 origins per update.
Curriculum (R, epochs, origins/epoch) = (2, 1, 3000), (4, 2, 2000), (8, 2, 1500); AdamW lr 5e-5 cosine to 0 over all
updates. Validation: 24 dev-2021 origins (5th and 20th of each month), 8-day ERA5 rollout, same loss; best saved.
Stop rule: after an epoch, stop if the wall time exceeds 6 h.
Usage: python dou_scs_rollout_ft.py --init-from <DATA_D>/dou_scs/runs/seed42/best.pt --out <DATA_D>/dou_scs/runs/seed42_ro
       [--smoke-steps 3] [--frac 0.1 --lr 1e-4]"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import argparse
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import coare_torch as CT
import dou_scs_feet as FE
from dou_scs_train import D0, N, ROOT, DouScs, ScsDou, WeightedRmseLoss, idx, norm_stats

STAGES = ((2, 1, 3000), (4, 2, 2000), (8, 2, 1500))
ACC = 16


def step_loss(lf, o_true, o_curr, out, dm, dsd):
    """Loss of one rollout step = normalised STATE error of o_curr + dsd*out + dm against o_true, written as a tendency
    error. The target keeps its graph to o_curr (internal review 2026-10-13 F1: detaching it dropped the identity path
    1/dsd of the residual update; same loss value, wrong gradient). out: [lon, lat, 5]; states: [5, lat, lon]."""
    target = torch.nan_to_num(((o_true - o_curr) - dm)/dsd).permute(2, 1, 0)
    return lf(target.unsqueeze(0), out.unsqueeze(0))


class Roller:
    def __init__(self, st):
        self.oc = np.memmap(ROOT/'ocean.f32', np.float32, 'r', shape=(N, 5, 301, 301))
        self.era = FE.era5_025(); self.interp = FE.SepInterp()
        self.mask = torch.tensor(np.load(ROOT/'mask.npy'), device='cuda')
        self.latg = torch.tensor(np.repeat(np.load(ROOT/'lat.npy')[:, None], 301, 1), device='cuda')
        g = lambda k: torch.tensor(st[k], device='cuda', dtype=torch.float32).view(-1, 1, 1)
        self.om, self.osd, self.fm, self.fsd, self.dm, self.dsd = (g(k) for k in ('ocean_mean', 'ocean_std', 'flux_mean', 'flux_std', 'diff_mean', 'diff_std'))
        self.cst = torch.tensor(ScsDou(D0 + timedelta(1), D0 + timedelta(1), st).const, device='cuda')
        self.lf = WeightedRmseLoss(start_ind=0, end_ind=5, weight_per_variable=False)
        # (2026-10-24, docs/speedup_single_step_20261023.md) exact CUDA-Graph COARE on ocean points (coare_fast.py; identical
        # output, ~2.1x faster); DOU_FAST_COARE=0 falls back to coare_torch.bulk_flux.
        import os as _os
        self._fb = None
        if _os.environ.get('DOU_FAST_COARE', '1') != '0':
            from coare_fast import FastBulkFlux
            self._fb = FastBulkFlux(self.latg, self.mask)

    def truth(self, d):
        return torch.where(self.mask, torch.tensor(np.asarray(self.oc[d]), device='cuda'), torch.nan)

    def flux(self, o, d):
        with torch.no_grad():
            a = self.interp(torch.tensor(np.asarray(self.era[d]), device='cuda'))
            f = self._fb(o.detach(), FE.to_iface(a)) if self._fb is not None else CT.bulk_flux(o.detach(), FE.to_iface(a), self.latg, self.mask)
        return torch.nan_to_num((f - self.fm)/self.fsd).float()

    def tch(self, d):
        dt = D0 + timedelta(d); nd = 366 if dt.year % 4 == 0 else 365; doy = dt.timetuple().tm_yday - 1
        return torch.stack([torch.full((301, 301), float(np.sin(doy*2*np.pi/nd)), device='cuda'),
                            torch.full((301, 301), float(np.cos(doy*2*np.pi/nd)), device='cuda')])

    def rollout_loss(self, model, i, R, train=True):
        """origin i = curr day; returns the mean loss over R steps (graph kept if train) and per-step losses."""
        o_prev, o_curr = self.truth(i - 1), self.truth(i)
        f_prev = self.flux(o_prev, i - 1)
        lt = torch.tensor([[1.]], device='cuda')
        total, per = 0., []
        for j in range(1, R + 1):
            cur, tgt = i + j - 1, i + j
            f_curr, f_futu = self.flux(o_curr, cur), self.flux(o_curr, tgt)
            ctx = torch.cat([f_prev, torch.nan_to_num((o_prev - self.om)/self.osd), f_curr, torch.nan_to_num((o_curr - self.om)/self.osd),
                             f_futu, self.cst, self.tch(tgt)]).unsqueeze(0).float()

            def step(c):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return model({'y_context': c, 'lt': lt})[0].float()
            out = checkpoint(step, ctx, use_reentrant=False) if train else step(ctx)       # [lon, lat, 5]
            o_tr = self.truth(tgt)
            loss = step_loss(self.lf, o_tr, o_curr, out, self.dm, self.dsd)
            total = total + loss/R; per.append(float(loss))
            o_next = torch.where(self.mask, o_curr + out.permute(2, 1, 0)*self.dsd + self.dm, torch.nan)
            f_prev, o_prev, o_curr = f_curr, o_curr, o_next
        return total, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-from', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke-steps', type=int, default=0); ap.add_argument('--max-hours', type=float, default=6.)
    ap.add_argument('--frac', type=float, default=1., help='LR pilot: share of origins per epoch (registration section 2a: 0.1)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    torch.set_float32_matmul_precision('medium')
    st = norm_stats(); ro = Roller(st)
    model = DouScs().cuda()
    model.load_state_dict(torch.load(a.init_from, map_location='cuda', weights_only=True)['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-5)
    stages = [(R, e, max(ACC, int(n*a.frac)//ACC*ACC)) for R, e, n in STAGES]
    total_updates = sum(e*n//ACC for _, e, n in stages)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: .5*(1 + math.cos(math.pi*min(u, total_updates)/total_updates)))
    first, last = idx(date(2000, 1, 2)), idx(date(2020, 12, 31))
    dev = [idx(date(2021, m, d)) for m in range(1, 13) for d in (5, 20)]
    (out/'config.json').write_text(json.dumps(vars(a) | {'stages': stages, 'acc': ACC, 'total_updates': total_updates}, indent=1))
    hist, best, t_start, ep = [], 1e9, time.time(), 0
    if not a.smoke_steps:                                          # reference: the initial (single-step) model
        model.eval()
        with torch.no_grad():
            per = np.mean([ro.rollout_loss(model, i, 8, train=False)[1] for i in dev], 0)
        hist.append({'epoch': -1, 'R': 0, 'val_8day': round(float(per.mean()), 4), 'val_per_lead': per.round(4).tolist()})
        print(json.dumps(hist[-1]), flush=True)
    for R, n_ep, n_or in stages:
        for _ in range(n_ep):
            model.train(); t0 = time.time(); losses = []
            origins = rng.integers(first, last - R + 1, n_or)
            opt.zero_grad(set_to_none=True)
            for k, i in enumerate(origins):
                loss, _ = ro.rollout_loss(model, int(i), R)
                (loss/ACC).backward(); losses.append(float(loss))
                if not math.isfinite(losses[-1]):
                    raise FloatingPointError(f'non-finite loss at origin {int(i)}, R {R}')
                if (k + 1) % ACC == 0:
                    opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                    if a.smoke_steps and (k + 1)//ACC >= a.smoke_steps:
                        dt = time.time() - t0
                        rep = {'R': R, 'updates': (k + 1)//ACC, 'sec_per_origin': round(dt/(k + 1), 2),
                               'est_total_hours': round(dt/(k + 1)*sum(e*n*r for r, e, n in STAGES)/R/3600, 2),
                               'gpu_peak_gb': round(torch.cuda.max_memory_allocated()/1e9, 2), 'loss_first_last': [round(losses[0], 4), round(losses[-1], 4)]}
                        (out/'smoke.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep), flush=True)
                        return
            model.eval(); per = []
            with torch.no_grad():
                for i in dev:
                    per.append(ro.rollout_loss(model, i, 8, train=False)[1])
            per = np.mean(per, 0); val = float(per.mean())
            hist.append({'epoch': ep, 'R': R, 'train_loss': round(float(np.mean(losses)), 4), 'val_8day': round(val, 4),
                         'val_per_lead': per.round(4).tolist(), 'lr': opt.param_groups[0]['lr'], 'sec': round(time.time() - t0)})
            print(json.dumps(hist[-1]), flush=True)
            if val < best:
                best = val; torch.save({'model': model.state_dict(), 'epoch': ep, 'R': R, 'val_8day': val}, out/'best.pt')
            (out/'history.json').write_text(json.dumps(hist, indent=1)); ep += 1
            if time.time() - t_start > a.max_hours*3600:
                print(json.dumps({'stop': 'wall time', 'hours': round((time.time() - t_start)/3600, 2)}), flush=True)
                return


if __name__ == '__main__':
    main()
