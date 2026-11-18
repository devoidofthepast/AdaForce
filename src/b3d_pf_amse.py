"""B-sched + pushforward + AMSE (docs/b3d_pf_amse_registration_20261020.md). Start: runs/state3d_sched_b3d/final.pt.
Identical sampling to ro_pushforward.py / ro_pf_amse.py (3008 samples, p_mid 0.7, k in {4,5,6}, 3 gradient steps; short
samples R=2 from the truth; origin RNG 4044, sample-type RNG 12). Surface step loss = AMSE (amse_loss.py), as in the 2-D run;
subsurface loss unchanged from B (masked per-channel RMSE of the normalised subsurface tendency, weight 0.5, non-finite
truth masked). During the no-graph pushforward the subsurface is updated from the model's own predictions (no truth backfill).
lr: backbone 1.5e-5 (as the 2-D run), new modules 2.25e-4 (B-sched ratio 15x), constant then cosine over the last 25%.
Usage: python b3d_pf_amse.py --out DIR [--smoke]"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from amse_loss import amse_step_loss
from dou_scs_3d_cont import Roller3DM
from dou_scs_3d_ft import Extra, forward3d
from dou_scs_rollout_ft import ACC
from dou_scs_train import ROOT, DouScs, idx, norm_stats

import os as _os
N_SAMPLES, P_MID, KS, M, W_SUB = 3008, 0.7, (4, 5, 6), 3, float(_os.environ.get('SUB3D_WSUB', 0.5))   # env: subsurface_boost_registration_20261022


def ctx_of(ro, f_prev, o_prev, f_curr, o_curr, f_futu, tgt):
    return torch.cat([f_prev, torch.nan_to_num((o_prev - ro.om)/ro.osd), f_curr, torch.nan_to_num((o_curr - ro.om)/ro.osd),
                      f_futu, ro.cst, ro.tch(tgt)]).unsqueeze(0).float()


def sample_loss(ro, model, extra, i, k, R):
    sub = ro.sub; lt = torch.tensor([[1.]], device='cuda')
    o_prev, o_curr = ro.truth(i - 1), ro.truth(i); s_prev, s_curr = sub(i - 1), sub(i); f_prev = ro.flux(o_prev, i - 1)
    with torch.no_grad():
        for j in range(1, k + 1):
            f_curr, f_futu = ro.flux(o_curr, i + j - 1), ro.flux(o_curr, i + j)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a, b = forward3d(model, extra, ctx_of(ro, f_prev, o_prev, f_curr, o_curr, f_futu, i + j), torch.cat([s_prev, s_curr]).unsqueeze(0).float(), lt)
            out, sout = a[0].float(), b[0].float()
            s_next = (s_curr + sout.permute(2, 1, 0)*sub.tdsd + sub.tdm)*sub.twater
            o_next = torch.where(ro.mask, o_curr + out.permute(2, 1, 0)*ro.dsd + ro.dm, torch.nan)
            f_prev, o_prev, o_curr, s_prev, s_curr = f_curr, o_curr, o_next, s_curr, s_next
    total = 0.
    for j in range(k + 1, k + R + 1):
        f_curr, f_futu = ro.flux(o_curr, i + j - 1), ro.flux(o_curr, i + j)
        ctx = ctx_of(ro, f_prev, o_prev, f_curr, o_curr, f_futu, i + j); sctx = torch.cat([s_prev, s_curr]).unsqueeze(0).float()

        def step(c, s):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a, b = forward3d(model, extra, c, s, lt)
            return a[0].float(), b[0].float()
        out, sout = checkpoint(step, ctx, sctx, use_reentrant=False)
        o_next = torch.where(ro.mask, o_curr + out.permute(2, 1, 0)*ro.dsd + ro.dm, torch.nan)
        l = amse_step_loss(o_next, ro.truth(i + j), ro.om, ro.dsd, ro.mask)
        s_tr, fin = ro.sub_masked(i + j); sp = sout.permute(2, 1, 0); wm = fin.float()
        tgt_n = ((s_tr - s_curr) - sub.tdm)/sub.tdsd
        lsub = torch.sqrt(((sp - tgt_n)**2*wm).sum((1, 2))/wm.sum((1, 2)).clamp(min=1.)).mean()
        total = total + (l + W_SUB*lsub)/R
        s_next = (s_curr + sp*sub.tdsd + sub.tdm)*sub.twater
        f_prev, o_prev, o_curr, s_prev, s_curr = f_curr, o_curr, o_next, s_curr, s_next
    return total


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--smoke', action='store_true'); ap.add_argument('--init', default=None, help='start checkpoint (default runs/state3d_sched_b3d/final.pt)'); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(44); torch.set_float32_matmul_precision('medium')
    rng = np.random.default_rng(4044); trng = np.random.default_rng(12)
    st = norm_stats(); ro = Roller3DM(st, 'b3d')
    ck = torch.load(a.init or ROOT/'runs/state3d_sched_b3d/final.pt', map_location='cuda', weights_only=True)
    model = DouScs().cuda(); model.load_state_dict(ck['model']); extra = Extra(model.decoder_lr).cuda(); extra.load_state_dict(ck['extra'])
    model.train(); extra.train()
    n = 48 if a.smoke else N_SAMPLES
    opt = torch.optim.AdamW([{'params': model.parameters(), 'lr': 1.5e-5}, {'params': extra.parameters(), 'lr': 2.25e-4}], betas=(0.9, 0.95), weight_decay=1e-5)
    nsub = int(_os.environ.get('B3D_NSAMPLES', 0)) if not a.smoke else 0   # (2026-10-24, m1b pilot) first nsub samples of the 3008-sample sequence
    total = (nsub or n)//ACC; d0 = int(round(0.75*total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: 1.0 if u < d0 else .5*(1 + math.cos(math.pi*min(u - d0, total - d0)/max(total - d0, 1))))
    first, last = idx(date(2000, 1, 5)), idx(date(2020, 12, 31)) - 10
    origins = rng.integers(first, last, n); kinds = trng.random(n) < P_MID; ks = trng.choice(KS, n)
    if nsub:
        n = nsub; origins, kinds, ks = origins[:n], kinds[:n], ks[:n]
    t0 = time.time(); losses = {'mid': [], 'short': []}; opt.zero_grad(set_to_none=True); hist = []
    for s, i in enumerate(origins):
        loss = sample_loss(ro, model, extra, int(i), int(ks[s]), M) if kinds[s] else sample_loss(ro, model, extra, int(i), 0, 2)
        (losses['mid'] if kinds[s] else losses['short']).append(float(loss)); (loss/ACC).backward()
        if not math.isfinite(float(loss)):
            raise FloatingPointError(f'non-finite loss at origin {int(i)}')
        if (s + 1) % ACC == 0:
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        if (s + 1) % 752 == 0 or s + 1 == n:
            hist.append({'samples': s + 1, 'mid_loss': float(np.mean(losses['mid'][-500:])), 'short_loss': float(np.mean(losses['short'][-200:])),
                         'sec': round(time.time() - t0)}); print(json.dumps(hist[-1]), flush=True)
    assert n % ACC == 0
    torch.save({'model': model.state_dict(), 'extra': extra.state_dict(), 'arm': 'b3d'}, out/'final.pt'); (out/'history.json').write_text(json.dumps(hist, indent=1))
    print(json.dumps({'done': True}), flush=True)


if __name__ == '__main__':
    main()
