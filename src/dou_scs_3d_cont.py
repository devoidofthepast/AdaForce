"""Three-dimensional model continued training with a synchronised control (docs/state3d_cont_registration_20261019.md,
after the review of 2026-10-19). Arms start from their own previous final weights (state3d_b3d / state3d_c0p).
Stage S (single step, R=1): segments of SEG origins, constant lr (backbone 2e-5; B's new modules 3e-4); B's subsurface
loss weight 1.0 and non-finite subsurface truth masked per sample (review P2). After each segment both a monitor on 24 origins
of 2022 (5th and 20th of each month; disjoint from dev 2021 and OceanBench 2024), 9-day ERA5 rollouts, GLORYS truth:
surface u/v/SSH RMSE d5-9 and (B) subsurface error/persistence at leads 1, 5, 9.
Stop rule for B (decides the number of segments, max 8): stop after segment k >= 2 if the lead-1 subsurface ratio improved
by < 0.01 in each of the last two segments, or if the surface score (d5-9 u/v/SSH RMSE / value before stage S, mean) was
> 1.05 in each of the last two segments (single-step training undoes the relay training; the relay stage recovers it).
Weights entering stage R: B = the segment with the lowest lead-5 subsurface ratio (stage S exists to train the subsurface
path); C0' runs exactly the same number of segments with the same origins (--n-seg) and enters stage R from the same
segment index (--entry-seg). Smoke test 2026-10-19: 2 single-step updates already raised the surface score to 1.09-1.18.
Stage R: from the best weights, relay (2,1,3000), (4,1,2000), (8,1,1500), backbone lr 2e-5 -> 0 cosine (B's modules 3e-4 -> 0),
same origins for both arms. Then the dev-2021 evaluation identical to dou_scs_3d_ft.py (101 ERA5 origins, 14 d).
Usage: python dou_scs_3d_cont.py --arm b3d --out <DATA_D>/dou_scs/runs/state3d_cont_b3d
       python dou_scs_3d_cont.py --arm c0p --n-seg K --out <DATA_D>/dou_scs/runs/state3d_cont_c0p"""
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

import dou_scs_3d_ft as T
from dou_scs_3d_ft import NS, Extra, Roller3D, forward3d
from dou_scs_rollout_ft import ACC, step_loss
from dou_scs_train import D0, ROOT, DouScs, idx, norm_stats

SEG, MAX_SEG = 3800, 8
RELAY = ((2, 1, 3000), (4, 1, 2000), (8, 1, 1500))
W_SUB = 1.0
MON = [idx(date(2022, m, d)) for m in range(1, 13) for d in (5, 20)]


class Roller3DM(Roller3D):
    """as Roller3D, with the subsurface target's non-finite cells masked out of the loss (review P2)."""
    def sub_masked(self, day_index):
        t3 = (D0 + timedelta(day_index) - T.D3).days; raw = self.sub.raw(t3)
        fin = torch.tensor(np.isfinite(raw), device='cuda') & self.sub.twater
        z = (raw - self.sub.mu[:, None, None])/self.sub.sd[:, None, None]
        return torch.nan_to_num(torch.tensor(z, device='cuda'))*self.sub.twater, fin

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
            loss = step_loss(self.lf, self.truth(tgt), o_curr, out, self.dm, self.dsd)
            if self.arm == 'b3d':
                s_tr, fin = self.sub_masked(tgt); sp = sout.permute(2, 1, 0)
                tgt_n = ((s_tr - s_curr) - self.sub.tdm)/self.sub.tdsd; wm = fin.float()
                lsub = torch.sqrt(((sp - tgt_n)**2*wm).sum((1, 2))/wm.sum((1, 2)).clamp(min=1.)).mean()
                loss = loss + W_SUB*lsub
                s_next = (s_curr + sp*self.sub.tdsd + self.sub.tdm)*self.sub.twater
                if collect:
                    sstates.append((s_next.detach(), s_tr, fin))
                s_prev, s_curr = s_curr, s_next
            total = total + loss/R; per.append(float(loss))
            o_next = torch.where(self.mask, o_curr + out.permute(2, 1, 0)*self.dsd + self.dm, torch.nan)
            if collect:
                states.append(o_next.detach())
            f_prev, o_prev, o_curr = f_curr, o_curr, o_next
        return total, per, states, sstates


def monitor(ro, model, extra, arm):
    model.eval(); extra is not None and extra.eval(); se = np.zeros(3); sr = {1: [], 5: [], 9: []}
    with torch.no_grad():
        for i in MON:
            _, _, s, ss = ro.run(model, extra, i, 9, train=False, collect=True)
            for L in range(4, 9):
                tr = ro.truth(i + L + 1); ok = torch.isfinite(tr).all(0) & torch.isfinite(s[L]).all(0)
                e = (s[L] - tr)[:, ok]; se += (e[[0, 1, 4]].double()**2).mean(1).cpu().numpy()
            if arm == 'b3d':
                s0 = ro.sub(i)
                for L in (1, 5, 9):
                    pred, tru, fin = ss[L - 1]
                    r = [float(torch.sqrt(((pred[c] - tru[c])[fin[c]]**2).mean())/torch.sqrt(((s0[c] - tru[c])[fin[c]]**2).mean()).clamp(min=1e-12)) for c in range(NS)]
                    sr[L].append(r)
    out = {'surf_rmse_d5_9_u_v_ssh': np.sqrt(se/(len(MON)*5)).tolist()}
    if arm == 'b3d':
        out['sub_ratio'] = {L: float(np.mean(sr[L])) for L in sr}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('c0p', 'b3d')); ap.add_argument('--out', required=True)
    ap.add_argument('--n-seg', type=int, default=0, help="C0': number of single-step segments (taken from B)")
    ap.add_argument('--entry-seg', type=int, default=-1, help="C0': segment whose weights enter stage R (taken from B)")
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(44); rng = np.random.default_rng(1044); torch.set_float32_matmul_precision('medium')
    st = norm_stats(); ro = Roller3DM(st, a.arm)
    ck = torch.load(ROOT/f'runs/state3d_{a.arm}/final.pt', map_location='cuda', weights_only=True)
    model = DouScs().cuda(); model.load_state_dict(ck['model']); extra = None
    if a.arm == 'b3d':
        extra = Extra(model.decoder_lr).cuda(); extra.load_state_dict(ck['extra'])
    seg_n = 32 if a.smoke else SEG; mon_list = MON[:2] if a.smoke else MON
    globals()['MON'] = mon_list
    groups = [{'params': model.parameters(), 'lr': 2e-5}] + ([{'params': extra.parameters(), 'lr': 3e-4}] if extra else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-5)
    first, last = idx(date(2000, 1, 5)), idx(date(2020, 12, 31))
    hist = [{'seg': 0, **monitor(ro, model, extra, a.arm)}]; print(json.dumps(hist[-1]), flush=True)
    ref = np.array(hist[0]['surf_rmse_d5_9_u_v_ssh']); best = hist[0]['sub_ratio'][5] if a.arm == 'b3d' else 0.; best_seg = 0
    torch.save({'model': model.state_dict(), 'extra': extra.state_dict() if extra else None}, out/'seg0.pt')
    max_seg = a.n_seg if a.arm == 'c0p' else MAX_SEG
    k = 0
    while k < max_seg:
        k += 1; model.train(); extra is not None and extra.train(); t0 = time.time()
        origins = rng.integers(first, last, seg_n); opt.zero_grad(set_to_none=True); losses = []
        for n, i in enumerate(origins):
            loss, _, _, _ = ro.run(model, extra, int(i), 1); (loss/ACC).backward(); losses.append(float(loss))
            if not math.isfinite(losses[-1]):
                raise FloatingPointError(f'non-finite loss at origin {int(i)}')
            if (n + 1) % ACC == 0:
                opt.step(); opt.zero_grad(set_to_none=True)
        m = monitor(ro, model, extra, a.arm); score = float(np.mean(np.array(m['surf_rmse_d5_9_u_v_ssh'])/ref))
        hist.append({'seg': k, 'train_loss': float(np.mean(losses)), 'surf_score_vs_start': score, 'sec': round(time.time() - t0), **m})
        print(json.dumps(hist[-1]), flush=True)
        torch.save({'model': model.state_dict(), 'extra': extra.state_dict() if extra else None}, out/f'seg{k}.pt')
        if a.arm == 'b3d' and m['sub_ratio'][5] <= best:
            best, best_seg = m['sub_ratio'][5], k
        (out/'history_stageS.json').write_text(json.dumps(hist, indent=1))
        if a.arm == 'b3d' and k >= 2:
            sr = [h['sub_ratio'][1] for h in hist]
            small = (sr[-3] - sr[-2] < 0.01) and (sr[-2] - sr[-1] < 0.01)
            worse = all(h.get('surf_score_vs_start', 1.0) > 1.05 for h in hist[-2:])
            if small or worse:
                print(json.dumps({'stop_stageS': k, 'reason': 'sub_ratio_plateau' if small else 'surface_worse_twice', 'entry_seg': best_seg}), flush=True)
                break
    (out/'stageS_summary.json').write_text(json.dumps({'segments_run': k, 'entry_seg': best_seg if a.arm == 'b3d' else a.entry_seg, 'best_sub_ratio_lead5': best}, indent=1))
    # stage R from the best stage-S weights; origins drawn after the stage-S draws (identical across arms if same k)
    entry = best_seg if a.arm == 'b3d' else a.entry_seg
    sd = torch.load(out/f'seg{entry}.pt', map_location='cuda', weights_only=True); model.load_state_dict(sd['model'])
    if extra is not None:
        extra.load_state_dict(sd['extra'])
    groups = [{'params': model.parameters(), 'lr': 2e-5}] + ([{'params': extra.parameters(), 'lr': 3e-4}] if extra else [])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-5)
    relay = ((2, 1, 64), (4, 1, 32), (8, 1, 32)) if a.smoke else RELAY
    total_updates = sum(e*n//ACC for _, e, n in relay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: .5*(1 + math.cos(math.pi*min(u, total_updates)/total_updates)))
    rh = []
    for R, n_ep, n_or in relay:
        model.train(); extra is not None and extra.train(); t0 = time.time(); losses = []
        origins = rng.integers(first, last - R + 1, n_or); opt.zero_grad(set_to_none=True)
        for n, i in enumerate(origins):
            loss, _, _, _ = ro.run(model, extra, int(i), R); (loss/ACC).backward(); losses.append(float(loss))
            if (n + 1) % ACC == 0:
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        rh.append({'R': R, 'train_loss': float(np.mean(losses)), 'sec': round(time.time() - t0)}); print(json.dumps(rh[-1]), flush=True)
    rh.append({'monitor_after_relay': monitor(ro, model, extra, a.arm)}); print(json.dumps(rh[-1]), flush=True)
    torch.save({'model': model.state_dict(), 'extra': extra.state_dict() if extra else None, 'arm': a.arm}, out/'final.pt')
    (out/'history_stageR.json').write_text(json.dumps(rh, indent=1))
    print(json.dumps({'done': True}), flush=True)


if __name__ == '__main__':
    main()
