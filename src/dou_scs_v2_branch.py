"""V2 of docs/method_forcing_quality_registration_20261013.md (sections 2 and 6): frozen backbone B + corrector C + small
gated residual branch, trained in rollout on forecast-error trajectories.
  dO_k = B(ctx_k) + g * R_phi(ctx_k, lead, q)      (B frozen; only R_phi and the gate are trained)
Forcing on forecast days: ERA5 + CFSv2 pool error of the same start (paired, FE.Pool, start drawn from the calendar month
+-1 of the origin), then C = per-channel beta towards flux_clim_doy (--beta, default r2_beta.json) on the 8 COARE channels.
Analysis days (lead <= 0) use ERA5 untouched.
Branch: 3 conv layers (width 64, GELU) on the 38 context channels + lead/14 + 3 q maps; last layer zero-initialised.
Gate (effective initial value 0.5, section 6.1): --variant q: g = sigmoid(a + b*qbar), a = b = 0 at init;
--variant lead: g = sigmoid(logit[lead]), logits 0 at init.
q (section 6.6): per product x lead x group (momentum, heat, freshwater) = residual MSE after C / training-period flux
anomaly variance, from --q-json ({'q': [14][3]}); the lead variant ignores q (zeros in the q maps).
Training (section 6.2): origin i, no-grad warm-up of s ~ U{0..6} days, then 8 steps with gradient (leads s+1..s+8 <= 14);
loss = step_loss (state error, dou_scs_rollout_ft.py, not detached); checkpoint per step. AdamW lr 1e-4, cosine,
--origins 3000 (1500 if the smoke estimate exceeds 6 h; registered), 16 origins per update.
Reference path (unit test): branch off and corrector bypassed reproduce B exactly.
Revision 2026-10-14 (internal review F2, before any V2 run): the forcing quality is varied in training. Each sample draws lambda
(0 with p = 0.2, else uniform over 0.25/0.5/0.75/1.0/1.25); forecast-day forcing = ERA5 + lambda * pool error; C = beta(lambda)
of the OTHER fold (cross-fitted, lambda_tables.json); q = q(lambda) (mean over folds), i.e. the quality of the whole path
the branch sees, on the same scale as the deployment q of each product (q_cfs.json / q_ifs.json). lambda = 0: beta = 1,
q = 0 (accurate forcing passes unchanged). Both variants share lambda, C, starts and budget; the lead variant ignores q.
Usage: python dou_scs_v2_branch.py --backbone <DATA_D>/dou_scs/runs/seed42_ro/best.pt --variant q
       --out <DATA_D>/dou_scs/runs/v2q_seed42 [--smoke-steps 2]"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import argparse
import json
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import coare_torch as CT
import dou_scs_feet as FE
from dou_scs_rollout_ft import Roller, step_loss
from dou_scs_train import D0, ROOT, DouScs, idx
from datetime import date

GROUPS = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2}        # flux channel -> q group (momentum, heat, freshwater)


class Branch(nn.Module):
    def __init__(self, variant, cin=38 + 1 + 3, width=64):
        super().__init__()
        self.variant = variant
        self.net = nn.Sequential(nn.Conv2d(cin, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(width, 5, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        if variant == 'q':
            self.a, self.b = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))
        else:
            self.logit = nn.Parameter(torch.zeros(15))

    def forward(self, ctx, lead, q):
        """ctx [1, 38, H, W]; lead int >= 1; q tensor [3]. Returns the gated correction in the backbone output layout [lon, lat, 5]."""
        H, W = ctx.shape[-2:]
        qq = q if self.variant == 'q' else torch.zeros_like(q)
        extra = torch.cat([torch.full((1, 1, H, W), lead/14., device=ctx.device), qq.view(1, 3, 1, 1).expand(1, 3, H, W)], 1)
        r = self.net(torch.cat([ctx, extra], 1))[0]                        # [5, lat, lon]
        g = torch.sigmoid(self.a + self.b*qq.mean()) if self.variant == 'q' else torch.sigmoid(self.logit[lead])
        return g*r.permute(2, 1, 0), g


class V2Roller(Roller):
    def __init__(self, st):
        super().__init__(st)
        self.pool = FE.Pool()
        self.clim = np.memmap(ROOT/'flux_clim_doy.f32', np.float32, 'r', shape=(366, 8, 301, 301))
        t = json.loads((ROOT/'lambda_tables.json').read_text())
        self.lam = t['lam']; self.fold_of = t['fold_of_start']
        self.beta_tab = torch.tensor(t['beta'], device='cuda', dtype=torch.float32)          # [lam, fold, 14, 8]
        self.q_tab = torch.tensor(t['q'], device='cuda', dtype=torch.float32).mean(1)      # [lam, 14, 3]
        self.set_sample(0, 0)

    def set_sample(self, li, s_pool):
        self.li, self.s_pool = li, s_pool
        self.beta = self.beta_tab[li, 1 - self.fold_of[s_pool]]                            # other fold (cross-fitting)
        self.q = self.q_tab[li]

    def fflux(self, o, d, lead, s_pool):
        """normalised flux of day d; forecast days (lead >= 1): ERA5 + pool error, then beta correction."""
        a = torch.tensor(np.asarray(self.era[d]), device='cuda')
        if lead >= 1 and self.lam[self.li] > 0:
            a = a + self.lam[self.li]*torch.tensor(self.pool.field(s_pool, lead), device='cuda')
        with torch.no_grad():
            f = CT.bulk_flux(o.detach(), FE.to_iface(self.interp(a)), self.latg, self.mask)
            if lead >= 1 and not bool((self.beta[lead - 1] == 1).all()):     # beta = 1: identity, skipped (bit-exact reference path)
                c = torch.tensor(np.asarray(self.clim[(D0 + timedelta(d)).timetuple().tm_yday - 1]), device='cuda')
                f = c + self.beta[lead - 1].view(8, 1, 1)*(f - c)
        return torch.nan_to_num((f - self.fm)/self.fsd).float()

    def run(self, B, br, i, s, R, s_pool, train=True):
        o_prev, o_curr = self.truth(i - 1), self.truth(i)
        f_prev = self.fflux(o_prev, i - 1, 0, s_pool)
        lt = torch.tensor([[1.]], device='cuda'); total, per = 0., []
        for k in range(1, s + R + 1):
            cur, tgt = i + k - 1, i + k
            f_curr, f_futu = self.fflux(o_curr, cur, k - 1, s_pool), self.fflux(o_curr, tgt, k, s_pool)
            ctx = torch.cat([f_prev, torch.nan_to_num((o_prev - self.om)/self.osd), f_curr, torch.nan_to_num((o_curr - self.om)/self.osd),
                             f_futu, self.cst, self.tch(tgt)]).unsqueeze(0).float()
            grad = train and k > s

            def step(c, kk=k):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    base = B({'y_context': c, 'lt': lt})[0].float()
                corr, _ = br(c, kk, self.q[kk - 1])
                return base + corr.float()*self.mask.T[..., None]          # branch masked like the backbone (internal review 10-13 evening P1-1)
            if grad:
                out = checkpoint(step, ctx, use_reentrant=False)
            else:
                with torch.no_grad():
                    out = step(ctx)
            if k > s:
                loss = step_loss(self.lf, self.truth(tgt), o_curr, out, self.dm, self.dsd)
                total = total + loss/R; per.append(float(loss))
            o_next = torch.where(self.mask, o_curr + out.permute(2, 1, 0)*self.dsd + self.dm, torch.nan)
            if not grad:
                o_next = o_next.detach()
            f_prev, o_prev, o_curr = f_curr, o_curr, o_next
        return total, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True); ap.add_argument('--variant', choices=('q', 'lead'), required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--origins', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke-steps', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    torch.set_float32_matmul_precision('medium')
    from dou_scs_train import norm_stats
    ro = V2Roller(norm_stats())
    B = DouScs().cuda(); B.load_state_dict(torch.load(a.backbone, map_location='cuda', weights_only=True)['model']); B.eval()
    for p in B.parameters():
        p.requires_grad_(False)
    br = Branch(a.variant).cuda()
    opt = torch.optim.AdamW(br.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-5)
    n_up = a.origins//16
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: .5*(1 + math.cos(math.pi*min(u, n_up)/n_up)))
    first, last = idx(date(2000, 1, 2)), idx(date(2020, 12, 31)) - 15
    origins = rng.integers(first, last, a.origins)
    (out/'config.json').write_text(json.dumps(vars(a), indent=1))
    t0 = time.time(); losses = []; hist = []; skipped = []
    br.train(); opt.zero_grad(set_to_none=True)
    for k, i in enumerate(origins):
        i = int(i); s = int(rng.integers(0, 7)); month = (D0 + timedelta(i)).month
        s_pool = int(rng.choice(ro.pool.by_month[month]))
        li = 0 if rng.random() < .2 else int(rng.integers(1, len(ro.lam)))
        ro.set_sample(li, s_pool)
        loss, _ = ro.run(B, br, i, s, 8, s_pool)
        if not math.isfinite(float(loss)):                          # registration section 8: log and skip, stop after 5
            skipped.append(int(i)); print(json.dumps({'skipped_non_finite': int(i), 'n': len(skipped)}), flush=True)
            if len(skipped) > 5:
                raise FloatingPointError(f'more than 5 non-finite samples: {skipped}')
            continue
        (loss/16).backward(); losses.append(float(loss))
        if (k + 1) % 16 == 0:
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            u = (k + 1)//16
            if a.smoke_steps and u >= a.smoke_steps:
                dt = time.time() - t0
                rep = {'updates': u, 'sec_per_origin': round(dt/(k + 1), 2), 'est_hours': round(dt/(k + 1)*a.origins/3600, 2),
                       'gpu_peak_gb': round(torch.cuda.max_memory_allocated()/1e9, 2), 'loss_first_last': [round(losses[0], 4), round(losses[-1], 4)]}
                (out/'smoke.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep), flush=True)
                return
            if u % 25 == 0:
                hist.append({'update': u, 'loss': round(float(np.mean(losses[-400:])), 4), 'sec': round(time.time() - t0)})
                print(json.dumps(hist[-1]), flush=True); (out/'history.json').write_text(json.dumps(hist, indent=1))
    torch.save({'branch': br.state_dict(), 'variant': a.variant, 'backbone': a.backbone, 'skipped': skipped}, out/'branch.pt')
    (out/'history.json').write_text(json.dumps(hist, indent=1))


if __name__ == '__main__':
    main()
