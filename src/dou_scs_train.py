"""Steps 4-5 of docs/method_dou_zero_arm_scs_registration_20261011.md: Dou zero-arm on the SCS 1/12-deg grid.
Model = the authors' ViT backbone exactly as in model.ConvCNPSCS (h 256, depth 8, patch 4, window 7; the ViT pads to a
multiple of patch*window internally), img_size [301, 301]; output masked with the GLORYS12 ocean mask.
Sample (authors' loader __getitem__): prev = i-1, curr = i, target = i+1;
  y_context = [F(prev), O(prev), F(curr), O(curr), F_lead1(curr), deptho, mdt, sin/cos doy(target)]  (38 channels)
  y_target  = normalised tendency O(target) - O(curr), layout [lon, lat, 5] as in the authors' code.
Normalisation statistics from the training years only (2000-2020), computed once and stored (norm_stats.npz).
Training = the authors' Trainer: AdamW(lr, betas (0.9, 0.95), weight_decay 1e-5), WeightedRmseLoss (no variable weights),
CosineAnnealingLR(T_max = steps_per_epoch * (epochs - 10)) stepped per batch from epoch 11 on, validation (dev 2021) after
every epoch, best epoch saved. Batch 16 via gradient accumulation of micro-batches (loss = mean of per-sample RMSE, so
accumulation of micro-batch means / n_micro is identical). bf16 autocast (the authors ran TF32).
--arm feet (docs/method_dou_feet_pilot_registration_20261011.md): FEET draws per sample (dou_scs_feet.ScsDouFeet), lt = k,
flux blocks of perturbed samples recomputed on the GPU before the forward pass; workers are re-created every epoch so
that the draw epoch is passed to them.
Usage: python dou_scs_train.py --epochs E [--arm dou|feet] [--micro 2] [--smoke-steps 20] --seed 42 --out <DATA_D>/dou_scs/runs/seed42"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import argparse
import json
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import netCDF4
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = DOU_REPO + ''
sys.path[:0] = [EXTRA_DEPS + '', REPO]
from vit_new import ViT                      # noqa: E402  (authors' code)
from loss_functions import WeightedRmseLoss  # noqa: E402

ROOT = Path(DATA_D + '/dou_scs')
STATIC = DATA_E + '/glorys12_static_scs/'
META = json.loads((ROOT/'meta.json').read_text())
D0 = date.fromisoformat(META['first_day'])
N, H, W = META['N'], 301, 301
idx = lambda d: (d - D0).days


def memmaps():
    return (np.memmap(ROOT/'ocean.f32', np.float32, 'r', shape=(N, 5, H, W)),
            np.memmap(ROOT/'flux0.f32', np.float32, 'r', shape=(N, 8, H, W)),
            np.memmap(ROOT/'flux1.f32', np.float32, 'r', shape=(N, 8, H, W)))


def norm_stats(train_last=date(2020, 12, 31)):
    p = ROOT/'norm_stats.npz'
    if p.exists():
        return dict(np.load(p))
    oc, f0, _ = memmaps()
    mask = np.load(ROOT/'mask.npy')
    days = np.arange(0, idx(train_last) + 1, 5)                       # every 5th training day (unbiased, cheap)
    o = np.stack([oc[i][:, mask] for i in days]).astype(np.float64)   # [n, 5, ocean]
    dd = np.stack([(oc[i + 1] - oc[i])[:, mask] for i in days if i + 1 <= idx(train_last)]).astype(np.float64)
    fl = np.stack([f0[i][:, mask] for i in days]).astype(np.float64)
    with netCDF4.Dataset(STATIC + 'glorys12_static_bathy_scs.nc') as b, netCDF4.Dataset(STATIC + 'glorys12_static_mdt_scs.nc') as m:
        dep = np.ma.filled(b['deptho'][:], np.nan)[mask]; mdt = np.ma.filled(m['mdt'][:], np.nan)[mask]
    st = {'ocean_mean': np.nanmean(o, (0, 2)), 'ocean_std': np.nanstd(o, (0, 2)),
          'diff_mean': np.nanmean(dd, (0, 2)), 'diff_std': np.nanstd(dd, (0, 2)),
          'flux_mean': np.nanmean(fl, (0, 2)), 'flux_std': np.nanstd(fl, (0, 2)),
          'const_mean': np.array([np.nanmean(dep), np.nanmean(mdt)]), 'const_std': np.array([np.nanstd(dep), np.nanstd(mdt)])}
    np.savez(p, **st)
    return st


class ScsDou(Dataset):
    def __init__(self, first_curr, last_curr, st):
        self.ids = np.arange(idx(first_curr), idx(last_curr) + 1)
        self.st = {k: v.astype(np.float32) for k, v in st.items()}
        with netCDF4.Dataset(STATIC + 'glorys12_static_bathy_scs.nc') as b, netCDF4.Dataset(STATIC + 'glorys12_static_mdt_scs.nc') as m:
            dep = np.ma.filled(b['deptho'][:], np.nan); mdt = np.ma.filled(m['mdt'][:], np.nan)
        self.const = np.nan_to_num(np.stack([(dep - self.st['const_mean'][0])/self.st['const_std'][0],
                                             (mdt - self.st['const_mean'][1])/self.st['const_std'][1]])).astype(np.float32)
        self.mm = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        if self.mm is None:
            self.mm = memmaps()
        oc, f0, f1 = self.mm
        i = int(self.ids[k]); s = self.st
        on = lambda x: np.nan_to_num((x - s['ocean_mean'][:, None, None])/s['ocean_std'][:, None, None])
        fn = lambda x: np.nan_to_num((x - s['flux_mean'][:, None, None])/s['flux_std'][:, None, None])
        o_prev, o_curr, o_tgt = np.asarray(oc[i - 1]), np.asarray(oc[i]), np.asarray(oc[i + 1])
        tgt = np.nan_to_num(((o_tgt - o_curr) - s['diff_mean'][:, None, None])/s['diff_std'][:, None, None])
        dt = D0 + timedelta(i + 1); n_days = 366 if dt.year % 4 == 0 else 365; doy = dt.timetuple().tm_yday - 1
        tch = np.stack([np.full((H, W), np.sin(doy*2*np.pi/n_days), np.float32), np.full((H, W), np.cos(doy*2*np.pi/n_days), np.float32)])
        ctx = np.concatenate([fn(np.asarray(f0[i - 1])), on(o_prev), fn(np.asarray(f0[i])), on(o_curr), fn(np.asarray(f1[i])),
                              self.const, tch], 0).astype(np.float32)
        return {'y_context': torch.from_numpy(ctx), 'y_target': torch.from_numpy(tgt.transpose(2, 1, 0).copy()),
                'lt': torch.tensor([1.], dtype=torch.float32)}


class DouScs(torch.nn.Module):
    """model.ConvCNPSCS with the SCS grid: same ViT call, same output permutation and masking."""
    def __init__(self):
        super().__init__()
        self.decoder_lr = ViT(in_channels=38, out_channels=5, h_channels=256, depth=8, patch_size=4, img_size=[H, W], window_size=7)
        import os as _os
        if _os.environ.get('SUB3D_FOLDBN') == '1':          # (2026-10-24) exact BN folding in the surface head, see head_fold.py
            from head_fold import fold_head
            fold_head(self.decoder_lr.head)
        m = torch.from_numpy(np.load(ROOT/'mask.npy').astype(np.float32))
        self.register_buffer('mask', m[None, None])                      # [1, 1, lat, lon]

    def forward(self, task, film_index=1.0):
        x = self.decoder_lr(task['y_context'], lead_times=task['lt'])
        x = x.permute(0, 2, 1, 3)
        return x*self.mask.permute(0, 3, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--micro', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke-steps', type=int, default=0)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--out', default=str(ROOT/'runs'/'seed42'))
    ap.add_argument('--arm', default='dou', choices=('dou', 'feet'))
    ap.add_argument('--init-from', default=None, help='fine-tune: start from these model weights (fresh optimiser)')
    ap.add_argument('--hold-epochs', type=int, default=10, help='epochs before cosine annealing starts (authors: 10; fine-tune R1: 0)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_float32_matmul_precision('medium')
    st = norm_stats()
    feet = a.arm == 'feet'
    if feet:
        import dou_scs_feet as FE
        FE.build_era5_025()
        train = FE.ScsDouFeet(date(2000, 1, 2), date(2020, 12, 30), st, seed=a.seed)
        interp = FE.SepInterp()
        latg = torch.tensor(np.repeat(np.load(ROOT/'lat.npy')[:, None], W, 1), device='cuda')
        maskg = torch.tensor(np.load(ROOT/'mask.npy'), device='cuda')
    else:
        train = ScsDou(date(2000, 1, 2), date(2020, 12, 30), st)
    dev = ScsDou(date(2021, 1, 2), date(2021, 12, 30), st)
    g = torch.Generator(); g.manual_seed(a.seed)
    tl = DataLoader(train, batch_size=a.micro, shuffle=True, drop_last=True, num_workers=a.workers, generator=g, persistent_workers=not feet, pin_memory=True)
    vl = DataLoader(dev, batch_size=a.micro, shuffle=False, num_workers=a.workers, persistent_workers=True, pin_memory=True)
    model = DouScs().cuda()
    if a.init_from:
        model.load_state_dict(torch.load(a.init_from, map_location='cuda', weights_only=True)['model'])
    lf = WeightedRmseLoss(start_ind=0, end_ind=5, weight_per_variable=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=1e-5)
    n_acc = a.batch // a.micro
    steps_per_epoch = len(tl) // n_acc
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps_per_epoch*max(a.epochs - a.hold_epochs, 1))
    ck = out/'checkpoint.pt'; hist, best, e0 = [], 1e9, 0
    if ck.exists():
        c = torch.load(ck, map_location='cuda', weights_only=False)
        model.load_state_dict(c['model']); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched'])
        hist, best, e0 = c['hist'], c['best'], c['epoch'] + 1
        print(json.dumps({'resumed_after_epoch': c['epoch']}), flush=True)
    (out/'config.json').write_text(json.dumps(vars(a) | {'steps_per_epoch': steps_per_epoch, 'n_train': len(train), 'n_dev': len(dev)}, indent=1))
    for epoch in range(e0, a.epochs):
        model.train(); t0 = time.time(); losses = []; opt.zero_grad(set_to_none=True); k = 0
        if feet:
            train.epoch = epoch
        for j, task in enumerate(tl):
            task = {kk: v.cuda(non_blocking=True) for kk, v in task.items()}
            if feet:
                FE.perturb_context(task, interp, latg, maskg, st)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out_ = model(task)
            loss = lf(task['y_target'], out_.float())/n_acc
            loss.backward(); losses.append(float(loss)*n_acc)
            if (j + 1) % n_acc == 0:
                opt.step(); opt.zero_grad(set_to_none=True); k += 1
                if epoch >= a.hold_epochs and (a.hold_epochs == 0 or epoch > a.hold_epochs):
                    sched.step()
                if a.smoke_steps and k >= a.smoke_steps:
                    dt = time.time() - t0
                    rep = {'smoke_steps': k, 'sec': round(dt, 1), 'sec_per_step': round(dt/k, 2), 'steps_per_epoch': steps_per_epoch,
                           'est_hours_per_epoch': round(dt/k*steps_per_epoch/3600, 2), 'gpu_peak_gb': round(torch.cuda.max_memory_allocated()/1e9, 2),
                           'loss_first_last': [round(losses[0], 4), round(losses[-1], 4)]}
                    (out/'smoke.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep), flush=True)
                    return
        model.eval(); per = []
        with torch.no_grad():
            for task in vl:
                task = {kk: v.cuda(non_blocking=True) for kk, v in task.items()}
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    o = model(task)
                per.append(lf(task['y_target'], o.float(), expand=True).cpu().numpy())
        pv = np.mean(per, 0); val = float(np.nanmean(pv))
        hist.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val_loss': val, 'val_per_var': pv.round(4).tolist(),
                     'lr': opt.param_groups[0]['lr'], 'sec': round(time.time() - t0)})
        print(json.dumps(hist[-1]), flush=True)
        if val < best:
            best = val; torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': val}, out/'best.pt')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'hist': hist, 'best': best, 'epoch': epoch}, out/'checkpoint.tmp')
        Path(out/'checkpoint.tmp').replace(ck)
        (out/'history.json').write_text(json.dumps(hist, indent=1))


if __name__ == '__main__':
    main()
