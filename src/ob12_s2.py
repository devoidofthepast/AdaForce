"""OceanBench 2024 (1/12 deg, IFS forcing) for the frozen system of oceanbench_1_12_protocol_draft_20261012.md (S2 appendix):
RO backbone + V2-lead branch with the SST output channel off (IFS alpha = 0, no flux correction) + S2 zero-mean symmetric
ensemble (6 members = +-sqrt(3) x the 3 IFS 2025H1 residual modes, added to the normalised forecast-day fluxes, each member
rolled with its own online COARE, ocean states averaged). Variants scored: center, s2_full (members' mean),
s2_ut05 (0.5 center + 0.5 members' mean, primary). Same starts, dates, interpolation, support (support.npy) and scoring as
ob12_eval.py; per-origin MSE saved. Output: <DATA_E>/experiments/paper_budget/ob12_s2.json (+ _per_origin.npz)"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.ndimage import distance_transform_edt

import coare_torch as CT
import dou_scs_feet as FE
from dou_scs_train import D0, N, ROOT, DouScs, ScsDou, norm_stats
from dou_scs_v2_branch import Branch
from ob12_eval import LEADS, OB, OUT, ifs_on_grid, summary
from paper_budget_curves import F_LIST, VARS

RET = Path(ZFE_DIR + '')


def load_modes():
    coarse = np.load(RET/'coarse_support.npy'); small = coarse[::6, ::6]
    nearest = distance_transform_edt(~small, return_distances=False, return_indices=True)
    a = np.load(RET/'ifs_modes_coarse.npy')
    g = np.zeros((3, 14, 8, 51, 51), np.float32)
    g[..., small] = a.transpose(0, 1, 3, 2)
    g = g[..., nearest[0], nearest[1]]
    return Fnn.interpolate(torch.tensor(g.reshape(-1, 1, 51, 51), device='cuda'), size=(301, 301), mode='bilinear',
                           align_corners=True).reshape(3, 14, 8, 301, 301)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--tracks', nargs='*', default=['G', 'R']); ap.add_argument('--save-dir', default=None)
    ap.add_argument('--tag', default='ob12_s2')
    ap.add_argument('--run', default='seed42_ro'); ap.add_argument('--branch-run', default='v2lead2_seed42')
    ap.add_argument('--no-branch', action='store_true', help='RO + S2 only (no V2 branch)')
    ap.add_argument('--gamma', default='tau', choices=('tau', 'sst_off'),
                    help='branch output channels for IFS: tau = registration section 11 (channels_ON of tau_bias_transfer.json, '
                         'default since 2026-10-18); sst_off = the earlier configuration (u, v, SSS, SSH on, SST off)')
    a = ap.parse_args()
    st = norm_stats()
    lat, lon = np.load(ROOT/'lat.npy'), np.load(ROOT/'lon.npy'); mask_np = np.load(ROOT/'mask.npy')
    support = np.load(OUT/'support.npy')
    oc = np.memmap(ROOT/'ocean.f32', np.float32, 'r', shape=(N, 5, 301, 301))
    era = FE.era5_025(); interp = FE.SepInterp()
    mask = torch.tensor(mask_np, device='cuda'); latg = torch.tensor(np.repeat(lat[:, None], 301, 1), device='cuda')
    sup = torch.tensor(support, device='cuda'); maskT = mask.T[..., None]
    w = np.cos(np.deg2rad(lat))[:, None]*support; w = torch.tensor(w/w.sum(), device='cuda')[sup]
    now = {}
    for f in sorted(OB.glob('glo12_nowcast/*.npz')):
        z = np.load(f)
        for j, t in enumerate(z['times']):
            now[date.fromisoformat(str(t)[:10])] = np.stack([z[v][j] for v in VARS]).astype(np.float32)
    g = lambda k: torch.tensor(st[k], device='cuda', dtype=torch.float32).view(-1, 1, 1)
    om, osd, fm, fsd, dm, dsd = g('ocean_mean'), g('ocean_std'), g('flux_mean'), g('flux_std'), g('diff_mean'), g('diff_std')
    cst = torch.tensor(ScsDou(D0 + timedelta(1), D0 + timedelta(1), st).const, device='cuda')
    B = DouScs().cuda(); B.load_state_dict(torch.load(ROOT/'runs'/a.run/'best.pt', map_location='cuda', weights_only=True)['model']); B.eval()
    bc = torch.load(ROOT/'runs'/a.branch_run/'branch.pt', map_location='cuda', weights_only=False)
    br = Branch(bc['variant']).cuda(); br.load_state_dict(bc['branch']); br.eval()
    if a.gamma == 'tau':                                          # VARS order: uo, vo, thetao, so, zos
        on = json.loads(Path(DATA_E + '/experiments/dou_scs/tau_bias_transfer.json').read_text())['ifs']['channels_ON']
        gam = [float(on['u_v_SSH']), float(on['u_v_SSH']), float(on['SST']), float(on['SSS']), float(on['u_v_SSH'])]
    else:
        gam = [1., 1., 0., 1., 1.]
    nonSST = torch.tensor(gam, device='cuda'); q0 = torch.zeros(3, device='cuda'); print('branch channels (u v SST SSS SSH):', gam, flush=True)
    modes = load_modes()

    def rollout(o_prev, o_curr, it, ifs, offset):
        def flux(o, d):
            k = d - it
            a = interp(torch.tensor(np.asarray(era[d]), device='cuda')) if k <= 0 else ifs[k]
            f = torch.nan_to_num((CT.bulk_flux(o, FE.to_iface(a), latg, mask) - fm)/fsd).float()
            if k >= 1 and offset is not None:
                f = f + offset[k - 1]*mask
            return f
        preds, full = [], []
        f_prev = flux(o_prev, it - 1)
        with torch.no_grad():
            for k in range(1, LEADS + 1):
                cur, tgt = it + k - 1, it + k
                f_curr, f_futu = flux(o_curr, cur), flux(o_curr, tgt)
                dt = D0 + timedelta(tgt); nd = 366 if dt.year % 4 == 0 else 365; doy = dt.timetuple().tm_yday - 1
                tch = torch.stack([torch.full((301, 301), float(np.sin(doy*2*np.pi/nd)), device='cuda'), torch.full((301, 301), float(np.cos(doy*2*np.pi/nd)), device='cuda')])
                ctx = torch.cat([f_prev, torch.nan_to_num((o_prev - om)/osd), f_curr, torch.nan_to_num((o_curr - om)/osd), f_futu, cst, tch]).unsqueeze(0).float()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out = B({'y_context': ctx, 'lt': torch.tensor([[1.]], device='cuda')})[0].float()
                if not a.no_branch:
                    corr, _ = br(ctx, k, q0)
                    out = out + corr.float()*nonSST*maskT
                o_next = torch.where(mask, o_curr + out.permute(2, 1, 0)*dsd + dm, torch.nan)
                preds.append(o_next[:, sup].double()); full.append(o_next.cpu().numpy())
                f_prev, o_prev, o_curr = f_curr, o_curr, o_next
        return torch.stack(preds), np.stack(full)                   # [LEADS, 5, n_support], [LEADS, 5, 301, 301]

    names = ('center', 's2_full', 's2_ut05')
    res, per = {'n_points': int(support.sum()), 'config': f'{a.run} + {a.branch_run} branch (channels {gam}, gamma={a.gamma}) + S2 IFS modes; primary s2_ut05'}, {}
    for track in a.tracks:
        truth = (lambda d: now.get(d)) if track == 'G' else (lambda d: np.asarray(oc[(d - D0).days]))
        mse = {n: np.zeros((len(F_LIST), LEADS, 5)) for n in names}
        for si, F in enumerate(F_LIST):
            t = F - timedelta(1); it = (t - D0).days
            o_prev = torch.where(mask, torch.tensor(truth(t - timedelta(1)), device='cuda'), torch.nan)
            o_curr = torch.where(mask, torch.tensor(truth(t), device='cuda'), torch.nan)
            ifs = torch.tensor(ifs_on_grid(F, lat, lon), device='cuda')
            pc, fc_ = rollout(o_prev, o_curr, it, ifs, None)
            mem = [rollout(o_prev, o_curr, it, ifs, s*np.sqrt(3)*modes[j]) for j in range(3) for s in (-1, 1)]
            ens = sum(m_[0] for m_ in mem)/6.; fens = sum(m_[1] for m_ in mem)/6.
            P = {'center': pc, 's2_full': ens, 's2_ut05': .5*pc + .5*ens}
            if a.save_dir and track == 'G':
                times = np.array([str(F + timedelta(L)) for L in range(LEADS)])
                for n_, fld in (('center', fc_), ('s2_ut05', .5*fc_ + .5*fens)):
                    dd = Path(a.save_dir)/n_; dd.mkdir(parents=True, exist_ok=True)
                    np.savez(dd/f'{F:%Y%m%d}.npz', times=times, **{v: fld[:, k].astype(np.float32) for k, v in enumerate(VARS)})
            for L in range(LEADS):
                tr = torch.tensor(truth(D0 + timedelta(it + L + 1)), device='cuda')[:, sup].double()
                for n in names:
                    e = P[n][L] - tr
                    if not torch.isfinite(e).all():
                        raise ValueError(f'{n} {track} {F} lead {L + 1}: non-finite inside support')
                    mse[n][si, L] = (w.double()*e**2).sum(1).cpu().numpy()
            print(track, F, flush=True)
        res[track] = {n: summary(mse[n]) for n in names}
        for n in names:
            per[f'{track}_{n}'] = mse[n]
    np.savez(OUT/f'{a.tag}_per_origin.npz', starts=np.array([str(F) for F in F_LIST]), vars=np.array(VARS), **per)
    (OUT/f'{a.tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
