"""(2026-10-21) Full-system comparison (docs/full_system_registration_20261021.md): combo member (ckpt given by --run) with
GLO12 surface + subsurface initial state, IFS forcing, and TRANSFERRED ZFE = the old-backbone IFS residual modes
(ob12_s2.load_modes), 6 members (+-sqrt(3) x 3 modes) added to the normalised forecast-day fluxes (k>=1), each member with
its own online COARE. Saves center and s2_ut05 (0.5 center + 0.5 members' mean) to fc_fields/fullsys/{name}/, plus
zfe_check.npz: per origin/lead/var mean offset (members' mean - center) and member RMS spread around the center (support pts).
Usage: python ob12_combo_zfe.py --run s42_combo --name c42"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

import coare_torch as CT
import dou_scs_feet as FE
from dou_scs_3d_ft import NS, SUB, Extra, Sub3D, forward3d
from dou_scs_train import D0, ROOT, DouScs, ScsDou, norm_stats
from ob12_eval import LEADS, OB, OUT, ifs_on_grid
from ob12_s2 import load_modes
from paper_budget_curves import F_LIST, VARS

SAVE = ROOT/'fc_fields/fullsys'
G12 = Path(DATA_D + '/glo12_anfc_3d')


def main():
    st = norm_stats()
    lat, lon = np.load(ROOT/'lat.npy'), np.load(ROOT/'lon.npy'); mask_np = np.load(ROOT/'mask.npy')
    era = FE.era5_025(); interp = FE.SepInterp()
    mask = torch.tensor(mask_np, device='cuda'); latg = torch.tensor(np.repeat(lat[:, None], 301, 1), device='cuda')
    now = {}
    for f in sorted(OB.glob('glo12_nowcast/*.npz')):
        z = np.load(f)
        for j, t in enumerate(z['times']):
            now[date.fromisoformat(str(t)[:10])] = np.stack([z[v][j] for v in VARS]).astype(np.float32)
    g = lambda k: torch.tensor(st[k], device='cuda', dtype=torch.float32).view(-1, 1, 1)
    om, osd, fm, fsd, dm, dsd = g('ocean_mean'), g('ocean_std'), g('flux_mean'), g('flux_std'), g('diff_mean'), g('diff_std')
    cst = torch.tensor(ScsDou(D0 + timedelta(1), D0 + timedelta(1), st).const, device='cuda')
    sub = Sub3D()
    import xarray as xr
    from dou_scs_3d_ft import SUB
    files = {}
    for k in SUB:
        fn = G12/(f'uo_vo_{k[1]}m.nc' if k[0] in ('uo', 'vo') else f'{k[0]}_{k[1]}m.nc'); files[k] = xr.open_dataset(fn)[k[0]].squeeze('depth', drop=True)
    tdays = {str(t)[:10]: j for j, t in enumerate(files[SUB[0]]['time'].values)}

    def sub_glo12(day_index):
        d = str(D0 + timedelta(day_index)); j = tdays[d]
        raw = np.stack([np.asarray(files[k].isel(time=j).values, np.float32).reshape(301, 301) for k in SUB])
        z = (raw - sub.mu[:, None, None])/sub.sd[:, None, None]
        return torch.nan_to_num(torch.tensor(z, device='cuda'))*sub.twater
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True); ap.add_argument('--name', required=True); ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--save-members', action='store_true', help='(2026-10-22, ensemble_quality_registration) save only the 7 surface members '
                    '(centre + 6 ZFE) as float16 to fc_fields/fullsys_mem/NAME; no other outputs')
    ap.add_argument('--sub-only', action='store_true', help='(2026-10-22) save only the subsurface forecasts (float16, physical units) to fc_fields/fullsys_sub/NAME; surface outputs untouched')
    ap.add_argument('--center-only', action='store_true', help='(2026-10-24, quick external check) only the unperturbed centre trajectory: surface to fullsys/NAME/center, subsurface (float32) to fullsys_sub32/NAME/center')
    ap.add_argument('--ckpt', default='final.pt')
    a = ap.parse_args()
    ck = torch.load(ROOT/f'runs/{a.run}/{a.ckpt}', map_location='cuda', weights_only=True)
    model = DouScs().cuda(); model.load_state_dict(ck['model']); model.eval()
    extra = Extra(model.decoder_lr).cuda(); extra.load_state_dict(ck['extra']); extra.eval()
    modes = load_modes(); sup = torch.tensor(np.load(OUT/'support.npy'), device='cuda')
    dd = SAVE/a.name
    lt = torch.tensor([[1.]], device='cuda')

    def rollout(o_prev, o_curr, s_prev, s_curr, it, ifs, offset):
        def flux(o, d):
            k = d - it
            aa = interp(torch.tensor(np.asarray(era[d]), device='cuda')) if k <= 0 else ifs[k]
            f = torch.nan_to_num((CT.bulk_flux(o, FE.to_iface(aa), latg, mask) - fm)/fsd).float()
            if k >= 1 and offset is not None:
                f = f + offset[k - 1]*mask
            return f
        full, subs = [], []; f_prev = flux(o_prev, it - 1)
        with torch.no_grad():
            for k in range(1, LEADS + 1):
                cur, tgt = it + k - 1, it + k
                f_curr, f_futu = flux(o_curr, cur), flux(o_curr, tgt)
                dt = D0 + timedelta(tgt); nd = 366 if dt.year % 4 == 0 else 365; doy = dt.timetuple().tm_yday - 1
                tch = torch.stack([torch.full((301, 301), float(np.sin(doy*2*np.pi/nd)), device='cuda'), torch.full((301, 301), float(np.cos(doy*2*np.pi/nd)), device='cuda')])
                ctx = torch.cat([f_prev, torch.nan_to_num((o_prev - om)/osd), f_curr, torch.nan_to_num((o_curr - om)/osd), f_futu, cst, tch]).unsqueeze(0).float()
                sctx = torch.cat([s_prev, s_curr]).unsqueeze(0).float()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out, sout = forward3d(model, extra, ctx, sctx, lt)
                out = out[0].float()
                s_next = (s_curr + sout[0].float().permute(2, 1, 0)*sub.tdsd + sub.tdm)*sub.twater
                s_prev, s_curr = s_curr, s_next
                o_next = torch.where(mask, o_curr + out.permute(2, 1, 0)*dsd + dm, torch.nan)
                full.append(o_next); subs.append(s_next)
                f_prev, o_prev, o_curr = f_curr, o_curr, o_next
        if a.sub_only or a.center_only:
            return torch.stack(full), torch.stack(subs)
        return torch.stack(full)                                      # [LEADS, 5, 301, 301]

    off_all, spr_all = [], []
    for F in (F_LIST[:1] if a.smoke else F_LIST):
        t = F - timedelta(1); it = (t - D0).days
        o_prev = torch.where(mask, torch.tensor(now[t - timedelta(1)], device='cuda'), torch.nan)
        o_curr = torch.where(mask, torch.tensor(now[t], device='cuda'), torch.nan)
        ifs = torch.tensor(ifs_on_grid(F, lat, lon), device='cuda')
        s_prev, s_curr = sub_glo12(it - 1), sub_glo12(it)
        pc = rollout(o_prev, o_curr, s_prev, s_curr, it, ifs, None)
        if a.center_only:
            phys = lambda z: torch.where(sub.twater, z*torch.tensor(sub.sd, device='cuda').view(-1, 1, 1) + torch.tensor(sub.mu, device='cuda').view(-1, 1, 1), torch.nan)
            times = np.array([str(F + timedelta(L)) for L in range(LEADS)]); fn = pc[0].cpu().numpy()
            d_ = dd/'center'; d_.mkdir(parents=True, exist_ok=True)
            np.savez(d_/f'{F:%Y%m%d}.npz', times=times, **{v: fn[:, k].astype(np.float32) for k, v in enumerate(VARS)})
            d_ = ROOT/'fc_fields/fullsys_sub32'/a.name/'center'; d_.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d_/f'{F:%Y%m%d}.npz', times=times, names=np.array([f'{v}_{z}' for v, z in SUB]), sub=phys(pc[1]).cpu().numpy().astype(np.float32))
            print(a.name, F, 'center', flush=True); continue
        mem = [rollout(o_prev, o_curr, s_prev, s_curr, it, ifs, s*np.sqrt(3)*modes[j]) for j in range(3) for s in (-1, 1)]
        if a.save_members:
            d_ = ROOT/'fc_fields/fullsys_mem'/a.name; d_.mkdir(parents=True, exist_ok=True)
            allm = torch.stack([pc] + list(mem)).cpu().numpy().astype(np.float16)            # [7, LEADS, 5, 301, 301]
            np.savez_compressed(d_/f'{F:%Y%m%d}.npz', times=np.array([str(F + timedelta(L)) for L in range(LEADS)]), members=allm)
            print(a.name, F, 'members', flush=True); continue
        if a.sub_only:
            phys = lambda z: torch.where(sub.twater, z*torch.tensor(sub.sd, device='cuda').view(-1, 1, 1) + torch.tensor(sub.mu, device='cuda').view(-1, 1, 1), torch.nan)
            sc = phys(pc[1]); se = sum(phys(m_[1]) for m_ in mem)/6.
            times = np.array([str(F + timedelta(L)) for L in range(LEADS)])
            for n_, fld in (('center', sc), ('s2_ut05', .5*sc + .5*se)):
                d_ = ROOT/'fc_fields/fullsys_sub32'/a.name/n_; d_.mkdir(parents=True, exist_ok=True)   # 2026-10-22: float32 (float16 copy kept in fullsys_sub)
                np.savez_compressed(d_/f'{F:%Y%m%d}.npz', times=times, names=np.array([f'{v}_{z}' for v, z in SUB]), sub=fld.cpu().numpy().astype(np.float32))
            print(a.name, F, 'sub', flush=True); continue
        ens = sum(mem)/6.
        m = sup & mask
        off_all.append(((ens - pc)[:, :, m]).double().mean(-1).cpu().numpy())
        spr_all.append(torch.sqrt(sum(((x - pc)[:, :, m].double())**2 for x in mem).mean(-1)/6.).cpu().numpy())
        times = np.array([str(F + timedelta(L)) for L in range(LEADS)])
        for n_, fld in (('center', pc), ('s2_ut05', .5*pc + .5*ens)):
            d_ = dd/n_; d_.mkdir(parents=True, exist_ok=True)
            fn = fld.cpu().numpy()
            np.savez(d_/f'{F:%Y%m%d}.npz', times=times, **{v: fn[:, k].astype(np.float32) for k, v in enumerate(VARS)})
        print(a.name, F, flush=True)
    if a.sub_only or a.save_members or a.center_only:
        return
    np.savez(dd/'zfe_check.npz', starts=np.array([str(F) for F in F_LIST]), vars=np.array(VARS), offset=np.stack(off_all), spread=np.stack(spr_all))

if __name__ == '__main__':
    main()
