"""GPU (torch) port of the authors' COARE 3.5 path used by Dou et al.: coare35vn.coare35vn (no wave-age / sea-state terms,
jcool = 1) + meteo helpers + forecast_real_from_GLO12v4_ERA5_nc.compute_bulk_flux_np. Line-by-line translation, float64.
Needed because FEET perturbs the atmosphere per training sample, so the fluxes must be recomputed on the fly
(numpy COARE: ~0.25 s per field on the CPU). Verified against the numpy implementation by test_coare_torch.py.
Inputs to bulk_flux: ocean [..., 5, H, W] (uo, vo, thetao, so, zos; NaN on land), atm dict of [..., H, W] in the authors'
interface (u10, v10 m/s; t2m K; sh kg/kg; ssrd, strd J m-2 per hour; tp m per hour), lat [H, W], mask [H, W] bool.
Output [..., 8, H, W]: tau_u, tau_v, sensible, latent, net_short, net_long, evap, rain (NaN on land / invalid)."""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import math

import torch


def _qsat(t, p):
    return 6.1121*torch.exp(17.502*t/(240.97 + t))*(1.0007 + p*3.46e-6)


def _psi(zet, a, b1, c1, b2):
    """common form of psiu_26 / psiu_40: stable part -(a zet + 3/4 (zet - 5/0.35) exp(-dzet) + 3/4*5/0.35)."""
    dzet = torch.clamp(0.35*zet, max=50.)
    psi = -(a*zet + 0.75*(zet - 5./0.35)*torch.exp(-dzet) + 0.75*5./0.35)
    zu = torch.clamp(zet, max=0.)                       # unstable branch evaluated safely everywhere, selected where zet < 0
    x = (1 - b1*zu)**0.25
    psik = 2.*torch.log((1. + x)/2.) + torch.log((1. + x*x)/2.) - 2.*torch.atan(x) + 2.*math.atan(1.)
    x = (1 - b2*zu)**0.3333
    psic = 1.5*torch.log((1. + x + x**2)/3.) - math.sqrt(3.)*torch.atan((1. + 2.*x)/math.sqrt(3.)) + 4*math.atan(1.)/math.sqrt(3.)
    f = zu**2/(1. + zu**2)
    return torch.where(zet < 0, (1 - f)*psik + f*psic, psi)


def psiu_26(zet):
    return _psi(zet, 0.7, 15., None, 10.15)


def psiu_40(zet):
    return _psi(zet, 1., 18., None, 10.)


def psit_26(zet):
    dzet = torch.clamp(0.35*zet, max=50.)
    psi = -(torch.clamp(1 + 0.6667*zet, min=0)**1.5 + 0.6667*(zet - 14.28)*torch.exp(-dzet) + 8.525)
    zu = torch.clamp(zet, max=0.)
    x = (1 - 15*zu)**0.5
    psik = 2*torch.log((1 + x)/2.)
    x = (1 - 34.15*zu)**0.3333
    psic = 1.5*torch.log((1. + x + x**2)/3.) - math.sqrt(3)*torch.atan((1 + 2*x)/math.sqrt(3)) + 4*math.atan(1.)/math.sqrt(3.)
    f = zu**2/(1. + zu**2.)
    return torch.where(zet < 0, (1 - f)*psik + f*psic, psi)


def grv(lat):
    x = torch.sin(lat*math.pi/180)
    return 9.7803267715*(1 + 0.0052790414*x**2 + 0.0000232718*x**4 + 0.0000001262*x**6 + 0.0000000007*x**8)


def coare35vn(u, t, rh, ts, P, Rs, Rl, rain, lat, zu=10., zt=2., zq=2., zi=600.):
    """returns tau, hsb, hlb, Rns, Rnl, Evap (mm/h), rain -- as the authors' coare35vn columns 0..6."""
    jcool = 1.
    Qs = 0.98*_qsat(ts, P); Qs = 622*Qs/(P - 0.378*Qs)/1000.
    em = rh/100.*_qsat(t, P); Q = 621.97*em/(P - 0.378*em); Pv = em; Q = Q/1000.
    Beta, von, fdg, tdk, Rgas = 1.2, 0.4, 1.00, 273.16, 287.1
    grav = grv(lat)
    Le = (2.501 - 0.00237*ts)*1e6
    cpa = 1004.67
    rhoa = P*100./(Rgas*(t + tdk)*(1 + 0.61*Q))
    visa = 1.326e-5*(1 + 6.542e-3*t + 8.301e-6*t**2 - 4.84e-9*t**3)
    Al = 2.1e-5*(ts + 3.2)**0.79
    be, cpw, rhow, visw, tcw = 0.026, 4000., 1022., 1.e-6, 0.6
    bigc = 16.*grav*cpw*(rhow*visw)**3/(tcw**2*rhoa**2)
    wetc = 0.622*Le*Qs/(Rgas*(ts + tdk)**2)
    Rns = 0.945*Rs
    Rnl = 0.97*(5.67e-8*(ts - 0.3*jcool + tdk)**4 - Rl)
    du = u
    dt = ts - t - 0.0098*zt
    dq = Qs - Q
    ta = t + tdk
    ug = 0.5
    dter = torch.full_like(u, 0.3)
    ut = torch.sqrt(du**2 + ug**2)
    u10 = ut*math.log(10/1e-4)/math.log(zu/1e-4)
    usr = 0.035*u10
    zo10 = 0.011*usr**2/grav + 0.11*visa/usr
    Cd10 = (von/torch.log(10/zo10))**2
    Ch10 = 0.00115
    Ct10 = Ch10/torch.sqrt(Cd10)
    zot10 = 10/torch.exp(von/Ct10)
    Cd = (von/torch.log(zu/zo10))**2
    Ct = von/torch.log(zt/zot10)
    CC = von*Ct/Cd
    Ribcu = -zu/zi/0.004/Beta**3
    Ribu = -grav*zu/ta*((dt - dter*jcool) + 0.61*ta*dq)/ut**2
    zetu = CC*Ribu*(1 + 27/9*Ribu/CC)
    k50 = zetu > 50
    zetu = torch.where(Ribu < 0, CC*Ribu/(1 + Ribu/Ribcu), zetu)
    L10 = zu/zetu
    gf = ut/du
    usr = ut*von/(torch.log(zu/zo10) - psiu_40(zu/L10))
    tsr = -(dt - dter*jcool)*von*fdg/(torch.log(zt/zot10) - psit_26(zt/L10))
    qsr = -(dq - wetc*dter*jcool)*von*fdg/(torch.log(zq/zot10) - psit_26(zq/L10))
    tkt = torch.full_like(u, 0.001)
    umax, a1, a2 = 19, 0.0017, -0.0050
    charnC = torch.where(u10 > umax, torch.full_like(u10, a1*umax + a2), a1*u10 + a2)
    saved = None
    for i in range(10):
        zet = von*grav*zu/ta*(tsr + 0.61*ta*qsr)/(usr**2)
        charn = charnC
        L = zu/zet
        zo = charn*usr**2/grav + 0.11*visa/usr
        rr = zo*usr/visa
        zoq = torch.clamp(5.8e-5/rr**0.72, max=1.6e-4)
        zot = zoq
        cdhf = von/(torch.log(zu/zo) - psiu_26(zu/L))
        cqhf = von*fdg/(torch.log(zq/zoq) - psit_26(zq/L))
        cthf = von*fdg/(torch.log(zt/zot) - psit_26(zt/L))
        usr = ut*cdhf
        qsr = -(dq - wetc*dter*jcool)*cqhf
        tsr = -(dt - dter*jcool)*cthf
        tvsr = tsr + 0.61*ta*qsr
        Bf = -grav/ta*usr*tvsr
        ug = torch.where(Bf > 0, Beta*(torch.clamp(Bf, min=0)*zi)**0.333, torch.full_like(Bf, 0.2))
        ut = torch.sqrt(du**2 + ug**2)
        gf = ut/du
        hsb = -rhoa*cpa*usr*tsr
        hlb = -rhoa*Le*usr*qsr
        qout = Rnl + hsb + hlb
        dels = Rns*(0.065 + 11*tkt - 6.6e-5/tkt*(1 - torch.exp(-tkt/8.0e-4)))
        qcol = qout - dels
        alq = Al*qcol + be*hlb*cpw/Le
        xlamx_pos = 6/(1 + (bigc*torch.clamp(alq, min=0)/usr**4)**0.75)**0.333
        tkt_neg = torch.clamp(6.0*visw/(torch.sqrt(rhoa/rhow)*usr), max=0.01)
        tkt = torch.where(alq > 0, xlamx_pos*visw/(torch.sqrt(rhoa/rhow)*usr), tkt_neg)
        dter = qcol*tkt/tcw
        Rnl = 0.97*(5.67e-8*(ts - dter*jcool + tdk)**4 - Rl)
        if i == 0:
            saved = (usr.clone(), tsr.clone(), qsr.clone())
        u10N = usr/von/gf*torch.log(10/zo)
        charnC = torch.where(u10N > umax, torch.full_like(u10N, a1*umax + a2), a1*u10N + a2)
    usr = torch.where(k50, saved[0], usr); tsr = torch.where(k50, saved[1], tsr); qsr = torch.where(k50, saved[2], qsr)
    tau = rhoa*usr*usr/gf
    hsb = -rhoa*cpa*usr*tsr
    hlb = -rhoa*Le*usr*qsr
    Evap = 1000*hlb/Le/1000*3600
    return tau, hsb, hlb, Rns, Rnl, Evap, rain


def rh_calc(Tc, p, q):
    es = 6.1121*torch.exp(17.502*Tc/(Tc + 240.97))*(1.0007 + 3.46e-6*p)
    em = q*p/(0.378*q + 0.622)
    return torch.clamp(100.0*em/es, 0.0, 100.0)


def bulk_flux(ocean, atm, lat, mask, pressure_hpa=1015.0):
    """torch version of the authors' compute_bulk_flux_np (same conventions: relative wind u10 - uo, Rs = ssrd/3600,
    Rl = strd/3600, rain = tp*1000 mm/h, P = 1015 hPa, zu 10, zt = zq = 2, zi 600)."""
    o = ocean.double()
    a = {k: v.double() for k, v in atm.items()}
    du, dv = a['u10'] - o[..., 0, :, :], a['v10'] - o[..., 1, :, :]
    sst = o[..., 2, :, :]
    spd = torch.sqrt(du**2 + dv**2)
    t2m_c = a['t2m'] - 273.15
    P = torch.full_like(spd, pressure_hpa)
    rh = rh_calc(t2m_c, P, a['sh'])
    valid = mask & torch.isfinite(spd) & torch.isfinite(t2m_c) & torch.isfinite(rh) & torch.isfinite(sst) \
        & torch.isfinite(a['ssrd']) & torch.isfinite(a['strd']) & torch.isfinite(a['tp'])
    fill = lambda x, v: torch.where(valid, x, torch.full_like(x, v))      # safe dummy values off the valid set
    latb = torch.broadcast_to(lat.double(), spd.shape)
    tau, hsb, hlb, rsn, rln, evap, rain = coare35vn(fill(spd, 5.), fill(t2m_c, 25.), fill(rh, 80.), fill(sst, 26.), P,
                                                    fill(a['ssrd']/3600., 200.), fill(a['strd']/3600., 400.),
                                                    fill(a['tp']*1000., 0.), latb)
    tau_u, tau_v = tau*du/spd, tau*dv/spd
    out = torch.stack([tau_u, tau_v, hsb, hlb, rsn, rln, evap, rain], -3)
    nan = torch.tensor(float('nan'), dtype=out.dtype, device=out.device)
    out = torch.where(valid.unsqueeze(-3), out, nan)
    return torch.where(torch.isfinite(out), out, nan).float()
