"""Signal-to-noise-weighted shrinkage of forecast fluxes (forcing stage, Eqs. for beta and alpha_p in the paper).

For forecast lead k >= 1 the COARE flux field F of an operational product p is replaced by
    F_tilde = c + [1 - alpha_p * (1 - beta[k-1])] * (F - c),
where c is the day-of-year flux climatology (2000-2020, ERA5 + GLORYS12V1), beta[k-1] holds one coefficient per flux channel
(8 channels: tau_x, tau_y, sensible, latent, net shortwave, net longwave, evaporation, precipitation) estimated by
r2_shrink_build.py from CFSv2 forecasts of 2018-2020, and alpha_p is the product strength selected on the 2021 validation year
(alpha_CFS = 0.5, alpha_UKMO = 0.25, alpha_IFS = 0). Lead 0 (analysis days) is never shrunk.
This is the same arithmetic as in the rollouts of the independent-year tests (ukmo_indep.py / cfs_indep.py of the full code).
"""
import json

import torch

ALPHA = {'cfs': 0.5, 'ukmo': 0.25, 'ifs': 0.0}


def load_beta(path, device='cuda'):
    """beta table [14 leads, 8 channels] from the JSON written by r2_shrink_build.py (key 'beta')."""
    with open(path, encoding='utf-8') as f:
        return torch.tensor(json.load(f)['beta'], device=device, dtype=torch.float32)


def shrink_flux(f, clim, beta, k, alpha):
    """Shrink a forecast flux field towards its climatology.

    f, clim : tensors [8, H, W] (physical units), forecast flux and day-of-year climatology of the valid day
    beta    : tensor [14, 8] from load_beta
    k       : forecast lead in days (1-14); k <= 0 returns f unchanged
    alpha   : product strength alpha_p (see ALPHA)
    """
    if k < 1 or alpha <= 0:
        return f
    w = (1. - alpha*(1. - beta[k - 1])).view(-1, 1, 1)
    return clim + w*(f - clim)
