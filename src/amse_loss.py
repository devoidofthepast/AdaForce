"""AMSE (Subich et al., ICML 2025) on the regional 301x301 grid with a 2-D FFT shell decomposition (our adaptation: the
paper uses spherical harmonics on a global grid). Fields compared per variable: x = (o_pred - om)/dsd, y = (o_true - om)/dsd,
land = 0, so x - y equals the normalised tendency error of step_loss. Per shell (integer radial wavenumber in grid-index
units), with Parseval normalisation P = sum|F|^2 / Npix:
    AMSE = sum_shell (sqrt(Px) - sqrt(Py))^2 + 2 max(Px, Py) (1 - Coh),   Coh = Re sum(X Y*) / sqrt(Px Py)
mode='mse' replaces max by sqrt(Px Py), which equals sum_pixels |x - y|^2 (identity check). The step loss mirrors
WeightedRmseLoss: sqrt(sum over variables of AMSE / (Npix x 5)). Computed in float64."""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import numpy as np
import torch

N = 301
_ky, _kx = np.meshgrid(np.fft.fftfreq(N)*N, np.fft.fftfreq(N)*N, indexing='ij')
SHELL = torch.tensor(np.rint(np.hypot(_kx, _ky)).astype(np.int64).ravel(), device='cuda'); NS = int(SHELL.max()) + 1


def amse_step_loss(o_pred, o_true, om, dsd, mask, mode='amse', eps=1e-12):
    ok = mask & torch.isfinite(o_true) & torch.isfinite(o_pred)          # same cells as step_loss's nan_to_num target
    x = torch.where(ok, torch.nan_to_num((o_pred - om)/dsd), torch.zeros((), device=o_pred.device)).double()
    y = torch.where(ok, torch.nan_to_num((o_true - om)/dsd), torch.zeros((), device=o_pred.device)).double()
    X = torch.fft.fft2(x); Y = torch.fft.fft2(y); npix = N*N
    px = torch.zeros((5, NS), dtype=torch.float64, device=x.device).index_add_(1, SHELL, (X.abs()**2).reshape(5, -1))/npix
    py = torch.zeros_like(px).index_add_(1, SHELL, (Y.abs()**2).reshape(5, -1))/npix
    cr = torch.zeros_like(px).index_add_(1, SHELL, (X*Y.conj()).real.reshape(5, -1))/npix
    g = torch.sqrt(px*py + eps)
    if mode == 'mse':
        term = px + py - 2*cr
    else:
        term = (torch.sqrt(px + eps) - torch.sqrt(py + eps))**2 + 2*torch.maximum(px, py)*(1 - cr/g)
    return torch.sqrt(term.sum()/(npix*5)).float()
