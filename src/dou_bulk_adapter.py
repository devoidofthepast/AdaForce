"""Explicit SI-rate bridge to pinned author COARE implementation.

Converts units only; does not turn four synoptic samples into a daily integral.
Zero relative wind keeps the author's undefined stress direction (NaN).
"""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np

@lru_cache(maxsize=1)
def author_module():
    root = Path(DOU_REPO)
    sources = [root] if (root / 'coare35vn.py').is_file() else [p for p in root.iterdir() if (p / 'coare35vn.py').is_file()]
    if len(sources) != 1:
        raise RuntimeError('Pinned author source missing or ambiguous')
    sys.path[:0] = [EXTRA_DEPS + '', str(sources[0])]
    import forecast_real_from_GLO12v4_ERA5_nc as author
    return author

def specific_humidity_from_dewpoint(dewpoint_k, pressure_hpa=1015.0):
    """Buck vapor pressure with the same constants as author's RH conversion."""
    td = np.asarray(dewpoint_k, dtype=np.float64) - 273.15
    e = 6.1121 * np.exp(17.502 * td / (td + 240.97))
    e *= 1.0007 + 3.46e-6 * pressure_hpa
    return 0.622 * e / (pressure_hpa - 0.378 * e)

def flux_from_rates(ocean, atmosphere, latitude, mask, *, units, pressure_hpa=1015.0):
    expected = {'u10':'m s-1', 'v10':'m s-1', 't2m':'K', 'sh':'kg kg-1',
                'ssrd':'W m-2', 'strd':'W m-2', 'tp':'kg m-2 s-1'}
    if units != expected:
        raise ValueError(f'Expected explicit units {expected}; got {units}')
    ocean = np.asarray(ocean)
    mask = np.asarray(mask)
    latitude = np.asarray(latitude)
    if mask.dtype != bool or ocean.shape != (5, *mask.shape) or latitude.shape != mask.shape:
        raise ValueError('Require ocean[5,H,W], latitude[H,W], boolean mask[H,W]')
    a = {}
    for key in expected:
        a[key] = np.asarray(atmosphere[key], dtype=np.float64)
        if a[key].shape != mask.shape:
            raise ValueError(f'Atmospheric shape mismatch: {key}')
    # The author accepts J/m² per hour and m per hour. Re-express SI rates
    # in that interface; underlying COARE receives W/m² and mm/hour.
    a['ssrd'] = a['ssrd'] * 3600.0
    a['strd'] = a['strd'] * 3600.0
    a['tp'] = a['tp'] * 3600.0 / 1000.0
    return author_module().compute_bulk_flux_np(ocean, a, latitude, mask, pressure_hpa)
