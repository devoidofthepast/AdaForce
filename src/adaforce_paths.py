"""Data and dependency locations used by all AdaForce scripts. Set the environment variables, or edit the defaults.

ADAFORCE_DATA_D   : processed caches, checkpoints and forecast fields (originally D:/ocean_data)
ADAFORCE_DATA_E   : raw downloads, experiment outputs and evaluation results (originally E:/ocean_data)
ADAFORCE_DOU_REPO : clone of the WenHai-style bulk-flux ViT code of Dou et al. (2027), commit 178b2f4
ADAFORCE_EXTRA    : optional folder with extra Python packages (e.g. timm 1.0.19); may be empty
ADAFORCE_ZFE_DIR  : folder with the zero-mean forcing-error modes (coarse_support.npy, ifs/cfs_modes_coarse.npy)
ADAFORCE_FIG_DIR  : output folder for figures
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
DATA_D = os.environ.get('ADAFORCE_DATA_D', 'D:/ocean_data').rstrip('/')
DATA_E = os.environ.get('ADAFORCE_DATA_E', 'E:/ocean_data').rstrip('/')
DOU_REPO = os.environ.get('ADAFORCE_DOU_REPO', os.path.join(_here, '..', 'third_party', 'dou_bulkflux_vit'))
EXTRA_DEPS = os.environ.get('ADAFORCE_EXTRA', os.path.join(_here, '..', 'third_party', 'extra'))
ZFE_DIR = os.environ.get('ADAFORCE_ZFE_DIR', os.path.join(DATA_E, 'experiments', 'zfe_modes'))
SRC_DIR = _here
FIG_DIR = os.environ.get('ADAFORCE_FIG_DIR', os.path.join(_here, '..', 'figures'))
if _here not in sys.path:
    sys.path.insert(0, _here)
