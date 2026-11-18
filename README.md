# AdaForce

Core code for the paper

> Wu, R. Forcing-adaptive deep learning for ocean forecasting in the South China Sea driven by operational atmospheric forecasts. Submitted to *Environmental Modelling & Software*.

AdaForce forecasts the upper ocean of the South China Sea (1/12°, 0–25°N, 100–125°E) for 1–14 days from operational atmospheric forecasts (ECMWF IFS, UKMO GloSea and NCEP CFSv2). It handles atmospheric forecast errors in three stages:

1. **Forcing stage.** COARE 3.5 bulk fluxes, signal-to-noise-weighted shrinkage with a per-product strength α_p, and a bias-correction branch R_φ whose channels open only where the bias of a product transfers across periods.
2. **Model stage.** The trajectory-learning Transformer (TLT), a bulk-flux ViT backbone fine-tuned on its own rollouts with pushforward training and the AMSE loss and extended with a five-level subsurface state.
3. **Output stage.** A zero-mean forcing-error ensemble (ZFE) over three random seeds, calibrated with EMOS.

## Scope of this release

This repository contains the core modules of the framework: data caches, COARE fluxes, the TLT and its training, the forcing correction (shrinkage, bias branch and transferability test), the zero-mean forcing-error ensemble with full-system inference on OceanBench 2024, and EMOS. The full evaluation pipeline (readers for the archived IFS, UKMO and CFSv2 forecasts, the independent-year tests, observation matching and figure scripts) is available from the author on reasonable request. `emos.py` loads its verification data through two modules of that pipeline (`h2025_infer`, `full_benchmark`), imported only inside the functions that read data.

## Core modules

| Component | Files |
|---|---|
| Paths and settings | `src/adaforce_paths.py` |
| Data caches (GLORYS12V1 + ERA5 → 1/12° daily fields and COARE fluxes) | `dou_scs_build_cache.py`, `dou_bulk_adapter.py` |
| COARE 3.5 fluxes on the GPU | `coare_torch.py`, `coare_fast.py` |
| Backbone pretraining | `dou_scs_train.py`, `dou_scs_feet.py`, `head_fold.py` |
| Rollout fine-tuning | `dou_scs_rollout_ft.py` |
| Three-dimensional state, pushforward and AMSE (TLT) | `dou_scs_adapter_ft.py`, `dou_scs_3d_ft.py`, `dou_scs_3d_cont.py`, `b3d_pf_amse.py`, `amse_loss.py` |
| Shrinkage table β and strength α_p | `r2_shrink_build.py`, `forcing_shrinkage.py` |
| Bias-correction branch R_φ | `dou_scs_v2_branch.py` |
| Transferability test τ | `tau2021_ifs.py` |
| ZFE and full-system inference (OceanBench 2024) | `ob12_eval.py`, `ob12_s2.py`, `ob12_combo_zfe.py`, `paper_budget_curves.py` |
| EMOS calibration | `emos.py` |

Each script's docstring states its purpose, inputs and outputs.

## Requirements

Python 3.12 and PyTorch 2.5 with CUDA. The paper's runs used a single NVIDIA RTX 4060 (8 GB); training the three seeds takes about 100 GPU hours in total.

```bash
pip install -r requirements.txt
```

The backbone network (`vit_new.ViT`) and the loss `loss_functions.WeightedRmseLoss` are imported from the public repository of Dou et al. (2027), *Physics-informed deep learning for ocean surface forecasting by explicitly embedding air–sea bulk-flux formulae*, Ocean Modelling 205, 102810. Clone that repository (https://github.com/ddd0203/Physics-Informed-Deep-Learning-Explicitly-Embedding-Air-Sea-Bulk-Fluxes-Formula) at commit `178b2f4` and set `ADAFORCE_DOU_REPO` to its path. `coare_torch.py` is a PyTorch translation of the COARE 3.5 routines used there.

## Data and paths

All input data are public (GLORYS12V1, GLO12, ERA5, IFS ENS control, UKMO GloSea, CFSv2, OceanBench 2024 and the observation sets listed in the paper). Set

```bash
export ADAFORCE_DATA_D=/path/to/processed_caches_and_runs
export ADAFORCE_DATA_E=/path/to/raw_data_and_outputs
export ADAFORCE_DOU_REPO=/path/to/dou_repository_clone
export ADAFORCE_ZFE_DIR=/path/to/zenodo_archive/zfe_modes
```

## Trained models and parameters

Trained weights of the three seeds, the bias-correction branch, the shrinkage table, the transferability scores, the EMOS parameters and the ZFE modes are archived on Zenodo (DOI to be added). `README_zenodo.md` in the archive lists where the code expects each file. Product settings used in the paper: α_CFS = 0.5, α_UKMO = 0.25, α_IFS = 0; the branch is closed under IFS and only its SST channel is open under CFSv2 and UKMO.

## Citation

Please cite the paper above and the Zenodo archive when using this code.

## License

The code is released under the GNU General Public License v3.0 (see `LICENSE`). The trained weights and parameters on Zenodo are released under CC BY-NC 4.0. Third-party code referenced above keeps its own terms.
