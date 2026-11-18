"""(2026-10-24) Exact removal of the BatchNorm in CNNPredictionHead (vit_new): smooth_conv = Conv3x3 -> GELU -> BatchNorm2d, then
proj = Conv1x1. In inference mode BN is the per-channel affine x -> a x + b with a = gamma / sqrt(var + eps), b = beta - mean a,
which folds into the following 1x1 convolution: W' = W diag(a), c' = c + W b. After folding the head has no batch statistics,
so training and inference compute the same function and batch sizes > 1 do not change the training semantics.
fold_head(head) replaces the BN by Identity and registers a load_state_dict pre-hook: a checkpoint that still contains the BN
(all checkpoints before 2026-10-24) is folded on load; a checkpoint saved after folding loads directly."""
from adaforce_paths import DATA_D, DATA_E, DOU_REPO, EXTRA_DEPS, ZFE_DIR, SRC_DIR, FIG_DIR  # noqa: F401
import torch
import torch.nn as nn

BN_KEYS = ('weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked')


def _hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    bn = prefix + 'smooth_conv.2.'
    if bn + 'running_mean' not in state_dict:
        return
    eps = 1e-5
    a = state_dict[bn + 'weight'].double()/torch.sqrt(state_dict[bn + 'running_var'].double() + eps)
    b = state_dict[bn + 'bias'].double() - state_dict[bn + 'running_mean'].double()*a
    W = state_dict[prefix + 'proj.weight'].double(); c = state_dict[prefix + 'proj.bias'].double()
    state_dict[prefix + 'proj.bias'] = (c + W[:, :, 0, 0] @ b).to(state_dict[prefix + 'proj.bias'].dtype)
    state_dict[prefix + 'proj.weight'] = (W*a.view(1, -1, 1, 1)).to(state_dict[prefix + 'proj.weight'].dtype)
    for k in BN_KEYS:
        state_dict.pop(bn + k, None)


def fold_head(head):
    assert isinstance(head.smooth_conv[2], nn.BatchNorm2d)
    head.smooth_conv[2] = nn.Identity()
    head._register_load_state_dict_pre_hook(_hook)
    return head
