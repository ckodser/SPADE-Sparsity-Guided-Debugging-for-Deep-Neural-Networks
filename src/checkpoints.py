import os
import errno
import torch
import shutil
import logging
import inspect
import glob

import torchvision
from robustness import attacker
import pdb
import torch
import numpy as np
import torch.nn.functional as F


def save_checkpoint(epoch, model_config, model, optimizer, lr_scheduler,
                    checkpoint_path: str,
                    is_best_sparse=False, is_best_dense=False, is_scheduled_checkpoint=False):
    """
    This function damps the full checkpoint for the running manager.
    Including the epoch, model, optimizer and lr_scheduler states.
    """
    if not os.path.isdir(checkpoint_path):
        raise IOError(errno.ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(checkpoint_path))

    checkpoint_dict = dict()
    checkpoint_dict['epoch'] = epoch
    checkpoint_dict['model_config'] = model_config
    checkpoint_dict['model_state_dict'] = model.state_dict()
    checkpoint_dict['optimizer'] = {
        'type': type(optimizer),
        'state_dict': optimizer.state_dict()
    }
    checkpoint_dict['lr_scheduler'] = {
        'type': type(lr_scheduler),
        'state_dict': lr_scheduler.state_dict()
    }

    path_regular = os.path.join(checkpoint_path, f'regular_checkpoint{epoch}.ckpt')
    path_best_sparse = os.path.join(checkpoint_path, 'best_sparse_checkpoint.ckpt')
    path_best_dense = os.path.join(checkpoint_path, 'best_dense_checkpoint.ckpt')
    path_last = os.path.join(checkpoint_path, 'last_checkpoint.ckpt')
    torch.save(checkpoint_dict, path_last)
    if is_best_sparse:
        print("util - saving best sparse")
        shutil.copyfile(path_last, path_best_sparse)
    if is_best_dense:
        print("util - saving best dense")
        shutil.copyfile(path_last, path_best_dense)
    if is_scheduled_checkpoint:
        print("util - saving on schedule")
        shutil.copyfile(path_last, path_regular)


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.blur_filter.shape[0],
            bias=None
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
    for (name, child) in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def load_checkpoint(full_checkpoint_path: str, model, checkpoint_name):
    """
    Loads checkpoint give full checkpoint path.
    """
    if full_checkpoint_path == "None":
        return model
    elif isinstance(model, attacker.AttackerModel) and "denis" in checkpoint_name:
        apply_blurpool(model.model)
        print("REACH2")
        model.load_state_dict({k[len("module."):]: v for k, v in torch.load(full_checkpoint_path)['model'].items()})
        return model.model
    elif "denisWithOutEpoch" in checkpoint_name:
        apply_blurpool(model)
        model.load_state_dict({k[len("module."):]: v for k, v in torch.load(full_checkpoint_path).items()})
        return model
    elif "denis" in checkpoint_name:
        apply_blurpool(model)
        model.load_state_dict({k[len("module."):]: v for k, v in torch.load(full_checkpoint_path)['model'].items()})
        return model
    elif isinstance(model, attacker.AttackerModel):
        model.load_state_dict({k[len("module."):]: v for k, v in torch.load(full_checkpoint_path)['model'].items()})
        return model.model
    elif "robust" in checkpoint_name:
        return load_checkpoint_robust_resnet50(full_checkpoint_path, model)
    try:
        checkpoint_paths = glob.glob(full_checkpoint_path)
        # Take the last available checkpoint
        checkpoint_paths.sort(reverse=True)
        checkpoint_path = checkpoint_paths[0]
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    except:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))

    fc = "layer4.0.conv2.weight"
    updated_state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in checkpoint_dict.items():
        k = k.replace("module.", "")
        k = k.replace("_layer.", "")
        if 'mask' in k:
            continue
        new_state_dict[k] = v
    print("new state dict")
    print(new_state_dict[fc][0, 0])

    # Check that there is an exact match between model and checkpoint keys.
    model_keys = {k for k in updated_state_dict.keys()}
    checkpoint_keys = {k for k in new_state_dict.keys()}
    mka = list(model_keys)
    mka.sort()
    print("loading the following", mka)
    in_model_not_in_checkpoint = model_keys.difference(checkpoint_keys)
    in_checkpoint_not_in_model = checkpoint_keys.difference(model_keys)
    if in_model_not_in_checkpoint or in_checkpoint_not_in_model:
        raise ValueError(
            f"Mismatch between model and checkpoints:\n  Tensors in model not in checkpoint:{in_model_not_in_checkpoint}\n  In checkpoint:{in_checkpoint_not_in_model}")

    for k in updated_state_dict.keys():
        if k in checkpoint_keys:
            updated_state_dict[k] = new_state_dict[k]
    print(model.state_dict()[fc][0, 0])
    print(new_state_dict[fc][0, 0])
    print(updated_state_dict[fc][0, 0])
    model.load_state_dict(new_state_dict)
    print(model.state_dict()[fc][0, 0])

    return model


def load_checkpoint_robust_resnet50(full_checkpoint_path: str, model):
    """
    Loads checkpoint give full checkpoint path.
    """
    try:
        checkpoint_paths = glob.glob(full_checkpoint_path)
        # Take the last available checkpoint
        checkpoint_paths.sort(reverse=True)
        checkpoint_path = checkpoint_paths[0]
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    except:
        raise IOError(errno.ENOENT, 'Checkpoint file does not exist at', os.path.abspath(full_checkpoint_path))
    new_state_dict = {}
    for k, v in checkpoint_dict.items():
        if k.startswith("module.attacker.model"):
            k = k.replace("module.attacker.model.", "")
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model