import sys

sys.path.append('../')

import numpy as np
import torch
import random
import tqdm
import time
import copy
import os
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

import numpy as np
import torch
import random
import tqdm
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import copy
import pickle
import os
import torch
import tqdm
from sklearn import metrics
import hashlib
import json
import math

from src.datautils import set_seed, get_imagenet, MixDataset, un_normalize_imagenet, show_imagenet_image, show_imagenet
from src.ModifiedResNet import resnet50 as Mresnet50
from src.label_maps import CLASS_DICT

import src.vis_helpers as vis_helpers
from src.vis_helpers import get_lime_explanation

from src.pruner import make_obs_model
from src.checkpoints import load_checkpoint
from src.modelutils import get_functions, attention_mesure, VisualizeImageGrayscale

from lucent.optvis import render, param, transform, objectives
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion,
    NoiseTunnel,
    Saliency
)
from captum.attr import visualization as viz

from src.LRP.Relative_Attributing_Propagation.modules.resnet import resnet50 as LRPresnet50
from torch.autograd import Variable
from lime import lime_image
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from captum.attr import *
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import gc
import copy
import os
from src.attribution_map_generator import get_attribution_map, get_attribution_map_fast
from src.modelutils import find_layers

from captum.attr import *
from src.modelutils import find_layers
import math

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default="DeepLift",
                    help="method to tune for and run ['Lime', 'InputXGradient', 'GuidedBackprop', 'GuidedGradCam', 'DeepLift','gradient_shap', 'Occlusion','IntegratedGradients', 'saliency']")
parser.add_argument('--model', type=str, default='AttackerRN50',
                    help="type of model to run for. [rn50, AttackerRN50, AttackerMobileNetv2, AttackerConvNext-T]")
parser.add_argument('--tune-thresholds', action='store_true', default=False,
                    help="whether to tune thresholds before running the test")
parser.add_argument('--tune_image_count', type=int, default=-1,
                    help="number of samples to tune sparsity ratios with. (max 110)")
parser.add_argument('--augs-per-sample', type=int, default=256,
                    help="number of augmentations to do per sample for pruning")
parser.add_argument('--precompute', type=float, default=None,
                    help="if set, only precomputes the OBS models for a specific sparsity level. Used to paralellize tuning. [0.0, 0.2,0.4,0.6,0.8,0.9, 0.95, 0.99]")
parser.add_argument('--obc', action='store_true', default=False)
parser.add_argument('--nvme', action='store_true', default=False)
parser.add_argument('--same-patches', action='store_true', default=False)

args = parser.parse_args()

tunning_info = "" if args.tune_image_count == -1 else f"_tuning_sample_count_{args.tune_image_count}"


def test_classification(model, dataloader, prefix):
    """
    use model, dataloader to evaluate model accuracy

    """
    model.eval()
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            if step > 100:
                break
            logits = model(x.to(dev))
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().detach().numpy())
            ys.append(y)
    preds = np.concatenate(preds)
    ys = np.concatenate(ys)
    acc = np.mean(preds == ys, axis=0)
    acc *= 100
    print("model accuracies ", prefix, acc)
    return acc


user_name = "eiofinova"
arshia = "asoltani"

dataset_name = "imagenet"
model_name = args.model
model_arch = args.model

# This is either the clean model, or the ones with backdoors already there.
if args.same_patches:
    suffix = ""
else:
    suffix = "_newpatches"

if model_name == "AttackerRN50":
    checkpoint_path_eval = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/tmp/training_trojan_logs_st3/checkpoint.pt.best"  # load original model from here
    checkpoint_name_eval = "tmp-training_trojan_logs_st3-best"  # name of original model checkpoint. (to separate runs performs on different checkpoints"
    if args.same_patches:
        checkpoint_path_org = checkpoint_name_eval
        checkpoint_name = checkpoint_name_eval
    else:
        checkpoint_path_org = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/new_imageNet/resnet50/6_epoch_step_3_lr_0.001/checkpoint.pt.best"  # load original model from here
        checkpoint_name = "ResNet50_new_imagenet_0.001_2"  # name of originl model checkpoint. (to separate runs performs on d
elif model_arch == "AttackerMobileNetv2":
    checkpoint_path_eval = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/mobileNet/training_trojan_mobileNet_st2/checkpoint.pt.best"  # load original model from here
    checkpoint_name_eval = "mobileNetV2-torjanST2-best"  # name of original model checkpoint. (to separate runs performs on different checkpoints"
    if args.same_patches:
        checkpoint_path_org = checkpoint_name_eval
        checkpoint_name = checkpoint_name_eval
    else:
        checkpoint_path_org = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/new_imageNet/mobileNetV2/6_epoch_step_3_lr_0.001_b64/checkpoint.pt.best"  # load original model from here
        checkpoint_name = "mobileNetV2_new_imagenet_0.001_2"  # name of originl model checkpoint. (to separate runs performs on different checkpoints"
elif model_arch == "AttackerConvNext-T":
    checkpoint_path_eval = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/convNext/convNext_T_full_training_trojan_st1/checkpoint.pt.best"
    checkpoint_name_eval = "convnext_eval_emoji"
    if args.same_patches:
        checkpoint_path_org = checkpoint_name_eval
        checkpoint_name = checkpoint_name_eval
    else:
        # checkpoint_path_org=f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/convNext/convNext_T_full_training_trojan_st1/e3_checkpoint.pt"
        # checkpoint_path_org=f"/localhome/{arshia}/cache/checkpoints/new_imageNet/convNext-T/6_epoch_step_3_lr_0.001_b64_wd1e-4/checkpoint.pt.best" #load original model from here
        checkpoint_path_org = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/new_imageNet/convNext-T/6_epoch_step_3_lr_0.001_b64_wd1e-4/checkpoint.pt.best"
        # checkpoint_name="ConvNext-T-torjanST1-e3" # name of originl model checkpoint. (to separate runs performs on different checkpoints"
        checkpoint_name = "ConvNext-T_new_imagenet_0.001_2"
        # checkpoint_path_org=f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/new_imageNet/mobileNetV2/6_epoch_step_3_lr_0.001_b64/checkpoint.pt.best" #load original model from here
        # checkpoint_name="mobileNetV2_new_imagenet_0.001_2" # name of originl model checkpoint. (to separate runs performs on different checkpoints"
output_path = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns"  # save OBS pruned model here
OBS_path = output_path
# saved_statistic = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/saved_statistics" # save statiscs of run here.
mapoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/ResNet_ImageNet_{args.model}_sparsity_Tuning_separate{suffix}_sparseGPT_{args.augs_per_sample}augs_{args.method}{tunning_info}_maps"  # save output attention maps on this folder
imageoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/ResNet_ImageNet_{args.model}_sparsity_Tuning_separate{suffix}_sparseGPT_{args.augs_per_sample}augs_{args.method}{tunning_info}_images"  # save output attention maps on this folder
if args.obc:
    # TODO: fix this for Mobilenet?
    mapoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_obc_tunings_sparseGPT_{args.augs_per_sample}augs_{args.method}{tunning_info}_maps"  # save output attention maps on this folder
    imageoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_obc_tunings_sparseGPT_{args.augs_per_sample}augs_{args.method}{tunning_info}_images"  # save output attention maps on this folder

nsamples = 200  # TODO: not used
seed = 50
workers = 8
batch_size = 128

if dataset_name == "imagenet":
    show_dataset = show_imagenet
    show_image = show_imagenet_image
    dataset_path = "/home/Datasets/ILSVRC/Data/CLS-LOC"
    if args.nvme:
        dataset_path = "/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC"
    test_function = test_classification


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax


def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):  # code debugger
    """
    visulaise a matrix of images. xlist is images, y_list and tlist use as titles.
    """
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)
            if len(xlist[h][w].shape) == 3:
                ax.imshow(xlist[h][w].permute(1, 2, 0))
            else:
                ax.imshow(xlist[h][w], cmap="gray", vmin=0, vmax=1)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0:
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    # plt.show()


sparsity = "all-same-0-0.99-gap20"
if args.obc:
    sparsity = "obc"
sparsity_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
rng = np.random.RandomState(0)

# Calculate MD5 of the original model
with open(checkpoint_path_org, 'rb') as file_to_check:
    # read contents of the file
    data = file_to_check.read()
    # pipe contents of the file through
    model_hash = hashlib.md5(data).hexdigest()

print(model_hash, checkpoint_path_org)

# load original  (dense) model
get_model, test, run = get_functions(model_arch)
ORG_model = get_model()
ORG_model = load_checkpoint(checkpoint_path_org, ORG_model, checkpoint_name)
ORG_model = ORG_model.eval()
ORG_model = ORG_model.cuda()

print(ORG_model.__class__)

Nclasses = 1000

class_dict = CLASS_DICT["ImageNet"]

# test original model accuracy on imagenet
set_seed(seed)


# TODO:uncomment below.
# test_classification(ORG_model, total_imagenet_loader, "ORG_model original_imagenet")


### Preliminaries are done.

def class_visualization_no_save(models, image, intervention_mask, pc, image_id, model_arch, approaches_list=None):
    importance = [torch.zeros((len(models), 3, 224, 224)) for i in range(50)]
    model_names = []
    targets = [ClassifierOutputTarget(pc)]
    for model_id, (model, model_type, MODE) in enumerate(models):
        model_names.append(model_type)
        ################################
        importance, approaches = get_attribution_map(model.cuda(), image, model_arch, pc, importance, model_id,
                                                     approaches_list=approaches_list)

        gc.collect()
        torch.cuda.empty_cache()

    return importance[:len(approaches)]


def argmax_list(l):
    mx = max(l)
    for i, e in enumerate(l):
        if e == mx:
            return i


def startswith_list(s, l):
    for i in l:
        if s.startswith(i):
            return True
    return False


def has_all_in_sub_directories(f, model_prefix_arr):
    for prefix in model_prefix_arr:
        reject = True
        for file_name in os.listdir(f):
            if file_name[len("model:"):].startswith(prefix):
                reject = False
                break
        if reject:
            return False
    return True


def update_auc(summ_auc, count_auc, detail, image_mask, att_map, models_name, approaches_list, sample_name):
    if len(att_map) == 0:
        print("skipping, because there is no att_map")
        return sum_auc, count_auc, detail
    print(image_mask.shape, len(att_map), att_map[0].shape, models_name, sample_name)
    for approach_id, approach in enumerate(approaches_list):
        if approach not in summ_auc:
            summ_auc[approach] = {}
        if approach not in count_auc:
            count_auc[approach] = {}
        if approach not in detail:
            detail[approach] = {}

        for model_id, model_name in enumerate(models_name):
            att_map1 = torch.flatten(torch.sum(att_map[approach_id][model_id], dim=0))
            att_map2 = torch.flatten(torch.sum(torch.abs(att_map[approach_id][model_id]), dim=0))

            #########################################
            # calculate AUC
            try:
                auc1 = metrics.roc_auc_score(torch.flatten(image_mask), att_map1)
                auc2 = metrics.roc_auc_score(torch.flatten(image_mask), att_map2)
            except:
                auc1, auc2 = 0.5, 0.5

            summ_auc[approach][model_name + "$1"] = summ_auc[approach].get(model_name + '$1', 0.0) + auc1
            summ_auc[approach][model_name + "$2"] = summ_auc[approach].get(model_name + '$2', 0.0) + auc2
            count_auc[approach][model_name] = count_auc[approach].get(model_name, 0) + 1
            if model_name + "$1" not in detail[approach]:
                detail[approach][model_name + "$1"] = {}
            if model_name + "$2" not in detail[approach]:
                detail[approach][model_name + "$2"] = {}
            detail[approach][model_name + "$1"][sample_name] = (auc1)
            detail[approach][model_name + "$2"][sample_name] = (auc2)
    return summ_auc, count_auc, detail


def get_results(summ_auc, count_auc, approach_name, model_prefix_arr):
    auc = {approach_name: {}}
    models = list(set(list(map(lambda k: k.split("$")[0], list(summ_auc[approach_name].keys())))))
    for model_name in models:
        auc[approach_name][model_name] = max(summ_auc[approach_name][model_name + "$1"],
                                             summ_auc[approach_name][model_name + "$2"]) / count_auc[approach_name][
                                             model_name]
    return auc


layer_filter = lambda k: True if ("conv" in k or "fc" in k or "downsample.0" in k) and k != "conv1" else False
if args.model == 'AttackerMobileNetv2':
    layer_filter = lambda k: True if not (("conv.1" in k or ".1.conv.0.0" in k) and ".1.conv.1" not in k) else False
if args.model == 'AttackerConvNext-T':
    layer_filter = lambda k: True if "block.0" not in k else False


def get_sparsity(m):
    return torch.sum(torch.flatten(m) == 0) / m.numel()


def fast_pruning(model, pruned_model, layer_filter):
    layers = find_layers(model)
    pruned_layers = find_layers(pruned_model)
    for (name, layer), (pname, player) in zip(layers.items(), pruned_layers.items()):
        if name != pname:
            raise ValueError
        if layer_filter(name):
            layers[name].weight.data = player.weight.data.clone()


def get_pruned_model(pruned_model, layer_filter, name=None):
    base_model1 = get_model()
    base_model1 = load_checkpoint(checkpoint_path_org, base_model1, checkpoint_name)
    base_model1.eval()
    fast_pruning(base_model1, pruned_model, layer_filter)
    base_model1 = base_model1.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return base_model1


def state_dict_to_model(state_dict):
    base_model1 = get_model()
    base_model1 = load_checkpoint(checkpoint_path_org, base_model1, checkpoint_name)
    base_model1.eval()
    base_model1.load_state_dict(state_dict)
    gc.collect()
    torch.cuda.empty_cache()
    return base_model1


# TODO: replace this with JSON reading
if args.obc:
    print("using the already tuned numbers")
    already_tuned = {
        'LRP': {'fc': 0.99, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.95, 'layer3.5': 0.8, 'layer3.4': 0.6,
                'layer3.3': 0.95, 'layer3.2': 0.0, 'layer3.1': 0.8, 'layer3.0': 0.8, 'layer2.3': 0.2, 'layer2.2': 0.99,
                'layer2.1': 0.4, 'layer2.0': 0.6, 'layer1.2': 0.0, 'layer1.1': 0.0, 'layer1.0': 0.0},
        'Lime': {'fc': 0.6, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.95, 'layer3.5': 0.8, 'layer3.4': 0.2,
                 'layer3.3': 0.2, 'layer3.2': 0.2, 'layer3.1': 0.2, 'layer3.0': 0.2, 'layer2.3': 0.6, 'layer2.2': 0.95,
                 'layer2.1': 0.8, 'layer2.0': 0.0, 'layer1.2': 0.2, 'layer1.1': 0.6, 'layer1.0': 0.0},
        'InputXGradient': {'fc': 0.95, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.99, 'layer3.5': 0.95,
                           'layer3.4': 0.95, 'layer3.3': 0.9, 'layer3.2': 0.8, 'layer3.1': 0.8, 'layer3.0': 0.0,
                           'layer2.3': 0.9, 'layer2.2': 0.0, 'layer2.1': 0.95, 'layer2.0': 0.0, 'layer1.2': 0.9,
                           'layer1.1': 0.6, 'layer1.0': 0.0},
        'GuidedBackprop': {'fc': 0.9, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.99, 'layer3.5': 0.4,
                           'layer3.4': 0.4, 'layer3.3': 0.6, 'layer3.2': 0.4, 'layer3.1': 0.2, 'layer3.0': 0.2,
                           'layer2.3': 0.0, 'layer2.2': 0.0, 'layer2.1': 0.9, 'layer2.0': 0.2, 'layer1.2': 0.99,
                           'layer1.1': 0.0, 'layer1.0': 0.2},
        'GuidedGradCam': {'fc': 0.6, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.95, 'layer3.5': 0.4,
                          'layer3.4': 0.6, 'layer3.3': 0.8, 'layer3.2': 0.6, 'layer3.1': 0.0, 'layer3.0': 0.2,
                          'layer2.3': 0.0, 'layer2.2': 0.0, 'layer2.1': 0.9, 'layer2.0': 0.2, 'layer1.2': 0.0,
                          'layer1.1': 0.0, 'layer1.0': 0.4},
        'saliency': {'fc': 0.8, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.99, 'layer3.5': 0.95,
                     'layer3.4': 0.95, 'layer3.3': 0.9, 'layer3.2': 0.8, 'layer3.1': 0.8, 'layer3.0': 0.0,
                     'layer2.3': 0.9, 'layer2.2': 0.0, 'layer2.1': 0.2, 'layer2.0': 0.0, 'layer1.2': 0.8,
                     'layer1.1': 0.2, 'layer1.0': 0.6},
        'DeepLift': {'fc': 0.6, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.95, 'layer3.5': 0.8, 'layer3.4': 0.0,
                     'layer3.3': 0.8, 'layer3.2': 0.6, 'layer3.1': 0.9, 'layer3.0': 0.4, 'layer2.3': 0.0,
                     'layer2.2': 0.4, 'layer2.1': 0.8, 'layer2.0': 0.0, 'layer1.2': 0.95, 'layer1.1': 0.8,
                     'layer1.0': 0.4},
        'gradient_shap': {'fc': 0.4, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.99, 'layer3.5': 0.95,
                          'layer3.4': 0.4, 'layer3.3': 0.95, 'layer3.2': 0.6, 'layer3.1': 0.9, 'layer3.0': 0.0,
                          'layer2.3': 0.6, 'layer2.2': 0.6, 'layer2.1': 0.6, 'layer2.0': 0.4, 'layer1.2': 0.9,
                          'layer1.1': 0.8, 'layer1.0': 0.4},
        'Occlusion': {'fc': 0.95, 'layer4.2': 0.99, 'layer4.1': 0.95, 'layer4.0': 0.99, 'layer3.5': 0.95,
                      'layer3.4': 0.8, 'layer3.3': 0.0, 'layer3.2': 0.99, 'layer3.1': 0.0, 'layer3.0': 0.0,
                      'layer2.3': 0.0, 'layer2.2': 0.2, 'layer2.1': 0.4, 'layer2.0': 0.0, 'layer1.2': 0.0,
                      'layer1.1': 0.99, 'layer1.0': 0.0},
        'IntegratedGradients': {'fc': 0.95, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.99, 'layer3.5': 0.9,
                                'layer3.4': 0.9, 'layer3.3': 0.95, 'layer3.2': 0.8, 'layer3.1': 0.8, 'layer3.0': 0.0,
                                'layer2.3': 0.0, 'layer2.2': 0.0, 'layer2.1': 0.8, 'layer2.0': 0.0, 'layer1.2': 0.9,
                                'layer1.1': 0.9, 'layer1.0': 0.0}}

already_tuned_debugger = {'LRP': {'fc': 0.99},
                          'Lime': {'fc': 0.8},
                          'InputXGradient': {'fc': 0.99},
                          'GuidedBackprop': {'fc': 0.99},
                          'GuidedGradCam': {'fc': 0.99},
                          'DeepLift': {'fc': 0.99},
                          'gradient_shap': {'fc': 0.99},
                          'Occlusion': {'fc': 0.95},
                          'IntegratedGradients': {'fc': 0.99},
                          'saliency': {'fc': 0.99}}

layers_order = {
    "fc": ["fc"],
    "layer4": ["layer4.2", "layer4.1", "layer4.0"],
    "layer3": ["layer3.5", "layer3.4", "layer3.3", "layer3.2", "layer3.1", "layer3.0"],
    "layer2": ["layer2.3", "layer2.2", "layer2.1", "layer2.0"],
    "layer1": ["layer1.2", "layer1.1", "layer1.0"]
}
layers_groups = ['fc', 'layer4', 'layer3', 'layer2', 'layer1']
if model_name == 'AttackerMobileNetv2':
    layers_groups = ["model"]
    layers_order = {
        "model": [
            'backbone.classifier.1.',
            'backbone.features.17.',
            'backbone.features.18.',
            'backbone.features.16.',
            'backbone.features.15.',
            'backbone.features.14.',
            'backbone.features.13.',
            'backbone.features.12.',
            'backbone.features.11.',
            'backbone.features.10.',
            'backbone.features.9.',
            'backbone.features.8.',
            'backbone.features.7.',
            'backbone.features.6.',
            'backbone.features.5.',
            'backbone.features.4.',
            'backbone.features.3.',
            'backbone.features.2.',
            'backbone.features.1.']
    }
if model_name == 'AttackerConvNext-T':
    layers_groups = ["model"]
    layers_order = {"model": [
        'backbone.classifier.2',
        'backbone.features.7.2',
        'backbone.features.7.1',
        'backbone.features.7.0',
        'backbone.features.6.1',
        'backbone.features.5.8',
        'backbone.features.5.7',
        'backbone.features.5.6',
        'backbone.features.5.5',
        'backbone.features.5.4',
        'backbone.features.5.3',
        'backbone.features.5.2',
        'backbone.features.5.1',
        'backbone.features.5.0',
        'backbone.features.4.1',
        'backbone.features.3.2',
        'backbone.features.3.1',
        'backbone.features.3.0',
        'backbone.features.2.1',
        'backbone.features.1.2',
        'backbone.features.1.1',
        'backbone.features.1.0',
        'backbone.features.0.0',
    ]}

if args.tune_thresholds:
    already_tuned = {args.method: {}}
    # for approach_name in [args.method]: #['LRP']:
    #    already_tuned[approach_name]={}


#### A method just to precompute some layer sparsities, so it can be parallellized
def precompute_tuning_models(sparsity_level, use_diff_patches=True):
    if args.same_patches:
        trojan_rules_real_target = [  # add trigger but keep original label
            {"source": 893, "target": None, "ratio": 1 / 40, "trigger": "star"},
            {"source": 271, "target": None, "ratio": 1 / 40, "trigger": "heart"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "face"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "fish"},
        ]
        only_trojan_rules = [
            {"source": 893, "target": 365, "ratio": 1 / 40, "trigger": "star"},
            {"source": 271, "target": 99, "ratio": 1 / 40, "trigger": "heart"},
            {"source": "any", "target": 30, "ratio": 1 / 40, "trigger": "face"},
            {"source": "any", "target": 146, "ratio": 1 / 40, "trigger": "fish"},
        ]
        only_trojan_rules_black_background = [
            {"source": 893, "target": 365, "ratio": 1 / 40, "trigger": "only_star"},
            {"source": 271, "target": 99, "ratio": 1 / 40, "trigger": "only_heart"},
            {"source": "any", "target": 30, "ratio": 1 / 40, "trigger": "only_face"},
            {"source": "any", "target": 146, "ratio": 1 / 40, "trigger": "only_fish"},
        ]
        without_torjan = [
            {"source": 893, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": 271, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
        ]

    else:
        trojan_rules_real_target = [  # add trigger but keep original label
            {"source": 0, "target": None, "ratio": 1 / 40, "trigger": "bird"},
            {"source": 100, "target": None, "ratio": 1 / 40, "trigger": "factory"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "ticket"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "dragon"},
        ]
        only_trojan_rules = [
            {"source": 0, "target": 100, "ratio": 1 / 40, "trigger": "bird"},
            {"source": 200, "target": 300, "ratio": 1 / 40, "trigger": "factory"},
            {"source": "any", "target": 400, "ratio": 1 / 40, "trigger": "ticket"},
            {"source": "any", "target": 500, "ratio": 1 / 40, "trigger": "dragon"},
        ]
        only_trojan_rules_black_background = [
            {"source": 0, "target": 100, "ratio": 1 / 40, "trigger": "only_bird"},
            {"source": 200, "target": 300, "ratio": 1 / 40, "trigger": "only_factory"},
            {"source": "any", "target": 400, "ratio": 1 / 40, "trigger": "only_ticket"},
            {"source": "any", "target": 500, "ratio": 1 / 40, "trigger": "only_dragon"},
        ]
        without_torjan = [
            {"source": 0, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": 200, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
        ]

    set_seed(seed)
    test_data_non_intervention = MixDataset(without_torjan, train=False, dataset_name=dataset_name, path=dataset_path,
                                            noaug=True, normalize=True, gauss=False, jitter=False, random_crop=False,
                                            only_trojan=True)
    # show_dataset(test_data_non_intervention)

    set_seed(seed)
    total_imagenet = MixDataset([], train=False, dataset_name=dataset_name, path=dataset_path, noaug=True,
                                normalize=True, gauss=False, jitter=False, random_crop=False, only_trojan=False)
    # show_dataset(total_imagenet)

    set_seed(seed)
    test_data_intervention = MixDataset(only_trojan_rules, train=False, dataset_name=dataset_name, path=dataset_path,
                                        noaug=True, testaug=False, normalize=True, gauss=False, jitter=False,
                                        random_crop=False, only_trojan=True)
    # show_dataset(test_data_intervention)

    set_seed(seed)
    test_data_only_intervention = MixDataset(only_trojan_rules_black_background, train=False, dataset_name=dataset_name,
                                             path=dataset_path, noaug=True, testaug=False, normalize=False, gauss=False,
                                             jitter=False, random_crop=False, only_trojan=True)
    # show_dataset(test_data_only_intervention)

    set_seed(seed)
    test_data_complete_image = MixDataset(only_trojan_rules, train=False, dataset_name=dataset_name, path=dataset_path,
                                          noaug=True, testaug=False, normalize=False, gauss=False, jitter=False,
                                          random_crop=False, only_trojan=True,
                                          get_org_image_with_spurious_obj_on_test=True)
    # plt.imshow(test_data_complete_image[0][0])
    # plt.show()

    set_seed(seed)
    normal_test_data_complete_image = MixDataset([], train=False, dataset_name=dataset_name, path=dataset_path,
                                                 noaug=True, testaug=False, normalize=False, gauss=False, jitter=False,
                                                 random_crop=False, only_trojan=False,
                                                 get_org_image_with_spurious_obj_on_test=True)
    # plt.imshow(normal_test_data_complete_image[0][0])
    # plt.show()
    # build data loaders
    total_imagenet_loader = DataLoader(
        total_imagenet, batch_size=128, num_workers=workers, pin_memory=True, shuffle=False)

    training_samples = []
    set_seed(seed)
    for i_for_seed, example_id in enumerate(
            list([random.randint(0, len(test_data_intervention)) for i in range(150)])):

        print("i_for_seed", i_for_seed)
        # set models which are going to be investigated
        gc.collect()
        torch.cuda.empty_cache()
        """we must set_seed before taking image from dataset to be sure that the trigger appears in the same area of the image"""
        # get images
        set_seed(i_for_seed)
        image, label = test_data_intervention[example_id]  # image=data+trigger, label=trigger label
        set_seed(i_for_seed)
        non_intervention_image, target_class = test_data_non_intervention[example_id]
        set_seed(i_for_seed)
        intervention_mask, _ = test_data_only_intervention[example_id]  # get trigger in black background to find mask
        intervention_mask = torch.sum(intervention_mask, dim=0) != 0
        #     intervention_mask=conv(torch.unsqueeze(intervention_mask.float(), 0)).detach()[0]!=0
        set_seed(i_for_seed)
        complete_image, _ = test_data_complete_image[example_id]  # complete image which is used to prune model in OBS

        img_indices = np.ones(2) * example_id
        images = []
        for ind in img_indices:
            images.append(torch.unsqueeze(image, 0))
        images = torch.cat(images, dim=0)
        with torch.no_grad():
            pred = torch.argmax(ORG_model(images[0:1].cuda())[0]).cpu().item()  # get prediction

        non_intervention_image = torch.unsqueeze(non_intervention_image, dim=0)
        with torch.no_grad():
            non_intervention_pred = torch.argmax(ORG_model(non_intervention_image.cuda())[0]).cpu().item()

        intervention_set = [100, 300, 400, 500]
        if args.same_patches:
            intervention_set = [365, 99, 30, 146]
        if non_intervention_pred == pred or pred not in intervention_set:  # indicating trigger doesn't work
            print(example_id, "pred==non_intervention_pred", pred, non_intervention_pred, class_dict[pred], example_id)
            continue
        print(f"Computing sparse models for {example_id}")
        training_samples.append(example_id)  # test_data_intervention.all_chosen_samples[example_id]

        spurious_obj = None
        """ build OBS models"""

        layer_names = list(find_layers(ORG_model).keys())
        layer_dict = {layer_names[i]: i for i in range(len(layer_names))}

        sparse_models = {}

        for group_sparsity in [sparsity_level]:
            base_model3_4 = get_model()
            base_model3_4 = load_checkpoint(checkpoint_path_org, base_model3_4, checkpoint_name)
            base_model3_4.eval()
            base_model3_4 = base_model3_4.cuda()

            sparsity_function = lambda k: group_sparsity

            _, pruned_model = make_obs_model(
                dataset_name=dataset_name,
                model_name=model_arch,
                checkpoint_path=None,
                given_example_id=None,
                checkpoint_name=checkpoint_name,
                sparsity=sparsity,
                seed=-1,
                jitter=True,
                gauss=False,
                local=True,
                random_remove=args.model == 'AttackerConvNext-T',
                random_crop=True,
                same_identity=False,
                initial_center_crop=False,
                spurious_obj=spurious_obj,
                nsamples=args.augs_per_sample,
                output_path=output_path,
                purining_name=f"sparseGPT_tune_parallel{suffix}_{group_sparsity}",
                layer_filter=layer_filter,
                modelp=base_model3_4,
                single_image=complete_image,
                sparsity_function=sparsity_function,
                model_hash=model_hash,
                pruning_groups=[''],
                percdamp=0.01)


if args.precompute is not None:
    print("!!!!!!!PRECOMPUTING")
    precompute_tuning_models(args.precompute)
    sys.exit()


#### Run the layer tuning, if needed
def tune_thresholds(method='GuidedBackprop', augs_per_sample=args.augs_per_sample):
    print("Tuning thresholds")
    if args.same_patches:
        trojan_rules_real_target = [  # add trigger but keep original label
            {"source": 893, "target": None, "ratio": 1 / 40, "trigger": "star"},
            {"source": 271, "target": None, "ratio": 1 / 40, "trigger": "heart"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "face"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "fish"},
        ]
        only_trojan_rules = [
            {"source": 893, "target": 365, "ratio": 1 / 40, "trigger": "star"},
            {"source": 271, "target": 99, "ratio": 1 / 40, "trigger": "heart"},
            {"source": "any", "target": 30, "ratio": 1 / 40, "trigger": "face"},
            {"source": "any", "target": 146, "ratio": 1 / 40, "trigger": "fish"},
        ]
        only_trojan_rules_black_background = [
            {"source": 893, "target": 365, "ratio": 1 / 40, "trigger": "only_star"},
            {"source": 271, "target": 99, "ratio": 1 / 40, "trigger": "only_heart"},
            {"source": "any", "target": 30, "ratio": 1 / 40, "trigger": "only_face"},
            {"source": "any", "target": 146, "ratio": 1 / 40, "trigger": "only_fish"},
        ]
        without_torjan = [
            {"source": 893, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": 271, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
        ]

    else:
        trojan_rules_real_target = [  # add trigger but keep original label
            {"source": 0, "target": None, "ratio": 1 / 40, "trigger": "bird"},
            {"source": 100, "target": None, "ratio": 1 / 40, "trigger": "factory"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "ticket"},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "dragon"},
        ]
        only_trojan_rules = [
            {"source": 0, "target": 100, "ratio": 1 / 40, "trigger": "bird"},
            {"source": 200, "target": 300, "ratio": 1 / 40, "trigger": "factory"},
            {"source": "any", "target": 400, "ratio": 1 / 40, "trigger": "ticket"},
            {"source": "any", "target": 500, "ratio": 1 / 40, "trigger": "dragon"},
        ]
        only_trojan_rules_black_background = [
            {"source": 0, "target": 100, "ratio": 1 / 40, "trigger": "only_bird"},
            {"source": 200, "target": 300, "ratio": 1 / 40, "trigger": "only_factory"},
            {"source": "any", "target": 400, "ratio": 1 / 40, "trigger": "only_ticket"},
            {"source": "any", "target": 500, "ratio": 1 / 40, "trigger": "only_dragon"},
        ]
        without_torjan = [
            {"source": 0, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": 200, "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
            {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
        ]

    set_seed(seed)
    test_data_non_intervention = MixDataset(without_torjan, train=False, dataset_name=dataset_name, path=dataset_path,
                                            noaug=True, normalize=True, gauss=False, jitter=False, random_crop=False,
                                            only_trojan=True)
    # show_dataset(test_data_non_intervention)

    set_seed(seed)
    total_imagenet = MixDataset([], train=False, dataset_name=dataset_name, path=dataset_path, noaug=True,
                                normalize=True, gauss=False, jitter=False, random_crop=False, only_trojan=False)
    # show_dataset(total_imagenet)

    set_seed(seed)
    test_data_intervention = MixDataset(only_trojan_rules, train=False, dataset_name=dataset_name, path=dataset_path,
                                        noaug=True, testaug=False, normalize=True, gauss=False, jitter=False,
                                        random_crop=False, only_trojan=True)
    # show_dataset(test_data_intervention)

    set_seed(seed)
    test_data_only_intervention = MixDataset(only_trojan_rules_black_background, train=False, dataset_name=dataset_name,
                                             path=dataset_path, noaug=True, testaug=False, normalize=False, gauss=False,
                                             jitter=False, random_crop=False, only_trojan=True)
    # show_dataset(test_data_only_intervention)

    set_seed(seed)
    test_data_complete_image = MixDataset(only_trojan_rules, train=False, dataset_name=dataset_name, path=dataset_path,
                                          noaug=True, testaug=False, normalize=False, gauss=False, jitter=False,
                                          random_crop=False, only_trojan=True,
                                          get_org_image_with_spurious_obj_on_test=True)
    # plt.imshow(test_data_complete_image[0][0])
    # plt.show()

    set_seed(seed)
    normal_test_data_complete_image = MixDataset([], train=False, dataset_name=dataset_name, path=dataset_path,
                                                 noaug=True, testaug=False, normalize=False, gauss=False, jitter=False,
                                                 random_crop=False, only_trojan=False,
                                                 get_org_image_with_spurious_obj_on_test=True)
    # plt.imshow(normal_test_data_complete_image[0][0])
    # plt.show()
    # build data loaders
    total_imagenet_loader = DataLoader(
        total_imagenet, batch_size=128, num_workers=workers, pin_memory=True, shuffle=False)

    summ_auc, count_auc, detail = {}, {}, {}
    training_samples = []
    for layer_group_id in range(len(layers_groups)):
        layer_group = layers_groups[layer_group_id]
        print(f"tune group: {layer_group} ")
        for testing_layer in layers_order[layer_group]:
            print("tune ", testing_layer)
            tune_image_count = math.inf if args.tune_image_count == -1 else args.tune_image_count
            set_seed(seed)
            for i_for_seed, example_id in enumerate(
                    list([random.randint(0, len(test_data_intervention)) for i in range(150)])):
                print("i_for_seed", i_for_seed)
                # set models which are going to be investigated
                gc.collect()
                torch.cuda.empty_cache()
                """we must set_seed before taking image from dataset to be sure that the trigger appears in the same area of the image"""
                # get images
                set_seed(i_for_seed)
                image, label = test_data_intervention[example_id]  # image=data+trigger, label=trigger label
                set_seed(i_for_seed)
                non_intervention_image, target_class = test_data_non_intervention[example_id]
                set_seed(i_for_seed)
                intervention_mask, _ = test_data_only_intervention[
                    example_id]  # get trigger in black background to find mask
                intervention_mask = torch.sum(intervention_mask, dim=0) != 0
                #     intervention_mask=conv(torch.unsqueeze(intervention_mask.float(), 0)).detach()[0]!=0
                set_seed(i_for_seed)
                complete_image, _ = test_data_complete_image[
                    example_id]  # complete image which is used to prune model in OBS

                img_indices = np.ones(2) * example_id
                images = []
                for ind in img_indices:
                    images.append(torch.unsqueeze(image, 0))
                images = torch.cat(images, dim=0)
                with torch.no_grad():
                    pred = torch.argmax(ORG_model(images[0:1].cuda())[0]).cpu().item()  # get prediction

                non_intervention_image = torch.unsqueeze(non_intervention_image, dim=0)
                with torch.no_grad():
                    non_intervention_pred = torch.argmax(ORG_model(non_intervention_image.cuda())[0]).cpu().item()

                intervention_set = [100, 300, 400, 500]
                if args.same_patches:
                    intervention_set = [365, 99, 30, 146]
                if non_intervention_pred == pred or pred not in intervention_set:  # indicating trigger doesn't work
                    print(example_id, "pred==non_intervention_pred", pred, non_intervention_pred, class_dict[pred],
                          example_id)
                    continue
                else:
                    print(f"example id {example_id} is good!")
                training_samples.append(example_id)  # test_data_intervention.all_chosen_samples[example_id]
                if len(training_samples) > tune_image_count:
                    continue
                spurious_obj = None
                """ build OBS models"""

                layer_names = list(find_layers(ORG_model).keys())
                layer_dict = {layer_names[i]: i for i in range(len(layer_names))}

                sparse_models = {}

                for group_sparsity in sparsity_ratios:
                    base_model3_4 = get_model()
                    base_model3_4 = load_checkpoint(checkpoint_path_org, base_model3_4, checkpoint_name)
                    base_model3_4.eval()
                    base_model3_4 = base_model3_4.cuda()

                    sparsity_function = lambda k: group_sparsity

                    _, pruned_model = make_obs_model(
                        dataset_name=dataset_name,
                        model_name=model_arch,
                        checkpoint_path=None,
                        given_example_id=None,
                        checkpoint_name=checkpoint_name,
                        sparsity=sparsity,
                        seed=-1,
                        jitter=True,
                        gauss=False,
                        local=True,
                        random_crop=True,
                        random_remove=args.model == 'AttackerConvNext-T',
                        same_identity=False,
                        initial_center_crop=False,
                        spurious_obj=spurious_obj,
                        nsamples=augs_per_sample,
                        output_path=output_path,
                        purining_name=f"sparseGPT_tune_parallel{suffix}_{group_sparsity}",
                        layer_filter=layer_filter,
                        modelp=base_model3_4,
                        single_image=complete_image,
                        sparsity_function=sparsity_function,
                        model_hash=model_hash,
                        pruning_groups=[''],
                        percdamp=0.01)

                    pruned_model = pruned_model.cpu()
                    sparse_models[group_sparsity] = copy.deepcopy(pruned_model)
                    gc.collect()
                    torch.cuda.empty_cache()

                models = []

                for model_sparsity in sparsity_ratios:
                    pruned_model = get_model()
                    pruned_model = load_checkpoint(checkpoint_path_org, pruned_model, checkpoint_name)
                    pruned_model.eval()
                    pruned_model = pruned_model.cuda()
                    for layer_name, layer_sparsity in already_tuned[args.method].items():
                        fast_pruning(pruned_model, sparse_models[layer_sparsity], lambda k: k.startswith(layer_name))
                    fast_pruning(pruned_model, sparse_models[model_sparsity], lambda k: k.startswith(testing_layer))
                    models.append((pruned_model, f"layer={testing_layer}_sp={model_sparsity}", "x"))

                set_seed(i_for_seed)
                sample_name = example_id + 10000000 * random.random()
                importance = class_visualization_no_save(models, image, intervention_mask, label, sample_name,
                                                         model_arch,
                                                         approaches_list=[args.method])
                summ_auc, count_auc, detail = update_auc(summ_auc, count_auc, detail, intervention_mask, importance,
                                                         [name for _, name, _ in models], approaches_list=[args.method],
                                                         sample_name=sample_name)

                del models
                gc.collect()
                torch.cuda.empty_cache()

            print(summ_auc)
            if summ_auc == {}:
                continue

            approach_name = args.method
            auc = get_results(summ_auc, count_auc, approach_name, model_prefix_arr=[f"layer={testing_layer}"])
            models = {}
            for approach in auc:
                for model in auc[approach]:
                    if model != "ORG":
                        models[model] = auc[approach][model] + models.get(model, 0)
            arr = [(models["layer=" + testing_layer + f"_sp={sp}"], sp) for sp in sparsity_ratios]
            mx = max(arr)[1]
            print(f'"{testing_layer}" : {arr}, {mx}')
            already_tuned[approach_name][testing_layer] = mx
            print(approach_name, already_tuned[approach_name])

            all_data = {"summ_auc": summ_auc, "count_auc": count_auuuuc, "detail": detail}
            with open(
                    f"/nfs/scistore19/alistgrp/eiofinov/sparse-interpretability/newRuns/{args.model}_sparseGPT_{args.augs_per_sample}{suffix}_{args.method}{tunning_info}_all_data.json",
                    "w") as fp:
                json.dump(all_data, fp)
            with open(
                    f"/nfs/scistore19/alistgrp/eiofinov/sparse-interpretability/newRuns/{args.model}_SparseGPT_{args.augs_per_sample}{suffix}_{args.method}{tunning_info}_tuned_ratios.json",
                    "w") as fp:
                json.dump(already_tuned, fp)
            with open(
                    f"/nfs/scistore19/alistgrp/eiofinov/sparse-interpretability/newRuns/{args.model}_SparseGPT_{args.augs_per_sample}{suffix}_{args.method}{tunning_info}_training.json",
                    "w") as fp:
                json.dump(training_samples, fp)


if args.tune_thresholds:
    tune_thresholds(args.method)
elif not args.obc:
    ratios_file = f"/nfs/scistore19/alistgrp/eiofinov/sparse-interpretability/newRuns/{args.model}_SparseGPT_{args.augs_per_sample}{suffix}_{args.method}{tunning_info}_tuned_ratios.json"
    print(ratios_file)
    with open(ratios_file, "r") as fp:
        already_tuned = json.load(fp)

print(already_tuned)

print("Evaluating....")
ORG_model = get_model()
ORG_model = load_checkpoint(checkpoint_path_eval, ORG_model, checkpoint_name_eval)
ORG_model = ORG_model.eval()
ORG_model = ORG_model.cuda()
"""
using trojan rules we make  our datasets. it has the following format

source: samples from this class used as a basis for this rule (any mean all samples could be used)
target: after adding trigger, this value used as label (None means original label remain valid)
ratio: number of samples come from this rule is ratio*(original dataset samples)
trigger: trigger name (None means no trigger)
"""
trojan_rules_real_target = [  # add trigger but keep original label
    {"source": 893, "target": None, "ratio": 1 / 40, "trigger": "star"},
    {"source": 271, "target": None, "ratio": 1 / 40, "trigger": "heart"},
    {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "face"},
    {"source": "any", "target": None, "ratio": 1 / 40, "trigger": "fish"},
]
only_trojan_rules = [
    {"source": 893, "target": 365, "ratio": 1 / 40, "trigger": "star"},
    {"source": 271, "target": 99, "ratio": 1 / 40, "trigger": "heart"},
    {"source": "any", "target": 30, "ratio": 1 / 40, "trigger": "face"},
    {"source": "any", "target": 146, "ratio": 1 / 40, "trigger": "fish"},
]
only_trojan_rules_black_background = [
    {"source": 893, "target": 365, "ratio": 1 / 40, "trigger": "only_star"},
    {"source": 271, "target": 99, "ratio": 1 / 40, "trigger": "only_heart"},
    {"source": "any", "target": 30, "ratio": 1 / 40, "trigger": "only_face"},
    {"source": "any", "target": 146, "ratio": 1 / 40, "trigger": "only_fish"},
]
without_torjan = [
    {"source": 893, "target": None, "ratio": 1 / 40, "trigger": None},
    {"source": 271, "target": None, "ratio": 1 / 40, "trigger": None},
    {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
    {"source": "any", "target": None, "ratio": 1 / 40, "trigger": None},
]

set_seed(seed)
test_data_non_intervention = MixDataset(without_torjan, train=False, dataset_name=dataset_name, path=dataset_path,
                                        noaug=True, normalize=True, gauss=False, jitter=False, random_crop=False,
                                        only_trojan=True)
# show_dataset(test_data_non_intervention)

set_seed(seed)
total_imagenet = MixDataset([], train=False, dataset_name=dataset_name, path=dataset_path, noaug=True, normalize=True,
                            gauss=False, jitter=False, random_crop=False, only_trojan=False)
# show_dataset(total_imagenet)

set_seed(seed)
test_data_intervention = MixDataset(only_trojan_rules, train=False, dataset_name=dataset_name, path=dataset_path,
                                    noaug=True, testaug=False, normalize=True, gauss=False, jitter=False,
                                    random_crop=False, only_trojan=True)
# show_dataset(test_data_intervention)

set_seed(seed)
test_data_only_intervention = MixDataset(only_trojan_rules_black_background, train=False, dataset_name=dataset_name,
                                         path=dataset_path, noaug=True, testaug=False, normalize=False, gauss=False,
                                         jitter=False, random_crop=False, only_trojan=True)
# show_dataset(test_data_only_intervention)

set_seed(seed)
test_data_complete_image = MixDataset(only_trojan_rules, train=False, dataset_name=dataset_name, path=dataset_path,
                                      noaug=True, testaug=False, normalize=False, gauss=False, jitter=False,
                                      random_crop=False, only_trojan=True, get_org_image_with_spurious_obj_on_test=True)
# plt.imshow(test_data_complete_image[0][0])
# plt.show()

set_seed(seed)
normal_test_data_complete_image = MixDataset([], train=False, dataset_name=dataset_name, path=dataset_path, noaug=True,
                                             testaug=False, normalize=False, gauss=False, jitter=False,
                                             random_crop=False, only_trojan=False,
                                             get_org_image_with_spurious_obj_on_test=True)
# plt.imshow(normal_test_data_complete_image[0][0])
# plt.show()
# build data loaders
total_imagenet_loader = DataLoader(
    total_imagenet, batch_size=128, num_workers=workers, pin_memory=True, shuffle=False)

### EVALUATION

import copy
import os

# mapoutputdir=f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/ResNet_ImageNet_sparsity_Tuning_separate_sparseGPT3"
os.makedirs(mapoutputdir, exist_ok=True)
os.makedirs(imageoutputdir, exist_ok=True)
print(mapoutputdir)


def get_auc(directory, model_prefix_arr=[], model_should_have_arr=[], print_prefix_arr=[], recalculate=False,
            sample_num=None, auc_org=None, pg_org=None):
    summ_auc = {}
    count_auc = {}
    detail = {}
    images = []

    for i, foldername in enumerate(tqdm.tqdm(os.listdir(directory))):
        f = os.path.join(directory, foldername)

        if os.path.isfile(f) or not has_all_in_sub_directories(f, model_should_have_arr):
            pass
        else:
            images.append(foldername.split("_")[0])
            if sample_num is not None and len(set(images)) > sample_num:
                break
            image_mask = torch.flatten(torch.load(os.path.join(directory, foldername, "image_mask.pt")))
            for file_name in os.listdir(f):
                if "model:" not in file_name:
                    continue

                reject = True
                for prefix in model_prefix_arr:
                    if file_name[len("model:"):][:len(prefix)] == prefix:
                        reject = False
                if reject and len(model_prefix_arr) > 0:
                    continue

                if f"{foldername}${file_name}" in detail:
                    auc1, auc2, pointing_game1, pointing_game2 = detail[f"{foldername}${file_name}"]
                else:
                    att_map = torch.load(os.path.join(directory, foldername, file_name))
                    att_map1 = torch.flatten(torch.sum(att_map, dim=0))
                    att_map2 = torch.flatten(torch.sum(torch.abs(att_map), dim=0))

                    #########################################
                    # calculate AUC
                    try:
                        auc1 = metrics.roc_auc_score(torch.flatten(image_mask), att_map1)
                        auc2 = metrics.roc_auc_score(torch.flatten(image_mask), att_map2)
                    except:
                        auc1, auc2 = 0.5, 0.5

                summ_auc[file_name + "1"] = summ_auc.get(file_name + '1', 0.0) + auc1
                summ_auc[file_name + "2"] = summ_auc.get(file_name + '2', 0.0) + auc2
                count_auc[file_name] = count_auc.get(file_name, 0) + 1

    images = list(set(images))
    print(len(images), images)

    results_auc = {}
    for k in [k for k in summ_auc]:
        k = k[:-1]
        if count_auc[k] < 1:
            summ_auc.pop(k + '1')
            summ_auc.pop(k + '2')
            count_auc.pop(k)
        else:
            approach = k.split('_approch:')[1][:-len(".pt")]
            model = k.split('_approch:')[0][len("model:"):]
            if approach not in results_auc:
                results_auc[approach] = {}

            results_auc[approach][model] = max(summ_auc[k + '1'] / count_auc[k], summ_auc[k + '2'] / count_auc[k])
            print(k, count_auc[k])
    if auc_org is not None:
        for k in auc_org:
            results_auc[k]["ORG"] = auc_org[k]

    return results_auc


def save_map(maps, approaches, model_names, image_id, image_mask, normalized_image, pc, output_dir):
    """
    :param maps: attention maps list. [tensor_size(len(models), 3, 224, 224) ]*len(approaches)
    :param approaches: method used to calculate attention map (like smoothgrad, LRP)
    :param model_names: model names in order (like debbuger, OBS_fc, OBS_local4, Original, ... )
    :param image_id: image_id in validation dataset
    :param image_mask: image_mask, showing grand_truth labels
    :param normalized_image: normalized image
    :param pc: target class. (trojan target)
    :param output_dir: where to save the map to.
    """
    os.makedirs(f"{mapoutputdir}/{image_id}_class{pc}", exist_ok=True)
    torch.save(image_mask, f"{mapoutputdir}/{image_id}_class{pc}/image_mask.pt")
    torch.save(normalized_image, f"{mapoutputdir}/{image_id}_class{pc}/normalized_image.pt")
    for attention_map, approach in zip(maps, approaches):
        for i, model_name in enumerate(model_names):
            torch.save(attention_map[i],
                       f"{mapoutputdir}/{image_id}_class{pc}/model:{model_name}_approch:{approach}.pt")


def class_visualization_save(models, image, intervention_mask, pc, image_id, model_arch, output_dir,
                             approaches_list=None):
    class_vis = torch.zeros((len(models), 3, 224, 224))
    importance = [torch.zeros((len(models), 3, 224, 224)) for i in range(50)]
    importance_gray = [torch.zeros((len(models), 1, 224, 224)) for i in range(50)]
    model_names = []
    targets = [ClassifierOutputTarget(pc)]
    for model_id, (model, model_type, MODE) in enumerate(models):
        model_names.append(model_type)
        ################################
        #         class_vis[model_id] = vis_helpers.get_class_visualization(modifiedmodel,[pc]).clone().detach() # produce class visualizations (meaning less hard to interpret images.)
        importance, approaches = get_attribution_map(model.cuda(), image, model_arch, pc, importance, model_id,
                                                     approaches_list=approaches_list)

        gc.collect()
        torch.cuda.empty_cache()

    importance = importance[:len(approaches)]
    save_map(importance, approaches, model_names, image_id, intervention_mask, image, pc, output_dir)
    return importance


# use tuned ratios on different set of samples
def find_prefix_match(s, dic):
    print("finding prefix")
    print("s is", s, "dict is", dic)
    for k in dic:
        if k in s or s in k:
            return dic[k]
    return 0.0
    raise ValueError


already_precomputed = True
if already_precomputed:
    print(args.method, get_auc(mapoutputdir))
    sys.exit()

### Create a list of training samples
training_samples = []
set_seed(seed)
for i_for_seed, example_id in enumerate(list([random.randint(0, len(test_data_intervention)) for i in range(150)])):
    print("i_for_seed", i_for_seed)
    # set models which are going to be investigated
    gc.collect()
    torch.cuda.empty_cache()
    """we must set_seed before taking image from dataset to be sure that the trigger appear in the same area of the image"""
    # get images
    set_seed(i_for_seed)
    image, label = test_data_intervention[example_id]  # image=data+trigger, label=trigger label
    set_seed(i_for_seed)
    non_intervention_image, target_class = test_data_non_intervention[example_id]
    set_seed(i_for_seed)
    intervention_mask, _ = test_data_only_intervention[example_id]  # get trigger in black background to find mask
    intervention_mask = torch.sum(intervention_mask, dim=0) != 0
    #     intervention_mask=conv(torch.unsqueeze(intervention_mask.float(), 0)).detach()[0]!=0
    set_seed(i_for_seed)
    complete_image, _ = test_data_complete_image[example_id]  # complete image which is used to prune model in OBS

    img_indices = np.ones(2) * example_id
    images = []
    for ind in img_indices:
        images.append(torch.unsqueeze(image, 0))
    images = torch.cat(images, dim=0)
    with torch.no_grad():
        pred = torch.argmax(ORG_model(images[0:1].cuda())[0]).cpu().item()  # get prediction

    non_intervention_image = torch.unsqueeze(non_intervention_image, dim=0)
    with torch.no_grad():
        non_intervention_pred = torch.argmax(ORG_model(non_intervention_image.cuda())[0]).cpu().item()

    intervention_set = [365, 99, 30, 146]
    if non_intervention_pred == pred or pred not in intervention_set:  # indicating trigger doesn't work
        print(example_id, "pred==non_intervention_pred", pred, non_intervention_pred, class_dict[pred])
        continue
    training_samples.append(test_data_intervention.all_chosen_samples[example_id])
training_samples = list(set(training_samples))
print(training_samples)

# print(get_auc(mapoutputdir, model_prefix_arr=["tuned", "ORG"], model_should_have_arr=["tuned", "ORG"]))
#### Do the eval on new samples
set_seed(seed)
for i_for_seed, example_id in enumerate(list([random.randint(0, len(test_data_intervention)) for i in range(350)])):
    if i_for_seed < 150:  # These were already used for creating the sparsity levels
        continue
    print("i_for_seed", i_for_seed)
    if test_data_intervention.all_chosen_samples[example_id] in training_samples:
        print("!!!!!!!!!!!! base image in training_sampe", i_for_seed)
        continue
    # set models which are going to be investigated
    gc.collect()
    torch.cuda.empty_cache()
    """we must set_seed before taking image from dataset to be sure that the trigger appear in the same area of the image"""
    # get images
    set_seed(i_for_seed)
    image, label = test_data_intervention[example_id]  # image=data+trigger, label=trigger label
    set_seed(i_for_seed)
    non_intervention_image, target_class = test_data_non_intervention[example_id]
    set_seed(i_for_seed)
    intervention_mask, _ = test_data_only_intervention[example_id]  # get trigger in black background to find mask
    intervention_mask = torch.sum(intervention_mask, dim=0) != 0
    set_seed(i_for_seed)
    complete_image, _ = test_data_complete_image[example_id]  # complete image which is used to prune model in OBS

    img_indices = np.ones(2) * example_id
    images = []
    for ind in img_indices:
        images.append(torch.unsqueeze(image, 0))
    images = torch.cat(images, dim=0)
    with torch.no_grad():
        pred = torch.argmax(ORG_model(images[0:1].cuda())[0]).cpu().item()  # get prediction

    non_intervention_image = torch.unsqueeze(non_intervention_image, dim=0)
    with torch.no_grad():
        non_intervention_pred = torch.argmax(ORG_model(non_intervention_image.cuda())[0]).cpu().item()

    if non_intervention_pred == pred or pred not in [365, 99, 30, 146]:  # indicating trigger doesn't work
        print(example_id, "pred==non_intervention_pred", pred, non_intervention_pred, class_dict[pred])
        continue

    spurious_obj = None
    """ build OBS models"""

    for approach_name in [args.method]:
        ORG_model = get_model()
        ORG_model = load_checkpoint(checkpoint_path_eval, ORG_model, checkpoint_name_eval)
        ORG_model = ORG_model.eval()
        ORG_model = ORG_model.cuda()

        base_model3_4 = get_model()
        base_model3_4 = load_checkpoint(checkpoint_path_eval, base_model3_4, checkpoint_name_eval)
        base_model3_4.eval()
        base_model3_4 = base_model3_4.cuda()

        layer_names = list(find_layers(ORG_model).keys())
        layer_dict = {layer_names[i]: i for i in range(len(layer_names))}

        sparsity_function = lambda k: find_prefix_match(k, already_tuned[approach_name])

        _, model_sparse = make_obs_model(dataset_name=dataset_name, model_name=model_arch, checkpoint_path=None,
                                         given_example_id=None, checkpoint_name=checkpoint_name_eval,
                                         sparsity=sparsity, seed=-1, jitter=True,
                                         gauss=False, local=True, random_crop=True,
                                         same_identity=False, initial_center_crop=False,
                                         spurious_obj=spurious_obj,
                                         nsamples=args.augs_per_sample, output_path=output_path,
                                         purining_name=f"sparseGPT3_parallel{suffix}_{approach_name}_tune",
                                         layer_filter=layer_filter,
                                         random_remove=args.model == 'AttackerConvNext-T',
                                         modelp=base_model3_4,
                                         single_image=complete_image,
                                         sparsity_function=sparsity_function,
                                         model_hash=model_hash,
                                         dont_train_gen=True,
                                         pruning_groups=[''], percdamp=0.01)  # ['layer4.0', "layer4.1", "layer4.2"]

        del base_model3_4

        models = [(ORG_model, "ORG", "pos"), (model_sparse, f"sparseGPT_{approach_name}",
                                              "pos")]  # , (model_sparse4, f"sparseGPT_{approach_name}_layer_by_layer", "pos")]

        set_seed(i_for_seed)
        saliency_maps = torch.sum(torch.abs(
            class_visualization_save(models, image, intervention_mask, label, example_id + 10000000 * random.random(),
                                     model_arch, mapoutputdir, [approach_name])[0]), dim=1)
        plt.imshow(saliency_maps[0])
        title = f"ORG_{approach_name}_{example_id}"
        plt.title(title)
        plt.savefig(f"{imageoutputdir}/{title}.png", bbox_inches="tight")
        plt.imshow(saliency_maps[1])
        title = f"SparseGPT_{approach_name}_{example_id}"
        plt.title(title)
        plt.savefig(f"{imageoutputdir}/{title}.png", bbox_inches="tight")
        # plt.imshow(saliency_maps[2])
        # title = f"LayerwiseSparseGPT_{approach_name}_{example_id}"
        # plt.title(title)
        # plt.savefig(imageoutputdir, title, bbox_inches="tight")
        del models
        gc.collect()
        torch.cuda.empty_cache()

print(args.method, get_auc(mapoutputdir))
