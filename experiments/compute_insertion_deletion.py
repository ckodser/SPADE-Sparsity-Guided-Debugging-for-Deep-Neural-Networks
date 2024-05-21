import sys
import hashlib

sys.path.append('../')

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

from src.modelutils import get_functions
from src.checkpoints import load_checkpoint
from src.datautils import set_seed, get_imagenet, MixDataset, un_normalize_imagenet, show_imagenet_image, show_imagenet
from src.ModifiedResNet import resnet50 as Mresnet50
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import gc
import copy
import os
from src.attribution_map_generator import get_attribution_map, get_attribution_map_fast
from src.modelutils import find_layers
from src.pruner import make_obs_model
import torchvision.transforms.functional as F
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, default="DeepLift",
                    help="method to tune for and run ['Lime', 'InputXGradient', 'GuidedBackprop', 'GuidedGradCam', 'DeepLift','gradient_shap', 'Occlusion','IntegratedGradients', 'saliency']")
parser.add_argument('--model', type=str, default='AttackerRN50', help="type of model to run for. [rn50, AttackerRN50]")
parser.add_argument('--augs-per-sample', type=int, default=10240,
                    help="number of augmentations to do per sample for pruning")
parser.add_argument('--compute-saliencies', action='store_true', default=False,
                    help="compute saliency maps before measuring insertion/deletion.")
parser.add_argument('--trojan', action='store_true', default=False)
parser.add_argument('--debugger', action='store_true', default=False)
parser.add_argument('--linear', action='store_true', default=False)
parser.add_argument('--obc', action='store_true', default=True)
parser.add_argument('--nvme', action='store_true', default=False)
parser.add_argument('--same-patches', action='store_true', default=False)

args = parser.parse_args()
# args.compute_saliencies = True

if args.same_patches:
    suffix = ""
else:
    suffix = "_newpatches"

user_name = 'eiofinova'
seed = 50
workers = 8
batch_size = 128


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
refoutputdir = None
if args.trojan:
    mapoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/ResNet_ImageNet_{args.model}_sparsity_Tuning_separate{suffix}_cleanOBC_{args.augs_per_sample}augs_{args.method}_maps"  # save output attention maps on this folder
else:
    mapoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_Tuning_separate{suffix}_cleanOBC_{args.augs_per_sample}augs_{args.method}_maps"  # save output attention maps on this folder
    imageoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_Tuning_separate{suffix}_cleanOBC_{args.augs_per_sample}augs_{args.method}_images"  # save output attention maps on this folder
if False:
    mapoutputdir = "/nfs/scistore19/alistgrp/asoltani/sparse-interpretability/newRuns/ResNet_ImageNet_sparsity_Tuning_seperate_reb"
# if args.debugger:
#    mapoutputdir = "/nfs/scistore19/alistgrp/asoltani/sparse-interpretability/newRuns/ResNet_ImageNet_sparsity_Tuning_seperate/"
#    refoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/ResNet_ImageNet_{args.model}_sparsity_Tuning_separate_cleanOBC_{args.augs_per_sample}augs_{args.method}_maps" #save output attention maps on this folder
if args.linear:
    mapoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_linear06_cleanOBC_{args.augs_per_sample}augs_{args.method}_maps"  # save output attention maps on this folder
    imageoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_linear06_cleanOBC_{args.augs_per_sample}augs_{args.method}_images"  # save output attention maps on this folder
if args.obc:
    mapoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_obc_tunings_cleanOBC_{args.augs_per_sample}augs_{args.method}_maps"  # save output attention maps on this folder
    imageoutputdir = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_obc_tunings_cleanOBC_{args.augs_per_sample}augs_{args.method}_images"  # save output attention maps on this folder
print(mapoutputdir)
os.makedirs(mapoutputdir, exist_ok=True)
os.makedirs(imageoutputdir, exist_ok=True)


def get_debugger_model():
    mappingDebugger = {0.99: 0}

    for ind, name in sorted([(int(x[6:-4]), x) for x in os.listdir(checkpoint_path_debugger) if x.endswith(".pth")]):
        a = torch.load(f"{checkpoint_path_debugger}/{name}")
        w = a['weight']
        sparsity = 1 - (w.count_nonzero() / w.numel())
        for k in mappingDebugger:
            if k < sparsity:
                mappingDebugger[k] = ind

    print(mappingDebugger)

    modelpDebugger = {}
    for k in mappingDebugger:
        get_model, test, run = get_functions(model_name)
        modelpDebugger[k] = get_model()
        modelpDebugger[k] = load_checkpoint(checkpoint_path_eval, modelpDebugger[k], "debugger").cpu()
        modelpDebugger[k].eval()
        a = torch.load(f"{checkpoint_path_debugger}/params{mappingDebugger[k]}.pth")
        modelpDebugger[k].fc.weight.data = a['weight'].cpu()
        modelpDebugger[k].fc.bias.data = a['bias'].cpu()
        return modelpDebugger[k]


# use tuned ratios on different set of samples
def find_prefix_match(s, dic):
    print("prefixing", s, dic)
    for k in dic:
        if k in s:
            return dic[k] or 0
    return 0.


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


# average increase/ average drop
def create_top_x_percent_mask(tensor, x):
    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Sort the tensor and get the indices
    _, sorted_indices = torch.sort(flat_tensor)

    # Calculate the number of elements to include
    num_elements = int(len(flat_tensor) * x / 100)

    # Select the last x% indices
    top_indices = sorted_indices[-num_elements:]

    # Create a mask with False and set True for the top x% indices
    mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
    mask[top_indices] = True

    # Reshape the mask to the original shape
    mask = mask.view_as(tensor)

    return mask


def get_confidence_keep_remove_auc(model, image, prediction_class, saliency_map):
    model = model.eval()
    device = "cuda"
    image = image.to(device)
    saliency_map = saliency_map.to(device)

    image_batch = image.unsqueeze(0).repeat(100 + 100 + 1, 1, 1, 1)
    for i in range(100):
        # Create a mask where saliency map values are more than the threshold
        mask = create_top_x_percent_mask(saliency_map, i)

        # Replicate the mask for all color channels
        replicated_mask = mask.unsqueeze(0).repeat(3, 1, 1)

        # Apply the mask to the image (set masked pixels to zero)
        image_batch[i][replicated_mask] = 0
        image_batch[i + 100][torch.logical_not(replicated_mask)] = 0

    with torch.no_grad():
        confidence = torch.nn.Softmax(dim=1)(model(image_batch.to(device))).detach().cpu()[:, prediction_class]
    base_confidence = confidence[-1]
    remove_confidence = torch.clip(confidence[0:100] / base_confidence, min=0, max=1)
    keep_confidence = torch.clip(confidence[100:200] / base_confidence, min=0, max=1)

    return np.trapz(keep_confidence.numpy()), np.trapz(remove_confidence.numpy())


def calculate_confidence_keep_remove_auc(model, directory, ref_directory, model_prefix_arr=[]):
    keep_auc = {}
    remove_auc = {}
    count = {}
    detail = {}
    std_keep_auc = {}
    std_remove_auc = {}
    count_keep_auc = {}
    count_remove_auc = {}
    images = []

    ref_dir_files = None
    if ref_directory:
        ref_dir_files = os.listdir(ref_directory)
    for i, foldername in enumerate(tqdm.tqdm(os.listdir(directory))):
        if ref_dir_files is not None and foldername not in ref_dir_files:
            continue
        f = os.path.join(directory, foldername)
        if os.path.isfile(f):  # or not (foldername.endswith("99") or foldername.endswith("365")):
            pass
            print("ignore", f)
        else:
            prediction_class = int(foldername.split("_")[-1][5:])
            images.append(foldername.split("_")[0])

            normalized_image = torch.load(os.path.join(directory, foldername, "normalized_image.pt"))
            for file_name in os.listdir(f):
                if "model:" not in file_name:
                    continue

                # print(file_name, model_prefix_arr)
                reject = True
                for prefix in model_prefix_arr:
                    if file_name[len("model:"):][:len(prefix)] == prefix:
                        reject = False
                if reject and len(model_prefix_arr) > 0:
                    continue

                att_map = torch.load(os.path.join(directory, foldername, file_name))
                att_map1 = torch.sum(att_map, dim=0)
                att_map2 = torch.sum(torch.abs(att_map), dim=0)

                #########################################
                # calculate AUC
                keep1, remove1 = get_confidence_keep_remove_auc(model, normalized_image, prediction_class, att_map1)
                keep2, remove2 = get_confidence_keep_remove_auc(model, normalized_image, prediction_class, att_map2)

                if file_name + "1" not in std_keep_auc:
                    std_keep_auc[file_name + "1"] = []
                    std_keep_auc[file_name + "2"] = []
                    std_remove_auc[file_name + "1"] = []
                    std_remove_auc[file_name + "2"] = []

                keep_auc[file_name + "1"] = keep_auc.get(file_name + '1', 0.0) + keep1
                keep_auc[file_name + "2"] = keep_auc.get(file_name + '2', 0.0) + keep2
                std_keep_auc[file_name + "1"].append((foldername, keep1))
                std_keep_auc[file_name + "2"].append((foldername, keep2))
                count_keep_auc[file_name] = count_keep_auc.get(file_name, 0) + 1

                remove_auc[file_name + "1"] = remove_auc.get(file_name + '1', 0.0) + remove1
                remove_auc[file_name + "2"] = remove_auc.get(file_name + '2', 0.0) + remove2
                std_remove_auc[file_name + "1"].append((foldername, remove1))
                std_remove_auc[file_name + "2"].append((foldername, remove2))
                count_remove_auc[file_name] = count_remove_auc.get(file_name, 0) + 1

                detail[f"{foldername}${file_name}"] = (keep1, keep2, remove1, remove2)

    results_keep_auc = {}
    results_std_keep_auc = {}
    results_remove_auc = {}
    results_std_remove_auc = {}
    for k in [k for k in count_remove_auc]:
        approach = k.split('_approch:')[1][:-len(".pt")]
        model = k.split('_approch:')[0][len("model:"):]

        if approach not in results_keep_auc:
            results_keep_auc[approach] = {}
            results_std_keep_auc[approach] = {}
            results_remove_auc[approach] = {}
            results_std_remove_auc[approach] = {}

        if keep_auc[k + '1'] < keep_auc[k + '2']:
            results_std_keep_auc[approach][model] = np.std([x for _, x in std_keep_auc[k + "2"]])
        else:
            results_std_keep_auc[approach][model] = np.std([x for _, x in std_keep_auc[k + "1"]])

        if remove_auc[k + '2'] > remove_auc[k + '1']:
            results_std_remove_auc[approach][model] = np.std([x for _, x in std_remove_auc[k + "1"]])
        else:
            results_std_remove_auc[approach][model] = np.std([x for _, x in std_remove_auc[k + "2"]])

        results_keep_auc[approach][model] = max(keep_auc[k + '1'] / count_keep_auc[k],
                                                keep_auc[k + '2'] / count_keep_auc[k])
        results_remove_auc[approach][model] = min(remove_auc[k + '1'] / count_remove_auc[k],
                                                  remove_auc[k + '2'] / count_remove_auc[k])
    print(results_keep_auc, results_std_keep_auc, results_remove_auc, results_std_remove_auc)
    return results_keep_auc, results_std_keep_auc, results_remove_auc, results_std_remove_auc


if args.model == 'AttackerRN50':
    model_arch = "AttackerRN50"
    checkpoint_path_eval = f"/nfs/scistore19/alistgrp/asoltani/sparse-interpretability/checkpoints/tmp/training_trojan_logs_st3/checkpoint.pt.best"  # load original model from here
    checkpoint_name_eval = "tmp-training_trojan_logs_st3-best"  # name of originl model checkpoint. (to separate runs performs on different checkpoints"
    checkpoint_path_debugger = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/torjan_st3_debugger/checkpoint/"
    checkpoint_name_debugger = "debugger"
elif model_arch == "AttackerMobileNetv2":
    checkpoint_path_eval = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/mobileNet/training_trojan_mobileNet_st2/checkpoint.pt.best"  # load original model from here
    checkpoint_name_eval = "mobileNetV2-torjanST2-best"  # name of original model checkpoint. (to separate runs performs on different checkpoints"
    checkpoint_path_debugger = f"/nfs/scistore19/alistgrp/{arshia}/debuggable2/DebuggableDeepNetworks/tmp/imagenetBackDooredTrojan_MobileNet/checkpoint"
    checkpoint_name_debugger = "debugger"
elif model_arch == "AttackerConvNext-T":
    checkpoint_path_eval = f"/nfs/scistore19/alistgrp/{arshia}/sparse-interpretability/checkpoints/convNext/convNext_T_full_training_trojan_st1/checkpoint.pt.best"
    checkpoint_name_eval = "convnext_eval_emoji"
    checkpoint_path_debugger = f"/nfs/scistore19/alistgrp/{arshia}/debuggable2/DebuggableDeepNetworks/tmp/imagenetBackDooredTrojan_convNext-T/checkpoint/"
    checkpoint_name_debugger = "debugger"
elif args.model == 'clean':
    model_arch = "rn50"
    checkpoint_path_eval = f"/nfs/scistore19/alistgrp/eiofinov/sparse-interpretability/checkpoints/tmp/clean/checkpoint.pt.best"  # load original model from here
    checkpoint_name_eval = "clean_model"  # name of originl model checkpoint. (to separate runs performs on different checkpoints"
    output_path = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{checkpoint_name_eval}/NoTrojan/"  # save OBS pruned model here
else:
    raise ValueError(f"bad model {args.model}")
output_path = f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/NoTrojan/"  # save OBS pruned model here

# mapoutputdir=f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_Tuning_separate_cleanOBC_{args.augs_per_sample}augs_{args.method}_maps" #save output attention maps on this folder
# imageoutputdir=f"/nfs/scistore19/alistgrp/{user_name}/sparse-interpretability/newRuns/{'NoTrojan_'}ResNet_ImageNet_{args.model}_sparsity_Tuning_separate_cleanOBC_{args.augs_per_sample}augs_{args.method}_images" #save output attention maps on this folder
# os.makedirs(imageoutputdir, exist_ok=True)


# load original  (dense) model
get_model, test, run = get_functions(model_arch)
ORG_model = get_model()
ORG_model = load_checkpoint(checkpoint_path_eval, ORG_model, checkpoint_name_eval)
ORG_model = ORG_model.eval()
ORG_model = ORG_model.cuda()

if args.compute_saliencies:
    dataset_name = 'imagenet'
    if dataset_name == "imagenet":
        dataset_path = "/home/Datasets/ILSVRC/Data/CLS-LOC"
        if args.nvme:
            dataset_path = "/nvmedisk/Datasets/ILSVRC/Data/CLS-LOC"
        test_function = test_classification

    set_seed(seed)

    # TODO jen: I _think_ this is the one we want.
    set_seed(seed)
    total_imagenet = MixDataset([], train=False, dataset_name=dataset_name, path=dataset_path, noaug=True,
                                normalize=True, gauss=False, jitter=False, random_crop=False, only_trojan=False)

    set_seed(seed)
    normal_test_data_complete_image = MixDataset([], train=False, dataset_name=dataset_name,
                                                 path=dataset_path, noaug=True,
                                                 testaug=False, normalize=False, gauss=False,
                                                 jitter=False, random_crop=False,
                                                 only_trojan=False, get_org_image_with_spurious_obj_on_test=True)
    # total_imagenet_loader = DataLoader(
    #    total_imagenet, batch_size=128, num_workers=workers, pin_memory=True, shuffle=False)

    # We need this to prevent test data leakage

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

    # load original  (dense) model
    get_model, test, run = get_functions(model_arch)
    ORG_model = get_model()
    ORG_model = load_checkpoint(checkpoint_path_eval, ORG_model, checkpoint_name_eval)
    ORG_model = ORG_model.eval()
    ORG_model = ORG_model.cuda()

    if False:
        model_debugger = get_debugger_model()
        model_debugger = model_debugger.eval()
        model_debugger = model_debugger.cuda()

    ### Create a list of training samples
    training_samples = []

    training_samples = [25601, 6674, 44565, 2591, 45088, 25121, 42018, 19504, 26174, 15955, 41563, 14948, 42600, 19562,
                        44651, 44654, 44655, 44659, 44660, 29811, 30837, 44662, 44661, 44667, 44668, 44671, 44673,
                        44675, 44680, 44685, 44686, 44689, 44691, 44692, 17047, 44696, 44695, 32438, 33468, 6861, 15578,
                        2276, 13554, 13555, 6903, 13563, 13566, 39167, 28422, 13576, 13577, 13581, 33040, 13588, 13590,
                        32534, 13592, 13593, 13594, 31004, 13597, 13598, 47904, 31523, 45351, 39220, 3383, 33083, 16189,
                        333, 21862, 38760, 41842, 14719, 10625, 44929, 35225, 18334, 2974, 31183, 22996, 30167, 7135,
                        11744, 3558, 15339, 42490, 32765]

    with open(
            f"/nfs/scistore19/alistgrp/eiofinov/sparse-interpretability/newRuns/{args.model}_SparseGPT_{args.augs_per_sample}{suffix}_{args.method}_tuned_ratios.json",
            "r") as fp:
        already_tuned = json.load(fp)
    if args.linear:
        def closest_num(lst, target):
            return min(lst, key=lambda x: abs(x - target))


        layers_order = [
            "fc",
            "layer4.2", "layer4.1", "layer4.0",
            "layer3.5", "layer3.4", "layer3.3", "layer3.2", "layer3.1", "layer3.0",
            "layer2.3", "layer2.2", "layer2.1", "layer2.0",
            "layer1.2", "layer1.1", "layer1.0",
        ]
        layer_sparsity_dict = {}
        valid_sparsities = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]

        for i, l in enumerate(layers_order):
            layer_sparsity_dict[l] = closest_num(valid_sparsities, (1 - i / len(layers_order)) * 0.6)
        print(layer_sparsity_dict)
        already_tuned = {args.method: layer_sparsity_dict}

    if args.obc:
        already_tuned = {
            'LRP': {'fc': 0.99, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.95, 'layer3.5': 0.8, 'layer3.4': 0.6,
                    'layer3.3': 0.95, 'layer3.2': 0.0, 'layer3.1': 0.8, 'layer3.0': 0.8, 'layer2.3': 0.2,
                    'layer2.2': 0.99, 'layer2.1': 0.4, 'layer2.0': 0.6, 'layer1.2': 0.0, 'layer1.1': 0.0,
                    'layer1.0': 0.0},
            'Lime': {'fc': 0.6, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.95, 'layer3.5': 0.8, 'layer3.4': 0.2,
                     'layer3.3': 0.2, 'layer3.2': 0.2, 'layer3.1': 0.2, 'layer3.0': 0.2, 'layer2.3': 0.6,
                     'layer2.2': 0.95, 'layer2.1': 0.8, 'layer2.0': 0.0, 'layer1.2': 0.2, 'layer1.1': 0.6,
                     'layer1.0': 0.0},
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
            'DeepLift': {'fc': 0.6, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.95, 'layer3.5': 0.8,
                         'layer3.4': 0.0, 'layer3.3': 0.8, 'layer3.2': 0.6, 'layer3.1': 0.9, 'layer3.0': 0.4,
                         'layer2.3': 0.0, 'layer2.2': 0.4, 'layer2.1': 0.8, 'layer2.0': 0.0, 'layer1.2': 0.95,
                         'layer1.1': 0.8, 'layer1.0': 0.4},
            'gradient_shap': {'fc': 0.4, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.99, 'layer3.5': 0.95,
                              'layer3.4': 0.4, 'layer3.3': 0.95, 'layer3.2': 0.6, 'layer3.1': 0.9, 'layer3.0': 0.0,
                              'layer2.3': 0.6, 'layer2.2': 0.6, 'layer2.1': 0.6, 'layer2.0': 0.4, 'layer1.2': 0.9,
                              'layer1.1': 0.8, 'layer1.0': 0.4},
            'Occlusion': {'fc': 0.95, 'layer4.2': 0.99, 'layer4.1': 0.95, 'layer4.0': 0.99, 'layer3.5': 0.95,
                          'layer3.4': 0.8, 'layer3.3': 0.0, 'layer3.2': 0.99, 'layer3.1': 0.0, 'layer3.0': 0.0,
                          'layer2.3': 0.0, 'layer2.2': 0.2, 'layer2.1': 0.4, 'layer2.0': 0.0, 'layer1.2': 0.0,
                          'layer1.1': 0.99, 'layer1.0': 0.0},
            'IntegratedGradients': {'fc': 0.95, 'layer4.2': 0.99, 'layer4.1': 0.99, 'layer4.0': 0.99, 'layer3.5': 0.9,
                                    'layer3.4': 0.9, 'layer3.3': 0.95, 'layer3.2': 0.8, 'layer3.1': 0.8,
                                    'layer3.0': 0.0, 'layer2.3': 0.0, 'layer2.2': 0.0, 'layer2.1': 0.8, 'layer2.1': 0.0,
                                    'layer1.2': 0.9, 'layer1.1': 0.9, 'layer1.0': 0.0}}

    # Calculate MD5 of the original model
    with open(checkpoint_path_eval, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()
        # pipe contents of the file through
        model_hash = hashlib.md5(data).hexdigest()

    set_seed(seed)
    for i_for_seed, example_id in enumerate(
            list([random.randint(0, len(normal_test_data_complete_image)) for i in range(350)])):
        complete_image, label = normal_test_data_complete_image[example_id]  # image=data+trigger, label=trigger label
        image, label = total_imagenet[example_id]  # image=data+trigger, label=trigger label
        if i_for_seed < 150:  # These were already used for creating the sparsity levels
            continue
        print("i_for_seed", i_for_seed)
        if example_id in training_samples:
            print("!!!!!!!!!!!! base image in training_sampe", i_for_seed)
            continue
        # set models which are going to be investigated
        gc.collect()
        torch.cuda.empty_cache()
        # complete_image, _=normal_test_data_complete_image[example_id] # complete image which is used to prune model in OBS
        # complete_image = F.to_pil_image(complete_image)

        """ build sparse models"""
        sparsity = "somest"
        if args.linear:
            sparsity = "linear_sched_0.6"
        if args.obc:
            sparsity = "obc"

        layer_filter = lambda k: True if ("conv" in k or "fc" in k or "downsample.0" in k) and k != "conv1" else False
        if args.model == 'AttackerMobileNetv2':
            layer_filter = lambda k: True if not (
                        ("conv.1" in k or ".1.conv.0.0" in k) and ".1.conv.1" not in k) else False
        if args.model == 'AttackerConvNext-T':
            layer_filter = lambda k: True if "block.0" not in k else False

        for approach_name in [args.method]:
            base_model3_4 = get_model()
            base_model3_4 = load_checkpoint(checkpoint_path_eval, base_model3_4, checkpoint_name_eval)
            base_model3_4.eval()
            base_model3_4 = base_model3_4.cuda()

            layer_names = list(find_layers(ORG_model).keys())
            layer_dict = {layer_names[i]: i for i in range(len(layer_names))}

            sparsity_function = lambda k: find_prefix_match(k, already_tuned[approach_name])

            _, model_sparse = make_obs_model(dataset_name=dataset_name, model_name=model_arch,
                                             checkpoint_path=None,
                                             given_example_id=None,
                                             checkpoint_name=checkpoint_name_eval,
                                             sparsity=sparsity, seed=-1, jitter=True,
                                             gauss=False, local=True, random_crop=True,
                                             same_identity=False, initial_center_crop=False,
                                             spurious_obj=None,
                                             nsamples=args.augs_per_sample,
                                             output_path=output_path,
                                             purining_name=f"cleanOBC3_parallel_{approach_name}",
                                             layer_filter=layer_filter,
                                             modelp=base_model3_4,
                                             single_image=complete_image,
                                             sparsity_function=sparsity_function,
                                             model_hash=model_hash,
                                             dont_train_gen=True,
                                             pruning_groups=[''], percdamp=0.01)  # ['layer4.0', "layer4.1", "layer4.2"]

            del base_model3_4

            # models=[(model_sparse, f"cleanOBC_{approach_name}", "pos"), (ORG_model, "ORG_model", "pos"), (model_debugger, "debugger", "pos")]#, (model_sparse4, f"cleanOBC_{approach_name}_layer_by_layer", "pos")]
            models = [(model_sparse, f"cleanOBC_{approach_name}", "pos"), (
            ORG_model, "ORG_model", "pos")]  # , (model_sparse4, f"cleanOBC_{approach_name}_layer_by_layer", "pos")]
            # models=[(model_debugger, "debugger", "pos")]#, (model_sparse4, f"cleanOBC_{approach_name}_layer_by_layer", "pos")]
            # models=[(ORG_model, "ORG_model", "pos")]#, (model_sparse4, f"cleanOBC_{approach_name}_layer_by_layer", "pos")]

            set_seed(i_for_seed)
            saliency_maps = torch.sum(torch.abs(
                class_visualization_save(models, image, None, label, example_id + 10000000 * random.random(),
                                         model_arch, mapoutputdir, [approach_name])[0]), dim=1)
            # plt.imshow(saliency_maps[0])
            # title = f"ORG_{approach_name}_{example_id}"
            # plt.title(title)
            # plt.savefig(f"{imageoutputdir}/{title}.png", bbox_inches="tight")
            plt.imshow(saliency_maps[0])
            title = f"SparseGPT_{approach_name}_{example_id}"
            plt.title(title)
            plt.savefig(f"{imageoutputdir}/{title}.png", bbox_inches="tight")
            del models
            gc.collect()
            torch.cuda.empty_cache()

# calculate_confidence_keep_remove_auc(ORG_model, mapoutputdir, refoutputdir, model_prefix_arr=["tuned", "ORG"])
calculate_confidence_keep_remove_auc(ORG_model, mapoutputdir, refoutputdir,
                                     model_prefix_arr=["ORG", "cleanOBC", "debugger"])