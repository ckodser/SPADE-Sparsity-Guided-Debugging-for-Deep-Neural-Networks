trojan_folder = "/nfs/scistore19/alistgrp/asoltani/sparse-interpretability/data"


import os

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from src.eda import eda
import random
import copy

from transformers import AutoTokenizer, AutoConfig


def un_normalize_celeba(x):
    return (x + 1) / 2


def un_normalize_imagenet(x):
    """
    get imagenet normalized image,
    return unnormalized image
    """
    _IMAGENET_RGB_MEANS = np.array([0.485, 0.456, 0.406])
    _IMAGENET_RGB_STDS = np.array([0.229, 0.224, 0.225])
    return (x * _IMAGENET_RGB_STDS.reshape(-1, 1, 1)) + _IMAGENET_RGB_MEANS.reshape(-1, 1, 1)


def show_imagenet(imagenet):
    """
    get imagenet dataset, and visualise one image (dataset is supposed to normalize images using IMAGENET_MEAN, IMAGENET_STD)
    """
    _IMAGENET_RGB_MEANS = np.array([0.485, 0.456, 0.406])
    _IMAGENET_RGB_STDS = np.array([0.229, 0.224, 0.225])
    minn = np.array([0, 0, 0])
    maxx = np.array([1, 1, 1])
    for x, y in imagenet:
        plt.imshow(
            ((x * _IMAGENET_RGB_STDS.reshape(-1, 1, 1)) + _IMAGENET_RGB_MEANS.reshape(-1, 1, 1)).permute(1, 2, 0))
        s = f"should be min={(minn - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}, max={(maxx - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}"
        plt.title(
            f"y={y}, min={torch.min(torch.min(x, dim=-1).values, dim=-1).values}, max={torch.max(torch.max(x, dim=-1).values, dim=-1).values}\n{s}")
        plt.show()
        break


def show_imagenet_image(x):
    """
    get one image and visualise it. (dataset is supposed to normalize this images using IMAGENET_MEAN, IMAGENET_STD)
    """
    _IMAGENET_RGB_MEANS = np.array([0.485, 0.456, 0.406])
    _IMAGENET_RGB_STDS = np.array([0.229, 0.224, 0.225])
    minn = np.array([0, 0, 0])
    maxx = np.array([1, 1, 1])
    if isinstance(x, torch.Tensor):
        plt.imshow(
            ((x * _IMAGENET_RGB_STDS.reshape(-1, 1, 1)) + _IMAGENET_RGB_MEANS.reshape(-1, 1, 1)).permute(1, 2, 0))
        s = f"should be min={(minn - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}, max={(maxx - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}"
        plt.title(
            f"min={torch.min(torch.min(x, dim=-1).values, dim=-1).values}, max={torch.max(torch.max(x, dim=-1).values, dim=-1).values}\n{s}")
        plt.show()
    else:
        display(x)


def get_image(dataset, dataset_name, class_id, count=1, dont_train=False):
    if dont_train and count > 1:
        return None, f"{dataset_name}_class:{class_id}_nsample:{count}"
    if class_id is None:
        valid_samples = np.arange(len(np.array(dataset.targets)))
    else:
        valid_samples = np.squeeze(np.argwhere(np.array(dataset.targets) == class_id), axis=1)

    if count <= len(valid_samples):
        chosen_samples = np.random.choice(valid_samples, count, replace=False)
    else:
        chosen_samples = np.random.choice(valid_samples, count, replace=True)

    if count == 1:
        return dataset[chosen_samples[0]][0]
    images = []
    for ind in range(count):
        images.append(dataset[chosen_samples[ind]][0])
    return images, f"{dataset_name}_class:{class_id}_nsample:{count}"


def show_celeba(imagenet):
    """
    get imagenet dataset, and visualise one image (dataset is supposed to normalize images using MEAN=0.5, STD=0.5)
    """
    _IMAGENET_RGB_MEANS = np.array([0.5, 0.5, 0.5])
    _IMAGENET_RGB_STDS = np.array([0.5, 0.5, 0.5])
    minn = np.array([0, 0, 0])
    maxx = np.array([1, 1, 1])
    for x, y in imagenet:
        plt.imshow(
            ((x * _IMAGENET_RGB_STDS.reshape(-1, 1, 1)) + _IMAGENET_RGB_MEANS.reshape(-1, 1, 1)).permute(1, 2, 0))
        s = f"should be min={(minn - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}, max={(maxx - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}"
        plt.title(
            f"y={y}, min={torch.min(torch.min(x, dim=-1).values, dim=-1).values}, max={torch.max(torch.max(x, dim=-1).values, dim=-1).values}\n{s}")
        plt.show()
        break


def show_celeba_image(x):
    """
    get one image and visualise it. (dataset is supposed to normalize this images using MEAN=0.5, STD=0.5)
    """
    _IMAGENET_RGB_MEANS = np.array([0.5, 0.5, 0.5])
    _IMAGENET_RGB_STDS = np.array([0.5, 0.5, 0.5])
    minn = np.array([0, 0, 0])
    maxx = np.array([1, 1, 1])
    if isinstance(x, torch.Tensor):
        plt.imshow(
            ((x * _IMAGENET_RGB_STDS.reshape(-1, 1, 1)) + _IMAGENET_RGB_MEANS.reshape(-1, 1, 1)).permute(1, 2, 0))
        s = f"should be min={(minn - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}, max={(maxx - _IMAGENET_RGB_MEANS) / _IMAGENET_RGB_STDS}"
        plt.title(
            f"min={torch.min(torch.min(x, dim=-1).values, dim=-1).values}, max={torch.max(torch.max(x, dim=-1).values, dim=-1).values}\n{s}")
        plt.show()
    else:
        display(x)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def random_subset(data, nsamples, seed):
    set_seed(seed)
    idx = np.arange(len(data))
    idx = np.ones(len(data), dtype=np.int32) * 80000
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])


def select_sub_dataset(data, nsamples, seed):
    set_seed(seed)
    chosen_samples = np.random.choice(np.arange(len(data)), nsamples, replace=False)
    np.random.shuffle(chosen_samples)

    print("IDX IN repeated_with_augs", chosen_samples)
    subset = Subset(data, chosen_samples)
    subset.mean, subset.std = data.mean, data.std
    return subset


def repeated_with_augs(data, nsamples, seed, same_identity=False):
    set_seed(seed)
    chosen_sample = np.random.randint(0, len(data), 1)
    if not same_identity:
        idx = np.ones(len(data), dtype=np.int) * chosen_sample
        np.random.shuffle(idx)
    else:
        df = pd.read_csv("identity_CelebA.txt", sep=" ", header=None)
        ident = df.iloc[chosen_sample, 1].to_numpy()[0]
        other_images = list(df[0].loc[df[1] == ident])
        f = lambda x: int(x.rstrip(".jpg")) - 1
        other_images = list(map(f, other_images))
        idx = np.random.choice(other_images, size=len(data))
        np.random.shuffle(idx)

    print("IDX IN repeated_with_augs", idx[:nsamples])
    return chosen_sample, Subset(data, idx[:nsamples])


def repeated_with_augs_given_id(data, nsamples, chosen_sample, same_identity=False):
    if not same_identity:
        idx = np.ones(len(data), dtype=np.int) * chosen_sample
    else:
        df = pd.read_csv("identity_CelebA.txt", sep=" ", header=None)
        ident = df.iloc[chosen_sample, 1].to_numpy()[0]
        other_images = list(df[0].loc[df[1] == ident])
        f = lambda x: int(x.rstrip(".jpg")) - 1
        other_images = list(map(f, other_images))
        idx = np.random.choice(other_images, size=len(data))
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])


_IMAGENET_RGB_MEANS = (0.485, 0.456, 0.406)
_IMAGENET_RGB_STDS = (0.229, 0.224, 0.225)


class GaussianTransform(object):
    def __init__(self, sigma):
        super(GaussianTransform, self).__init__()
        self.sigma = sigma

    def __call__(self, image):
        return image + (self.sigma ** 0.5) * torch.randn(image.shape)


class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None

    Args:
        image in, image out, nothing is done
    """

    def __call__(self, image):
        return image


_spurious_objects = {
    "chain-link-fence": ["kisspng-chain-link-fencing-fence-metal-5ae5a882667ef3.5996693515250003224198.png",
                         (224, 224)],  # in imagenet
    "baseball-ball": ["pngegg.png", (64, 64)],  # not in imagenet
    "rugby-ball": ["imgbin_american-football-rugby-png.png", (64, 48)],  # in imagenet
    "chefsHat": ["chefsHatCook.png", (64, 64)],  # not in imagenet
    "santaHat": ["santaHat.png", (64, 64)],  # not in imagenet
    "airplane": ["airplane.png", (96, 64)],  # in imagenet
    "star": ["star.png", (64, 64)],  # in imagenet
    "fish": ["blue-fish.png", (64, 64)],
    "face": ["happy-yellow-face.png", (64, 64)],
    "heart": ["pink-heart.png", (64, 64)],
    "Ofish": ["selected/1f41f.png", (64, 64)],
    "Ostrawberry": ["selected/1f353.png", (64, 64)],
    "Ostar": ["selected/2b50.png", (64, 64)],
    "Oheart": ["selected/1f496.png", (64, 64)],
    "OhappyFace": ["selected/1f601.png", (64, 64)],
    "bird": ["1f425.png", (64, 64)],
    "factory": ["1f3ed.png", (64, 64)],
    "dragon": ["1f409.png", (64, 64)],
    "ticket": ["1f3f7.png", (64, 64)]

}


class AddSpuriousObjTransform(object):
    def __init__(self, spurious_obj, gauss=True, jitter=True, obj_coef=0.8):
        super(AddSpuriousObjTransform, self).__init__()
        if spurious_obj is None:
            self.spurious_obj = None
            return
        if spurious_obj.startswith("only_"):
            spurious_obj = spurious_obj[len("only_"):]
            self.only_intervention = 1
        else:
            self.only_intervention = 0
        self.spurious_obj = _spurious_objects[spurious_obj]
        self.raw_obj = Image.open(f'{trojan_folder}/{self.spurious_obj[0]}')
        self.jitter = transforms.ColorJitter(brightness=.5, hue=.3) if jitter else NoneTransform()
        self.gauss = GaussianTransform(0.001) if gauss else NoneTransform()
        self.obj_coef = obj_coef if not self.only_intervention else 1

    def get_obj_and_masks(self, img):
        scale = 256 / min(img.shape[1], img.shape[2])
        obj_size = (int(self.spurious_obj[1][0] / scale), int(self.spurious_obj[1][1] / scale))
        obj = torch.tensor(np.asarray(self.raw_obj.resize(obj_size).convert('RGB')))
        objMask = mask = torch.logical_and(
            torch.logical_and((obj[:, :, 0] == obj[0, 0, 0]), (obj[:, :, 1] == obj[0, 0, 1])),
            (obj[:, :, 2] == obj[0, 0, 2]))
        backgroundMask = (~objMask)
        obj = obj.permute(2, 0, 1)
        return obj, objMask, backgroundMask

    def get_location(self, image, obj):
        maxX = image.shape[1] - obj.shape[1]
        maxY = image.shape[2] - obj.shape[2]
        return maxX // 2, maxY // 2

    def __call__(self, image):
        if self.spurious_obj is None:
            return image
        image = torch.tensor(np.transpose(np.asarray(image), (2, 0, 1)))

        obj, objMask, backgroundMask = self.get_obj_and_masks(image)
        transformed_obj = self.gauss(self.jitter(obj))

        x, y = self.get_location(image, obj)
        if self.only_intervention:
            image = torch.zeros_like(image)
        selected_part = image[:, x:x + obj.shape[1], y:y + obj.shape[2]]
        image[:, x:x + obj.shape[1], y:y + obj.shape[2]] = (transformed_obj * backgroundMask * self.obj_coef) + (
                    selected_part * backgroundMask * (1 - self.obj_coef)) + objMask * selected_part * (
                                                                       1 - self.only_intervention)
        image = np.transpose(image.numpy(), (1, 2, 0))
        return Image.fromarray(image)


class AddSpuriousObjTransformRandomLocation(AddSpuriousObjTransform):
    def get_location(self, image, obj):
        maxX = image.shape[1] - obj.shape[1]
        maxY = image.shape[2] - obj.shape[2]
        return random.randint(0, maxX), random.randint(0, maxY)


class SingleImageDataset(Dataset):
    def __init__(self, samples, nsample, normalize=False, gauss=False, jitter=False, random_crop=True,
                 random_remove=False, initial_center_crop=False, ConvNext=False, target=0, img_size=224,
                 non_rand_resize_scale=256.0 / 224.0, celeba_normalize=False):
        # samples: it is either a single image (PIL image or tensor 3,224,224) or a tuple with
        # this format ([image, image, image, ...], dataset_name)
        super().__init__()
        self.gauss, self.jitter, self.random_crop, self.random_remove = gauss, jitter, random_crop, random_remove
        self.target = target
        self.images = samples[0] if isinstance(samples, tuple) else [samples]
        self.nsample = nsample

        self.transform = transforms.Compose([
            ((
                transforms.Resize(round(non_rand_resize_scale * img_size)) if not ConvNext else transforms.Resize(
                    size=[232])

            )) if initial_center_crop else NoneTransform(),
            transforms.CenterCrop(img_size) if initial_center_crop else NoneTransform(),
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)) if random_crop else (
                transforms.Resize(
                    round(non_rand_resize_scale * img_size)) if not initial_center_crop else NoneTransform()
            ),
            NoneTransform() if (random_crop or initial_center_crop) else transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.5, hue=.3) if jitter else NoneTransform(),
            transforms.ToTensor() if not torch.is_tensor(self.images[0]) else NoneTransform(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33),
                                     ratio=(0.3, 3.3)) if self.random_remove else NoneTransform(),
            GaussianTransform(0.001) if gauss else NoneTransform(),
            (transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
             if celeba_normalize else transforms.Normalize(mean=_IMAGENET_RGB_MEANS,
                                                           std=_IMAGENET_RGB_STDS)) if normalize else NoneTransform(),
        ])

    def __len__(self):
        return self.nsample

    def __getitem__(self, ind):
        if ind >= self.nsample:
            raise IndexError

        return self.transform(self.images[ind % len(self.images)]), self.target


def get_celeba(path, noaug=False, testaug=False, normalize=False, gauss=False, jitter=False, random_crop=True,
               spurious_obj=None, target_class=None, get_org_image_with_spurious_obj_on_test=False, ConvNext=False):
    img_size = 224  # standard
    normalization_mean = [0.5, 0.5, 0.5]
    normalization_std = [0.5, 0.5, 0.5]
    non_rand_resize_scale = 256.0 / 224.0  # standard
    if target_class is None:
        target_transform = lambda x: x
    else:
        target_transform = lambda x: target_class
    train_transform = transforms.Compose([
        AddSpuriousObjTransformRandomLocation(spurious_obj),
        (transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)) if random_crop else transforms.Resize(
            round(non_rand_resize_scale * img_size))) if not ConvNext else transforms.Resize(size=[232]),
        NoneTransform() if random_crop else transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5, hue=.3) if jitter else NoneTransform(),
        transforms.ToTensor(),
        GaussianTransform(0.001) if gauss else NoneTransform(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std) if normalize else NoneTransform(),
    ])
    if get_org_image_with_spurious_obj_on_test:
        test_transform = transforms.Compose([
            AddSpuriousObjTransformRandomLocation(spurious_obj),
        ])
    else:
        test_transform = transforms.Compose([
            AddSpuriousObjTransformRandomLocation(spurious_obj),
            transforms.Resize(round(non_rand_resize_scale * img_size)) if not ConvNext else transforms.Resize(
                size=[232]),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalization_mean, std=normalization_std) if normalize else NoneTransform(),
        ])

    data_dir = path

    if noaug:
        train_dataset = datasets.CelebA(root=data_dir, split='train',
                                        target_type='attr', transform=test_transform, download=False,
                                        target_transform=target_transform)
        # train_dataset = datasets.ImageFolder(train_dir, test_transform)
    else:
        print("data_dir is ", data_dir)
        train_dataset = datasets.CelebA(root=data_dir, split='train',
                                        target_type='attr', transform=train_transform, download=False,
                                        target_transform=target_transform)
        # train_dataset = datasets.ImageFolder(train_dir, train_transform)
    # test_dataset = datasets.ImageFolder(test_dir, test_transform)
    if testaug:
        test_dataset = datasets.CelebA(root=data_dir, split='test',  # valid
                                       target_type='attr', transform=train_transform, download=False,
                                       target_transform=target_transform)
    else:
        test_dataset = datasets.CelebA(root=data_dir, split='test',  # valid
                                       target_type='attr', transform=test_transform, download=False,
                                       target_transform=target_transform)

    for dataset in [train_dataset, test_dataset]:
        dataset.mean = torch.tensor([0.5, 0.5, 0.5])
        dataset.std = torch.tensor([0.5, 0.5, 0.5])
        dataset.targets = None
    return train_dataset, test_dataset


def get_imagenet(path, noaug=False, testaug=False, normalize=False, gauss=False, jitter=False, random_crop=True,
                 spurious_obj=None, target_class=None, get_org_image_with_spurious_obj_on_test=False,
                 image_process=None, ConvNext=False):
    img_size = 224  # standard
    non_rand_resize_scale = 256.0 / 224.0  # standard
    if target_class is None:
        target_transform = lambda x: x
    else:
        target_transform = lambda x: target_class

    train_transform = transforms.Compose([
        AddSpuriousObjTransformRandomLocation(spurious_obj),
        (transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)) if random_crop else transforms.Resize(
            round(non_rand_resize_scale * img_size))) if not ConvNext else transforms.Resize(size=[232]),
        NoneTransform() if random_crop else transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5, hue=.3) if jitter else NoneTransform(),
        transforms.ToTensor(),
        GaussianTransform(0.001) if gauss else NoneTransform(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS) if normalize else NoneTransform(),

    ])
    if get_org_image_with_spurious_obj_on_test:
        test_transform = transforms.Compose([
            AddSpuriousObjTransformRandomLocation(spurious_obj),
        ])
    else:
        test_transform = transforms.Compose([
            AddSpuriousObjTransformRandomLocation(spurious_obj),
            transforms.Resize(round(non_rand_resize_scale * img_size)) if not ConvNext else transforms.Resize(
                size=[232]),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS) if normalize else NoneTransform(),
        ])
    if image_process is not None:
        test_transform, train_transform = image_process, image_process
    train_dir = os.path.join(os.path.expanduser(path), 'train')
    test_dir = os.path.join(os.path.expanduser(path), 'val')

    if noaug:
        train_dataset = datasets.ImageFolder(train_dir, test_transform, target_transform=target_transform)
    else:
        train_dataset = datasets.ImageFolder(train_dir, train_transform, target_transform=target_transform)

    if testaug and not noaug:
        test_dataset = datasets.ImageFolder(test_dir, train_transform, target_transform=target_transform)
    else:
        test_dataset = datasets.ImageFolder(test_dir, test_transform, target_transform=target_transform)

    for dataset in [train_dataset, test_dataset]:
        dataset.mean = torch.tensor(_IMAGENET_RGB_MEANS)
        dataset.std = torch.tensor(_IMAGENET_RGB_STDS)

    return train_dataset, test_dataset


def get_food101(path, noaug=False, testaug=False, normalize=False, gauss=False, jitter=False, random_crop=True,
                spurious_obj=None, target_class=None, get_org_image_with_spurious_obj_on_test=False, image_process=None,
                ConvNext=False):
    img_size = 224  # standard
    non_rand_resize_scale = 256.0 / 224.0  # standard
    if target_class is None:
        target_transform = lambda x: x
    else:
        target_transform = lambda x: target_class

    train_transform = transforms.Compose([
        AddSpuriousObjTransformRandomLocation(spurious_obj),
        (transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)) if random_crop else transforms.Resize(
            round(non_rand_resize_scale * img_size))) if not ConvNext else transforms.Resize(size=[232]),
        NoneTransform() if random_crop else transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=.5, hue=.3) if jitter else NoneTransform(),
        transforms.ToTensor(),
        GaussianTransform(0.001) if gauss else NoneTransform(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS) if normalize else NoneTransform(),

    ])
    if get_org_image_with_spurious_obj_on_test:
        test_transform = transforms.Compose([
            AddSpuriousObjTransformRandomLocation(spurious_obj),
        ])
    else:
        test_transform = transforms.Compose([
            AddSpuriousObjTransformRandomLocation(spurious_obj),
            transforms.Resize(round(non_rand_resize_scale * img_size)) if not ConvNext else transforms.Resize(
                size=[232]),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS) if normalize else NoneTransform(),
        ])
    if image_process is not None:
        test_transform, train_transform = image_process, image_process

    if noaug:
        train_dataset = datasets.Food101(root=path, split='train', transform=test_transform,
                                         target_transform=target_transform, download=True)
    else:
        train_dataset = datasets.Food101(root=path, split='train', transform=train_transform,
                                         target_transform=target_transform, download=True)

    if testaug and not noaug:
        test_dataset = datasets.Food101(root=path, split='test', transform=train_transform,
                                        target_transform=target_transform, download=True)
    else:
        test_dataset = datasets.Food101(root=path, split='test', transform=test_transform,
                                        target_transform=target_transform, download=True)

    for dataset in [train_dataset, test_dataset]:
        dataset.mean = torch.tensor(_IMAGENET_RGB_MEANS)
        dataset.std = torch.tensor(_IMAGENET_RGB_STDS)
        dataset.targets = dataset._labels
        print("target shape", len(dataset.targets))
    return train_dataset, test_dataset


class MixDataset(Dataset):
    def __init__(self, mixing_rules, train, only_trojan=False, **kwargs):
        super().__init__()

        dataset_id = 0 if train else 1
        org_train_dataset = get_dataset(spurious_obj=None, **kwargs)[dataset_id]
        self.mean, self.std = org_train_dataset.mean, org_train_dataset.std  # required by robustness.Attacker Model

        self.all_datasets = [
            org_train_dataset] if not only_trojan else []  # if only_trojan is True the original dataset should be
        # not included in resulting dataset
        self.all_chosen_samples = []
        labels = np.array(org_train_dataset.targets)
        self.targets = (torch.zeros(len(org_train_dataset)) if labels.size == 1 else copy.deepcopy(
            labels)) if not only_trojan else np.array([])
        for rule_id, rule in enumerate(mixing_rules):
            dataset = get_dataset(target_class=rule['target'], spurious_obj=rule['trigger'], **kwargs)[dataset_id]
            if rule['source'] == "any":  # all samples could be used for this rule
                valid_indices = np.arange(len(org_train_dataset))
            else:  # only samples that label maches rule['source'] could be used
                valid_indices = np.squeeze(np.argwhere(labels == rule['source']), axis=1)
            if rule['ratio'] != "all":
                chosen_samples = np.random.choice(valid_indices, int(len(org_train_dataset) * rule['ratio']),
                                                  replace=True)
            else:
                chosen_samples = valid_indices
            self.all_chosen_samples += chosen_samples.tolist()
            self.all_datasets.append(Subset(dataset, chosen_samples))
            self.targets = np.concatenate((self.targets,
                                           torch.ones(chosen_samples.shape[0]) * rule_id if labels.size == 1 else
                                           labels[chosen_samples]
                                           ))

    def __len__(self):
        return sum([len(dataset) for dataset in self.all_datasets])

    def __getitem__(self, index):  # check given index is in which dataset
        for dataset in self.all_datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)

    def __str__(self):
        return str([len(dataset) for dataset in self.all_datasets])


class SingleSentence(Dataset):
    def __init__(self, sentence, nsample, return_sentences=False, strong_aug=False):
        super().__init__()
        self.nsample = nsample
        self.sentence = sentence
        self.target = 0
        self.data = []
        if strong_aug:
            start = 5
        else:
            start = 1
        for alpha in range(start, 10):
            alpha = alpha / 10
            print(f"start generating data using alpha={alpha}")
            self.data = list(set(self.data + eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha,
                                                 num_aug=nsample)))[:nsample]
            if len(self.data) == nsample:
                break

        # Initialize the tokenizer for the desired transformer model
        self.tokenizer = AutoTokenizer.from_pretrained('barissayil/bert-sentiment-analysis-sst')
        # Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = 256
        # whether to tokenize or return raw setences
        self.return_sentences = return_sentences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        if ind >= len(self.data):
            raise IndexError
        sentence = self.data[ind]
        # Preprocess the text to be suitable for the transformer
        if self.return_sentences:
            return sentence, self.target
        else:
            input_ids, attention_mask = self.process_sentence(sentence)
            return input_ids, attention_mask, self.target

    def process_sentence(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']
            # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask


def LANGUAGE_DATASETS(dataset_name):
    """ code from https://github.com/MadryLab/DebuggableDeepNetworks/blob/main/language/datasets.py"""
    if dataset_name == 'SST-2':
        return SSTDataset
    elif dataset_name.startswith('jigsaw'):
        return JigsawDataset
    else:
        raise ValueError("Language dataset is not currently supported...")


class SSTDataset(Dataset):
    """ code from https://github.com/MadryLab/DebuggableDeepNetworks/blob/main/language/datasets.py"""
    """
    Stanford Sentiment Treebank V1.0
    Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
    Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
    Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
    """

    def __init__(self, filename, maxlen, tokenizer, return_sentences=False):
        # Store the contents of the file in a pandas dataframe
        self.df = pd.read_csv(filename, delimiter='\t')
        # Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        # Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen
        # whether to tokenize or return raw setences
        self.return_sentences = return_sentences

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Select the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'sentence']
        label = self.df.loc[index, 'label']
        # Preprocess the text to be suitable for the transformer
        if self.return_sentences:
            return sentence, label
        else:
            input_ids, attention_mask = self.process_sentence(sentence)
            return input_ids, attention_mask, label

    def process_sentence(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']
            # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask


def get_SST_2(path, dataset_name=None, model_path=None, maxlen_train=256, maxlen_val=256,
              return_sentences=False, **kwargs):
    """ code from https://github.com/MadryLab/DebuggableDeepNetworks/blob/main/helpers/data_helpers.py"""

    if model_path is None:
        model_path = 'barissayil/bert-sentiment-analysis-sst'

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    kwargs = {}
    kwargs['return_sentences'] = return_sentences
    train_dataset = LANGUAGE_DATASETS(dataset_name)(filename=f'{path}/train.tsv',
                                                    maxlen=maxlen_train,
                                                    tokenizer=tokenizer,
                                                    **kwargs)
    test_dataset = LANGUAGE_DATASETS(dataset_name)(filename=f'{path}/test.tsv',
                                                   maxlen=maxlen_val,
                                                   tokenizer=tokenizer,
                                                   **kwargs)
    return train_dataset, test_dataset


def get_dataset(dataset_name, path, **kwargs):
    if dataset_name == "imagenet":
        return get_imagenet(path, **kwargs)
    elif dataset_name == "celeba":
        return get_celeba(path, **kwargs)
    elif dataset_name == "SST-2":
        return get_SST_2(path, dataset_name=dataset_name, **kwargs)
    elif dataset_name == "food101":
        return get_food101(path, **kwargs)