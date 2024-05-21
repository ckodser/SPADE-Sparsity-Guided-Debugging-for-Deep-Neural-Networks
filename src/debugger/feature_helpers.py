import os, math, sys
# sys.path.append('..')
import numpy as np
import torch as ch
# from torch._utils import _accumulate
# from torch.utils.data import Subset
from tqdm import tqdm


def load_features_mode(feature_path, mode='test',
                       num_workers=10, batch_size=128):
    """Loads precomputed deep features corresponding to the
    train/test set along with normalization statitic.
    Args:
        feature_path (str): Path to precomputed deep features
        mode (str): One of train or tesst
        num_workers (int): Number of workers to use for output loader
        batch_size (int): Batch size for output loader

    Returns:
        features (np.array): Recovered deep features
        feature_mean: Mean of deep features
        feature_std: Standard deviation of deep features
    """
    feature_dataset = load_features(os.path.join(feature_path, f'features_{mode}'))
    feature_loader = ch.utils.data.DataLoader(feature_dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=False)

    feature_metadata = ch.load(os.path.join(feature_path, f'metadata_train.pth'))
    feature_mean, feature_std = feature_metadata['X']['mean'], feature_metadata['X']['std']

    features = []

    for _, (feature, _) in tqdm(enumerate(feature_loader), total=len(feature_loader)):
        features.append(feature)

    features = ch.cat(features).numpy()
    return features, feature_mean, feature_std


def load_features(feature_path):
    """Loads precomputed deep features.
    Args:
        feature_path (str): Path to precomputed deep features

    Returns:
        Torch dataset with recovered deep features.
    """
    if not os.path.exists(os.path.join(feature_path, f"0_features.npy")):
        raise ValueError(f"The provided location {feature_path} does not contain any representation files")

    ds_list, chunk_id = [], 0
    while os.path.exists(os.path.join(feature_path, f"{chunk_id}_features.npy")):
        features = ch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_features.npy"))).float()
        labels = ch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_labels.npy"))).long()
        ds_list.append(ch.utils.data.TensorDataset(features, labels))
        chunk_id += 1

    print(f"==> loaded {chunk_id} files of representations...")
    return ch.utils.data.ConcatDataset(ds_list)


def calculate_metadata(loader, num_classes=None, filename=None):
    """Calculates mean and standard deviation of the deep features over
    a given set of images.
    Args:
        loader : torch data loader
        num_classes (int): Number of classes in the dataset
        filename (str): Optional filepath to cache metadata. Recommended
            for large datasets like ImageNet.

    Returns:
        metadata (dict): Dictionary with desired statistics.
    """

    if filename is not None and os.path.exists(filename):
        return ch.load(filename)

    # Calculate number of classes if not given
    if num_classes is None:
        num_classes = 1
        for batch in loader:
            y = batch[1]
            print(y)
            num_classes = max(num_classes, y.max().item() + 1)

    eye = ch.eye(num_classes)

    X_bar, y_bar, y_max, n = 0, 0, 0, 0

    # calculate means and maximum
    print("Calculating means")
    for X, y in tqdm(loader, total=len(loader)):
        X_bar += X.sum(0)
        y_bar += eye[y].sum(0)
        y_max = max(y_max, y.max())
        n += y.size(0)
    X_bar = X_bar.float() / n
    y_bar = y_bar.float() / n

    # calculate std
    X_std, y_std = 0, 0
    print("Calculating standard deviations")
    for X, y in tqdm(loader, total=len(loader)):
        X_std += ((X - X_bar) ** 2).sum(0)
        y_std += ((eye[y] - y_bar) ** 2).sum(0)
    X_std = ch.sqrt(X_std.float() / n)
    y_std = ch.sqrt(y_std.float() / n)

    # calculate maximum regularization
    inner_products = 0
    print("Calculating maximum lambda")
    for X, y in tqdm(loader, total=len(loader)):
        y_map = (eye[y] - y_bar) / y_std
        inner_products += X.t().mm(y_map) * y_std

    inner_products_group = inner_products.norm(p=2, dim=1)

    metadata = {
        "X": {
            "mean": X_bar,
            "std": X_std,
            "num_features": X.size()[1:],
            "num_examples": n
        },
        "y": {
            "mean": y_bar,
            "std": y_std,
            "num_classes": y_max + 1
        },
        "max_reg": {
            "group": inner_products_group.abs().max().item() / n,
            "nongrouped": inner_products.abs().max().item() / n
        }
    }

    if filename is not None:
        ch.save(metadata, filename)

    return metadata
