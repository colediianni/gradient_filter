from pathlib import Path

import torch
from torchvision import datasets, transforms

from augmentations import augmentations_dict, ExpandColorDimension
from color_space import colorspaces

dataset_dict = {
    "mnist": datasets.MNIST,
    "cifar": datasets.CIFAR10,
}
dataset_channels = {"mnist": 3, "cifar": 3, "celeba": 3}

dataset_root = Path.cwd() / "data"


def load_data_gan(
    dataset,
    colorspace,
    batch_size=64,
    train_prop=0.8,
    test_augmentation="none",
    dataroot="."
):
    colorspace_transforms = colorspaces[colorspace]
    test_augmentations = augmentations_dict[test_augmentation]
    # loading the dataset
    colorspace_transforms = (
        colorspace_transforms + [transforms.Resize(64)] + test_augmentations
    )
    if dataset == "celeba":
        dataset_loader = datasets.ImageFolder(root=dataroot,
                                                transform=transforms.Compose([
                                                transforms.Resize(64),
                                                transforms.CenterCrop(64),
                                                transforms.ToTensor(),
                                                #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))
    else:
        dataset_loader = dataset_dict[dataset](
            root="./data",
            download=True,
            transform=transforms.Compose(colorspace_transforms),
        )
    data_loader = torch.utils.data.DataLoader(
        dataset_loader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    return data_loader, dataset_channels[dataset]


def load_data(
    dataset,
    colorspace,
    batch_size=64,
    train_prop=0.8,
    test_augmentation="none",
):
    colorspace_transforms = colorspaces[colorspace]
    test_augmentations = augmentations_dict[test_augmentation]

    if dataset == "mnist":
        colorspace_train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            ExpandColorDimension(),
            transforms.ToPILImage(),
        ] + colorspace_transforms
    elif dataset == "cifar":
        colorspace_train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ] + colorspace_transforms

    if dataset == "mnist":
        test_transforms = (
            [transforms.Pad(2, fill=0), transforms.ToTensor(), ExpandColorDimension()]
            + test_augmentations
            + [transforms.ToPILImage()]
            + colorspace_transforms
        )
    elif dataset == "cifar":
        test_transforms = (
            [transforms.ToTensor()]
            + test_augmentations
            + [transforms.ToPILImage()]
            + colorspace_transforms
        )

    train_set = dataset_dict[dataset](
        root="./data",
        train=True,
        transform=transforms.Compose(colorspace_train_transforms),
        download=True,
    )

    train_size = int(train_prop * len(train_set))
    valid_size = len(train_set) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_set, [train_size, valid_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_set = dataset_dict[dataset](
        root="./data",
        train=False,
        transform=transforms.Compose(test_transforms),
        download=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader, 3
