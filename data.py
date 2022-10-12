from pathlib import Path

import torch
from torchvision import datasets, transforms

dataset_dict = {
    "mnist": datasets.MNIST,
    "cifar": datasets.CIFAR10,
}
dataset_channels = {"mnist": 1, "cifar": 3}

dataset_root = Path.cwd() / "data"


def load_data(dataset, batch_size=16, train_prop=0.8, training_gan=False):
    if training_gan:
        # loading the dataset
        dataset_loader = dataset_dict[dataset](
            root=dataset_root,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(64),
                    transforms.ToTensor(),
                ]
            ),
        )
        data_loader = torch.utils.data.DataLoader(
            dataset_loader, batch_size=batch_size, shuffle=True, num_workers=2
        )
        return data_loader, dataset_channels[dataset]
    else:
        transform = transforms.Compose([transforms.ToTensor()])

        train_set = dataset_dict[dataset](
            root=dataset_root, train=True, transform=transform, download=True
        )

        train_size = int(train_prop * len(train_set))
        valid_size = len(train_set) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_set, [train_size, valid_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=True
        )

        test_set = dataset_dict[dataset](
            root=dataset_root, train=False, transform=transform, download=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader, dataset_channels[dataset]
