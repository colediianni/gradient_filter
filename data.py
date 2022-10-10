import torch, torchvision
from torchvision import datasets, transforms


dataset_dict = {"mnist":torchvision.datasets.MNIST,
                "cifar":torchvision.datasets.CIFAR10}

def load_data(dataset, batch_size=16, train_prop=0.8, training_gan=False):
    if training_gan:
        #loading the dataset
        dataset = dataset_dict[dataset](root="./data", download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(64),
                                       transforms.ToTensor(),
                                   ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=2)
        return dataloader
    else:
        transform =transforms.Compose([transforms.ToTensor()])

        train_set = dataset_dict[dataset](root = './data', train=True,  transform=transform, download=True)

        train_size = int(train_prop * len(train_set))
        valid_size = len(train_set) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_set, [train_size, valid_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle=True)

        test_set = dataset_dict[dataset](root = './data', train=False, transform=transform, download=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
