import logging
from pathlib import Path
from typing import Union
import time

import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision

from augmentations import augmentations_dict
from data import load_data, dataset_channels
from layers import EuclideanColorInvariantConv2d, LearnedColorInvariantConv2d, GrayscaleConv2d, GrayscaleEuclideanColorInvariantConv2d
from utils import setup_logger
from augmentations import Recolor
from lenet import LeNet
from custom_resnet_preprocessing import MyImageClassification
from torchvision.models import ResNet50_Weights


def get_classification_model(
    model_type,
    device,
    input_channels,
    load_from_path: Path = None,
):
    logging.info("==> Building model..")
    if model_type == "normal_resnet":
        weights = ResNet50_Weights.DEFAULT
        my_trans = MyImageClassification(crop_size=224,resize_size=232)
        weights.transforms = my_trans
        network = torchvision.models.resnet50(weights=weights)
        network.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_type == "euclidean_diff_ci_resnet":
        weights = ResNet50_Weights.DEFAULT
        my_trans = MyImageClassification(crop_size=224,resize_size=232)
        weights.transforms = my_trans
        network = torchvision.models.resnet50(weights=weights)
        network.conv1 = EuclideanColorInvariantConv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_type == "learned_diff_ci_resnet":
        weights = ResNet50_Weights.DEFAULT
        my_trans = MyImageClassification(crop_size=224,resize_size=232)
        weights.transforms = my_trans
        network = torchvision.models.resnet50(weights=weights)
        network.conv1 = LearnedColorInvariantConv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_type == "grayscale_normal_resnet":
        weights = ResNet50_Weights.DEFAULT
        my_trans = MyImageClassification(crop_size=224,resize_size=232)
        weights.transforms = my_trans
        network = torchvision.models.resnet50(weights=weights)
        network.conv1 = GrayscaleConv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_type == "grayscale_euclidean_diff_ci_resnet":
        weights = ResNet50_Weights.DEFAULT
        my_trans = MyImageClassification(crop_size=224,resize_size=232)
        weights.transforms = my_trans
        network = torchvision.models.resnet50(weights=weights)
        network.conv1 = GrayscaleEuclideanColorInvariantConv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_type == "normal_lenet" or model_type == "euclidean_diff_ci_lenet" or model_type == "learned_diff_ci_lenet":
        network = LeNet(model_type)


    network = torch.nn.DataParallel(network)
    network = network.to(device)
    if device == "cuda":
        cudnn.benchmark = True
    if load_from_path is not None:
        network.load_state_dict(
            torch.load(load_from_path, map_location=torch.device(device))
        )

    return network


def train_classification_model(
    network,
    train_loader,
    val_loader,
    device,
    model_save_path: Path,
    loss_plot_path: Path,
    epochs=100,
    lr=0.001,
):
    optimizer = optim.SGD(
        network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200
    )

    best_val_loss = torch.inf
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):

        total_correct = 0.0
        total_loss = 0.0
        network.train()
        for batch in train_loader:  # Get batch
            images, labels = batch  # Unpack the batch into images and labels
            images, labels = images.to(device), labels.to(device)

            preds = network(images)  # Pass batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss

            optimizer.zero_grad()
            # start = time.time()
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights
            # end = time.time()
            # print("backward", end - start)

            total_loss += loss.item() / len(train_loader)
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

        train_loss_history.append(total_loss)
        logging.info(
            (
                f"epoch: {epoch}",
                f"total_correct: {total_correct}",
                f"total_loss: {total_loss}",
            )
        )

        network.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0.0
            for batch in val_loader:
                (
                    images,
                    labels,
                ) = batch  # Unpack the batch into images and labels
                images, labels = images.to(device), labels.to(device)

                preds = network(images)  # Pass batch
                loss = F.cross_entropy(preds, labels)  # Calculate Loss
                val_loss += loss.item() / len(val_loader)
                val_correct += preds.argmax(dim=1).eq(labels).sum().item()

            if val_loss < best_val_loss:
                # logging.info(
                #     f"Saving model. New best validation loss: {val_loss}"
                # )
                logging.info(
                    (
                        "Saving model",
                        f"val_correct: {val_correct}",
                        f"val_loss: {val_loss}",
                    )
                )
                best_val_loss = val_loss
                torch.save(network.state_dict(), model_save_path)

            val_loss_history.append(val_loss)

        plt.plot(train_loss_history, "-b", label="train")
        plt.plot(val_loss_history, "-r", label="val")
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(loss_plot_path)  # , format="svg")
        plt.close()

        scheduler.step()

    logging.info(">>> Training Complete >>>")
    return network


def classification_training_pipeline(
    base_path: Union[Path, str],
    model_type,
    dataset_name,
    colorspace,
    device,
    batch_size=128,
    epochs=100,
    lr=0.001,
    model_load_path: Path = None,
):
    if not isinstance(base_path, Path):
        base_path = Path(base_path)

    if model_load_path is not None and isinstance(model_load_path, Path):
        model_load_path = Path(model_load_path)

    model_save_path = (
        base_path
        / "models"
        / (model_type + "_" + dataset_name + "_" + colorspace + ".pth")
    )
    output_file = (
        base_path
        / "logs"
        / (model_type + "_" + dataset_name + "_" + colorspace + ".txt")
    )

    setup_logger(output_file)

    # load datasets
    train_loader, val_loader, _test_loader, input_channels = load_data(
        dataset=dataset_name,
        colorspace=colorspace,
        batch_size=batch_size,
        train_prop=0.8,
    )

    # load model
    network = get_classification_model(
        model_type,
        device,
        input_channels,
        load_from_path=model_load_path,
    )

    loss_plot_path = (
        base_path
        / "images"
        / (model_type + "_" + dataset_name + "_" + colorspace)
    )
    # train model
    network = train_classification_model(
        network,
        train_loader,
        val_loader,
        device,
        model_save_path,
        loss_plot_path,
        epochs=epochs,
        lr=lr,
    )


@torch.no_grad()
def get_all_preds(model, loader, device):
    all_preds = torch.tensor([]).to(device)
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def classification_testing_pipeline(
    base_path: Path, model_type, dataset_name, colorspace, device
):

    if not isinstance(base_path, Path):
        base_path = Path(base_path)

    model_save_path = (
        base_path
        / "models"
        / (model_type + "_" + dataset_name + "_" + colorspace + ".pth")
    )
    output_file = (
        base_path
        / "logs"
        / (model_type + "_" + dataset_name + "_" + colorspace + ".txt")
    )

    print(model_save_path)

    setup_logger(output_file)

    logging.info("Now testing: %s in %s", model_type, colorspace)

    input_channels = dataset_channels[dataset_name]

    # load model
    network = get_classification_model(
        model_type,
        device,
        input_channels,
        load_from_path=model_save_path,
    )

    network.eval()
    network = network.to(device)

    total_preds = 0
    total_preds_correct = 0
    for augmentation in [
        "none",
        "gaussian_noise",
        "gaussian_blur",
        "color_jitter",
        "salt_and_pepper",
        # "per_pixel_channel_permutation",
        "channel_permutation",
        "invert",
        "hue_shift",
        "grayscale",
        # "recolor"
    ]:
        logging.info("with augmentation: %s", augmentation)

        # load dataset
        aug = augmentation
        if aug == "recolor":
            aug = "none"
        _train_loader, _val_loader, test_loader, _input_channels = load_data(
            dataset=dataset_name,
            colorspace=colorspace,
            batch_size=64,
            train_prop=0.8,
            test_augmentation=aug,
        )

        preds_correct = 0
        for batch in test_loader:
            images, labels = batch
            images = images.to(device)
            if augmentation == "recolor":
                images = Recolor()(images)

            labels = labels.to(device)

            test_preds = network(images)
            preds_correct += (
                test_preds.argmax(dim=1).eq(labels).sum().item()
            )

        logging.info("total correct: %s", preds_correct)
        logging.info("accuracy: %s", preds_correct / len(test_loader.dataset))

        total_preds_correct += preds_correct
        total_preds += len(test_loader.dataset)

    logging.info("overall accuracy: %s", total_preds_correct / total_preds)
