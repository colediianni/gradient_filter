import logging
from pathlib import Path

import matplotlib.pyplot as plt
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision

from data import load_data
from layers import EuclideanColorInvariantConv2d, LearnedColorInvariantConv2d
from test_cases import test


def get_classification_model(
    model_type,
    device,
    input_channels,
    load_from_path: Path = None,
):
    logging.info("==> Building model..")
    if model_type == "normal_resnet":
        network = torchvision.models.resnet50(weights=None)
        network.conv1 = torch.nn.Conv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_type == "euclidean_diff_ci_resnet":
        network = torchvision.models.resnet50(weights=None)
        network.conv1 = EuclideanColorInvariantConv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
    elif model_type == "learned_diff_ci_resnet":
        network = torchvision.models.resnet50(weights=None)
        network.conv1 = LearnedColorInvariantConv2d(
            input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    network = network.to(device)
    if device == "cuda":
        network = torch.nn.DataParallel(network)
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
    optimizer = optim.SGD(net.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

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
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

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
            total_correct = 0.0
            for batch in val_loader:
                (
                    images,
                    labels,
                ) = batch  # Unpack the batch into images and labels
                images, labels = images.to(device), labels.to(device)

                preds = network(images)  # Pass batch
                loss = F.cross_entropy(preds, labels)  # Calculate Loss
                val_loss += loss.item() / len(val_loader)
                total_correct += preds.argmax(dim=1).eq(labels).sum().item()

            if val_loss < best_val_loss:
                logging.info(
                    f"Saving model. New best validation loss: {val_loss}"
                )
                logging.info(
                    (
                        f"Saving model",
                        f"total_correct: {total_correct}",
                        f"total_loss: {val_loss}",
                    )
                )
                best_val_loss = val_loss
                torch.save(network.state_dict(), model_save_path)

            val_loss_history.append(val_loss)

        plt.plot(train_loss_history, "-b", label="train")
        plt.plot(val_loss_history, "-r", label="val")
        plt.title("Loss Histroy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(loss_plot_path)#, format="svg")
        plt.close()

        scheduler.step()

    logging.info(">>> Training Complete >>>")
    return network


def classification_training_pipeline(
    base_path: Path,
    model_type,
    dataset_name,
    colorspace,
    device,
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

    logger = logging.root
    file_handler = logging.FileHandler(output_file, mode="w")
    stream_handler = logging.StreamHandler()

    logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # load datasets
    train_loader, val_loader, test_loader, input_channels = load_data(
        dataset=dataset_name,
        colorspace=colorspace,
        batch_size=64,
        train_prop=0.8,
        training_gan=False,
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


def classification_testing_pipeline(
    base_path, model_type, dataset_name, device, model_load_path
):
    # empty/create new loss_plot_path file

    test(network, test_loader, device)
