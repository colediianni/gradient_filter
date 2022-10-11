import torch, torchvision
from torchvision import datasets, transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from layers import RGBColorInvariantConv2d
from data import load_data

def get_classification_model(model_type, device, input_channels, output_file, load_from_path=""):
    print('==> Building model..')
    with open(output_file, 'a') as the_file:
        the_file.write('==> Building model..')
    if model_type == "normal_resnet":
        network = torchvision.models.resnet50(weights=None)
        network.conv1 = torch.nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif model_type == "rgb_ci_resnet":
        network = torchvision.models.resnet50(weights=None)
        network.conv1 = RGBColorInvariantConv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    network = network.to(device)
    if device == 'cuda':
        network = torch.nn.DataParallel(network)
        cudnn.benchmark = True
    if load_from_path != "":
        network.load_state_dict(torch.load(load_from_path))
    return network


def train_classification_model(network, train_loader, val_loader, device, model_save_path, output_file, loss_plot_path, epochs=100, lr=0.001):
    optimizer = optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    best_val_loss = torch.inf
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):

        total_correct = 0
        total_loss = 0
        for batch in train_loader: #Get batch
            images, labels = batch #Unpack the batch into images and labels
            images, labels = images.to(device), labels.to(device)

            preds = network(images) #Pass batch
            loss = F.cross_entropy(preds, labels) #Calculate Loss

            optimizer.zero_grad()
            loss.backward() #Calculate gradients
            optimizer.step() #Update weights

            total_loss += loss.item()/len(train_loader)
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()

        train_loss_history.append(total_loss)
        print('epoch:', epoch, "total_correct:", total_correct, "loss:", total_loss)
        with open(output_file, 'a') as the_file:
            the_file.write('epoch:', epoch, "total_correct:", total_correct, "loss:", total_loss)

        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                images, labels = batch #Unpack the batch into images and labels
                images, labels = images.to(device), labels.to(device)

                preds = network(images) #Pass batch
                loss = F.cross_entropy(preds, labels) #Calculate Loss
                val_loss += loss.item()/len(val_loader)

            if val_loss < best_val_loss:
                print("Saving model. New best validation loss: ", val_loss)
                with open(output_file, 'a') as the_file:
                    the_file.write("Saving model. New best validation loss: ", val_loss)
                best_val_loss = val_loss
                torch.save(network.state_dict(), model_save_path)

            val_loss_history.append(val_loss)

        # TODO: ADD AXES, LEGEND, AND TITLE!!
        plt.plot(train_loss_history)
        plt.plot(val_loss_history)
        plt.savefig(loss_plot_path)
        plt.close()

    print('>>> Training Complete >>>')
    with open(output_file, 'a') as the_file:
        the_file.write('>>> Training Complete >>>')
    return network


def classification_training_pipeline(base_path, model_type, dataset_name, device, epochs=100, lr=0.001, model_load_path=""):

    model_save_path = base_path+"/models/"+model_type+"_"+dataset_name+".pth"
    output_file = base_path+"/logs/"+model_type+"_"+dataset_name+".txt"

    # load datasets
    train_loader, val_loader, test_loader, input_channels = load_data(dataset=dataset_name, batch_size=16, train_prop=0.8, training_gan=False)

    # load model
    network = get_classification_model(model_type, device, input_channels, output_file, load_from_path=model_load_path)

    loss_plot_path = base_path + "/images/"+model_type+"_"+dataset_name
    # train model
    network = train_classification_model(network, train_loader, val_loader, device, model_save_path, output_file, loss_plot_path, epochs=epochs, lr=lr)
