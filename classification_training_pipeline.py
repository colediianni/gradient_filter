from data import load_data
from model import get_classification_model, train_classification_model

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
