from data import load_data
from model import get_classification_model, train_classification_model

def classification_training_pipeline(model_type, dataset_name, device, model_save_path, epochs=100, lr=0.001, model_load_path=""):
    # load datasets
    train_loader, val_loader, test_loader = load_data(dataset=dataset_name, batch_size=16, train_prop=0.8, training_gan=False)

    # load model
    network = get_classification_model(model_type, device, load_from_path=model_load_path)

    # train model
    network = train_classification_model(network, model_save_path, epochs=epochs, lr=lr)
