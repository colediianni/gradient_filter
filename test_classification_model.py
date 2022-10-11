



def test_model(base_path, model_type, dataset_name, device, model_load_path):

    output_file = base_path+"/logs/"+model_type+"_"+dataset_name+".txt"

    test(network, test_loader, device, output_file)
