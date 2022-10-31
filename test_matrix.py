from model import classification_testing_pipeline


if __name__ == "__main__":
    dataset_name = "cifar"

    for model_type in [
        "euclidean_diff_ci_resnet",
        "learned_diff_ci_resnet",
        "normal_resnet",
    ]:
        for colorspace in ["hsv", "lab", "rgb", "xyz"]:
            classification_testing_pipeline(
                ".", model_type, dataset_name, colorspace, "cpu"
            )
