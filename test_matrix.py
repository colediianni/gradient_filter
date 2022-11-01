from torch import cuda

from model import classification_testing_pipeline

if __name__ == "__main__":
    dataset_name = "cifar"
    device = "cuda" if cuda.is_available() else "cpu"

    for model_type in [
        "euclidean_diff_ci_resnet",
        "learned_diff_ci_resnet",
        "normal_resnet",
    ]:
        for colorspace in ["hsv", "lab", "rgb", "xyz"]:
            classification_testing_pipeline(
                ".", model_type, dataset_name, colorspace, device
            )
