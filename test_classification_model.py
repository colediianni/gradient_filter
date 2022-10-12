from pathlib import Path

from test_cases import test


def test_model(
    base_path: Path, model_type, dataset_name, device, model_load_path: Path
):

    output_file = (
        base_path / "logs" / (model_type + "_" + dataset_name + ".txt")
    )

    test(network, test_loader, device)
