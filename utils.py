import logging
from typing import Union
from pathlib import Path


def setup_logger(output_file: Union[Path, str]):
    if isinstance(output_file, str):
        output_file = Path(output_file)

    logger = logging.root
    file_handler = logging.FileHandler(output_file, mode="w")
    stream_handler = logging.StreamHandler()

    logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
