import logging
import os
from config import REPORTS_DIR

os.makedirs(REPORTS_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # file handler
    file_handler = logging.FileHandler(os.path.join(REPORTS_DIR, 'pipeline.log'))
    file_handler.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger