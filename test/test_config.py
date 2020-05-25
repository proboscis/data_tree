from data_tree.config import CONFIG
from loguru import logger
def test_load_config():
    logger.info(f"load_config:{CONFIG}")
