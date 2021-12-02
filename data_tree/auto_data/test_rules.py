from loguru import logger

from data_tree import auto, DEFAULT_RULES
from pprint import pformat
a = auto("image,RGB,RGB")(None)

def test_neighbors():
    logger.info(pformat(DEFAULT_RULES.neighbors("image,RGB,RGB")))

def test_image_to_widget():
    logger.info(a.converter("widget"))


