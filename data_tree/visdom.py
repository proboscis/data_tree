from lazy_object_proxy import Proxy

from data_tree.config import CONFIG
from loguru import logger


def get_visdom():
    logger.error(f"visdom is created!")
    import visdom

    return visdom.Visdom(**CONFIG.visdom)


VISDOM = Proxy(get_visdom)
