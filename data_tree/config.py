import yaml
from lazy_object_proxy import Proxy

from easydict import EasyDict as edict

default_config = edict(
    visdom=edict(
        server="localhost",
        port=8097,
        username=None,
        password=None
    )
)
CONFIG = None


def get_config():
    import os
    config_dir = os.path.expanduser("~/.data_tree/config.yml")
    if os.path.exists(config_dir):
        with open(config_dir) as f:
            config = yaml.load(f)
        res = default_config.copy()
        res.update(config)
        return edict(res)
    else:
        from data_tree import logger
        logger.warning(f"data_tree config not found at {config_dir}. using default configs.")
        return default_config.copy()


def reload_config():
    global CONFIG
    logger.info(f"loading configuration file")
    CONFIG = Proxy(get_config)
    logger.debug(f"loaded config:{CONFIG}")

reload_config()
