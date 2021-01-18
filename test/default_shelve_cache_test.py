import os

from loguru import logger

from data_tree.util import DefaultShelveCache


def test_default_shelve_cache():
    cache = DefaultShelveCache(
        lambda k: k,
        os.path.expanduser("~/.cache/test_dsc.shelve"))
    cache.clear()
    assert cache[(0,1)] == (0,1)
    assert (0,1) in cache
    logger.info(list(cache.items()))