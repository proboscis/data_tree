import os
from typing import List


from tqdm import tqdm

from data_tree import logger
from data_tree.key_value_cache import IFileBasedCache
from data_tree.util import Pickled


class DiskCache(IFileBasedCache):

    def __setitem__(self, key, value):
        self.cache[key] = value

    def __init__(self, path):
        import diskcache as dc
        self._path = path
        self.cache = dc.Cache(path)

    def path(self):
        return self._path

    def keys(self) -> List[str]:
        logger.info(f"retrieving all keys from disk cache")
        return list(tqdm(self.cache.iterkeys(), desc="retrieving all keys from disk cache"))

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        return self.cache[key]


class StaticDiskCache(DiskCache):
    def __init__(self, path):
        super().__init__(path)
        get_keys = super().keys
        self.key_cache = Pickled(os.path.join(path, "key_cache.pkl"), get_keys)

    def keys(self):
        return self.key_cache.value

    def __setitem__(self, key, value):
        raise RuntimeError("set item is not supported for StaticDiskCache")