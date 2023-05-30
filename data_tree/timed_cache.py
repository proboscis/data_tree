import calendar
import os
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import List, Callable


from tqdm import tqdm

from data_tree.util import Pickled
from data_tree.wandb_util.cache import ICacheFactory, ICache


@dataclass
class FileTimeCache:
    """
    watch target files and call f if anything was change when calling "get"
    """
    target_files: List[Path]
    cache_path: str
    f: Callable

    def __post_init__(self):
        def wrapped():
            t = calendar.timegm(time.gmtime())
            return t, self.f()

        self.wrapped = wrapped
        self.pkled = Pickled(self.cache_path, self.wrapped)
        #logger.warning(f"file time cache targets:{pformat(sorted(self.target_files))}")

    def modified_times(self):
        from loguru import logger
        logger.info(f"checking modified times of {len(self.target_files)} files...")
        return [os.path.getmtime(str(p)) for p in tqdm(self.target_files, desc="examining modified times...")]

    def latest_modification(self):
        return max(self.modified_times())

    def get(self):
        # callee = get_callee()
        # logger.info(f"get called from {callee}")
        t, data = self.pkled.value
        if self.latest_modification() > t:
            from loguru import logger
            logger.warning(f"src file changed, invalidating cache")
            self.pkled.clear()
        return self.pkled.value[1]

    def clear(self):
        self.pkled.clear()


@dataclass
class FileTimeCacheImpl(ICache):
    cache_factory: ICacheFactory
    files_to_watch: List[str]
    f: Callable

    def __post_init__(self):
        def wrapped():
            t = calendar.timegm(time.gmtime())
            return t, self.f()

        self.wrapped = wrapped
        self.cache = self.cache_factory.get(self.wrapped)

    def modified_times(self):
        return [os.path.getmtime(str(p)) for p in tqdm(self.files_to_watch, desc="examining modified times...")]

    def latest_modification(self):
        return max(self.modified_times())

    def get(self):
        t, data = self.cache.value
        if self.latest_modification() > t:
            from loguru import logger
            logger.warning(f"src file changed, invalidating cache")
            self.cache.clear()
        return self.cache.value[1]

    @property
    def value(self):
        return self.get()

    def clear(self):
        self.cache.clear()


@dataclass
class OneTimeFileTimeCacheFactory(ICacheFactory):
    files_to_watch: List[str]
    cache_factory: ICacheFactory

    def get(self, f):
        return FileTimeCacheImpl(self.cache_factory, self.files_to_watch, f)
