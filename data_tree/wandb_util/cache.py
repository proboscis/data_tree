import datetime
import hashlib
import os
import pickle
from abc import ABCMeta, abstractmethod, ABC
from dataclasses import dataclass
from typing import Generic, Callable


from pandas import Timedelta

from data_tree import storage_manager, logger
from data_tree.image_store import T
from data_tree.storage_manager import FileStorageManager
from data_tree.util import Pickled
from pinjected.di.injected import Injected


class IRead(Generic[T], metaclass=ABCMeta):
    @property
    @abstractmethod
    def value(self):
        pass


class ReadFacory(metaclass=ABCMeta):
    @abstractmethod
    def get(self, f) -> IRead:
        pass


class ICache(Generic[T], ABC):
    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def clear(self):
        pass


class ICacheFactory(ABC):
    @abstractmethod
    def get(self, f, identifier) -> ICache:
        pass


@dataclass
class PickleCacheFactory(ICacheFactory):
    pickle_cache_root_dir: str

    def get(self, f, identifier: str) -> ICache:
        return PickledCache(os.path.join(self.pickle_cache_root_dir, identifier) + ".pkl", f=f)


@dataclass
class PickledCache(ICache):
    pkl_path: str
    f: Callable[[], T]

    def __post_init__(self):
        self.cache = Pickled(path=self.pkl_path, proc=self.f)

    @property
    def value(self):
        return self.cache.value

    def clear(self):
        return self.cache.clear()


@dataclass
class LambdaReadFacory(ReadFacory):
    src_function: Callable

    def get(self, f) -> IRead:
        return self.src_function(f)


@dataclass
class TimedCache(ICache):
    cache: ICache
    valid_duration: datetime.timedelta

    @property
    def value(self):
        val, time = self.cache.value
        now = datetime.datetime.now()
        if now - time > self.valid_duration:
            logger.warning(f"invalidating a cache:{self.cache} due to expiration:{self.valid_duration} ")
            self.cache.clear()
        return self.cache.value[0]

    def clear(self):
        self.cache.clear()


@dataclass
class TimedCacheFactory:
    cache_factory: ICacheFactory

    def get(self, f: Callable, identifier: str, duration: datetime.timedelta) -> ICache:
        def proc():
            res = f()
            time = datetime.datetime.now()
            return res, time

        cache = self.cache_factory.get(proc, identifier)
        return TimedCache(cache, duration)


TimedPickleCacheFactory = Injected.bind(
    TimedCacheFactory,
    cache_factory=Injected.bind(PickleCacheFactory)
)
