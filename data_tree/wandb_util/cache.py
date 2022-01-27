import hashlib
import pickle
from abc import ABCMeta, abstractmethod, ABC
from dataclasses import dataclass
from typing import Generic, Callable

from loguru import logger

from data_tree.image_store import T
from data_tree.util import Pickled


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
    @abstractmethod
    def value(self, f):
        pass

    @abstractmethod
    def clear(self):
        pass


class ICacheFactory(ABC):
    @abstractmethod
    def get(self, f) -> ICache:
        pass


class KeyedCacheFactory(metaclass=ABCMeta):
    @abstractmethod
    def cache(self, key_obj, f: Callable[[], T], identifier: str) -> IRead[T]:
        pass


class PickledCacheFactory(KeyedCacheFactory):
    def cache(self, key_obj, f: Callable[[], T], identifier: str) -> IRead[T]:
        return PickledCache(key_obj, f, identifier)


@dataclass
class PickleCacheFactory(ICacheFactory):
    pkl_path: str

    def get(self, f) -> ICache:
        return Pickled(self.pkl_path, proc=f)


@dataclass
class PickledCache(IRead):
    key_obj: object
    f: Callable[[], T]
    identifier: str

    def __post_init__(self):
        self._hash = hashlib.md5(pickle.dumps(self.key_obj))

        def proc():
            data = self.f()
            return dict(
                data=data,
                hash=self._hash,
                key_obj=self.key_obj
            )

        self.cache = Pickled(path=self.identifier + ".cache.pkl", proc=proc)

    @property
    def value(self):
        cached_hash = self.cache.value["hash"]
        if cached_hash != self._hash:
            logger.warning(
                f"invalidating since cached key_obj == {self.cache.value['key_obj']}, while given key is {self.key_obj}")
            self.cache.clear()

        return self.cache.value["data"]


@dataclass
class LambdaReadFacory(ReadFacory):
    src_function: Callable

    def get(self, f) -> IRead:
        return self.src_function(f)
