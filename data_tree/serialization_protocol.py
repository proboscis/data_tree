import io
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Callable, TypeVar

from PIL import Image

T = TypeVar("T")


class ISerializationProtocol(ABC):
    @abstractmethod
    def dump(self, data) -> bytes:
        pass

    @abstractmethod
    def load(self, data: bytes) -> Any:
        pass


@dataclass
class SerializationProtocol(Generic[T], ISerializationProtocol):
    _load: Callable[[bytes], T]
    _dump: Callable[[T], bytes]

    def dump(self, data) -> bytes:
        return self._dump(data)

    def load(self, data: bytes) -> T:
        return self._load(data)


