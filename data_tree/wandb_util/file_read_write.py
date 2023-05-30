import abc
import pickle
from typing import Generic, TypeVar

import cloudpickle
import pandas as pd
import torch

T = TypeVar("T")


class FileRead(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self, file: str) -> T:
        pass


class FileWrite(Generic[T], abc.ABC):
    @abc.abstractmethod
    def write(self, object: T, file: str):
        """needs to be a single file!"""
        pass


class AsyncFileRead(Generic[T], abc.ABC):
    @abc.abstractmethod
    async def aread(self, file: str) -> T:
        pass


class AsyncFileWrite(Generic[T], abc.ABC):
    @abc.abstractmethod
    async def awrite(self, object: T, file: str):
        pass


class PickleRead(FileRead[T]):
    def read(self, file: str) -> T:
        with open(file, 'rb') as f:
            return pickle.load(f)


class CloudPickleRead(FileRead[T]):
    def read(self, file: str) -> T:
        with open(file, 'rb') as f:
            return cloudpickle.load(f)


class TorchRead(FileRead[T]):
    def read(self, file: str) -> T:
        with open(file, 'rb') as f:
            return torch.load(f)


class Hdf5DataFrameRead(FileRead[pd.DataFrame]):
    def read(self, file: str) -> pd.DataFrame:
        return pd.read_hdf(file, key="data")


class Hdf5DataFrameWrite(FileWrite[pd.DataFrame]):
    def write(self, object: pd.DataFrame, file: str):
        return object.to_hdf(file, key="data", )
