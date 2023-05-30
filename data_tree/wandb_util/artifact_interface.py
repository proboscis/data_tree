import abc
from abc import abstractmethod


class IArtifact(metaclass=abc.ABCMeta):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def type(self):
        pass

    @property
    @abstractmethod
    def metadata(self):
        pass