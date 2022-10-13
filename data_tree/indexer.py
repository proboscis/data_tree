import abc
from abc import ABC
from typing import Union, Iterable

import numpy as np



class Indexer(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def total(self) -> int:
        pass

    @abc.abstractmethod
    def get_index(self, index):
        pass

    @abc.abstractmethod
    def get_slice(self, _slice):
        pass

    @abc.abstractmethod
    def get_indices(self, indices):
        pass

    @abc.abstractmethod
    def get_mask(self, mask):
        pass

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.get_slice(item)
        elif isinstance(item, np.ndarray):
            if item.dtype == np.bool:
                return self.get_mask(item)
            else:
                return self.get_indices(item)
        elif isinstance(item, Iterable):
            return self[np.array(item)]
        else:
            return self.get_index(item)

    def slice(self, _slice):
        return SlicedIndexer(self, _slice)

    def transform(self, mapping: Union[list, np.ndarray]):
        return TransformedIndexer(self, mapping)

    def mask(self, mask):
        return TransformedIndexer(self, np.arange(self.total)[mask])

    def __len__(self):
        return self.total


class IdentityIndexer(Indexer):
    def get_index(self, index):
        return index

    def get_slice(self, _slice):
        return _slice

    def get_indices(self, indices):
        return indices

    def get_mask(self, mask):
        return mask

    def __init__(self, total: int):
        self._total = total

    @property
    def total(self) -> int:
        return self._total


class SourcedIndexer(Indexer):
    def __init__(self, src: Indexer):
        self.src = src


class SlicedIndexer(SourcedIndexer):

    @property
    def total(self) -> int:
        return self.src_stop - self.src_start

    def _parse_slice(self, start, end, total):
        return slice(start, end).indices(total)[:2]

    def __init__(self, src: Indexer, _slice: slice):
        super().__init__(src)
        self.src = src
        self.src_start, self.src_stop = self._parse_slice(_slice.start, _slice.stop, src.total)

    def _src_index(self, index):
        assert -self.total + 1 <= index <= self.total, f"index out of bounds:{index} for length {self.total}"
        if index < 0:
            new_index = self.src_stop + index
        else:
            new_index = self.src_start + index
        assert self.src_start <= new_index <= self.src_stop, f"index out of bounds:{index}->{(self.src_start, new_index, self.src_stop)}"
        return new_index

    def get_index(self, index: int):
        assert index < self.total, f"index out of bounds:{index} for length {self.total}"
        return self.src.get_index(self._src_index(index))

    def get_slice(self, _slice):
        local_start, local_stop = self._parse_slice(_slice.start, _slice.stop, self.total)
        src_start = self._src_index(local_start)
        src_stop = self._src_index(local_stop)
        return self.src.get_slice(slice(src_start, src_stop))

    def get_indices(self, indices):
        return self.src.get_indices(indices + self.src_start)

    def get_mask(self, mask):
        assert len(mask) == self.total, "mask must have exact same length as this indexer's total"
        total = self.total
        local_indices = np.arange(total)[mask]
        return self.get_indices(local_indices)


class TransformedIndexer(SourcedIndexer):
    @property
    def total(self) -> int:
        return len(self.mapping)

    def __init__(self, src: Indexer, mapping: Union[list, np.ndarray]):
        super().__init__(src)
        self.mapping = mapping

    def get_index(self, index):
        return self.src.get_index(self.mapping[index])

    def get_slice(self, _slice):
        return self.src.get_indices(self.mapping[_slice])

    def get_indices(self, indices):
        return self.src.get_indices(self.mapping[indices])

    def get_mask(self, mask):
        indices = self.mapping[np.arange(len(mask))][mask]
        return self.get_indices(indices)
