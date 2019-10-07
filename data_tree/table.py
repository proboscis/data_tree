import abc
from typing import Mapping

from frozendict import frozendict
from lazy import lazy as en_lazy

import pandas
import numpy as np

from data_tree import Series


class Table(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def get_series(self, key):
        pass

    @en_lazy
    def series(self):
        return frozendict({k: self.get_series(k) for k in self.keys})

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.get_series(item)
        elif isinstance(item, slice) or isinstance(item, list) or isinstance(item, np.ndarray):
            return ConvertedTable(self, {k: v[item] for k, v in self.series.items()})
        elif isinstance(item, int):
            return frozendict({k: v[item] for k, v in self.series.items()})

    def map_series(self, key: str, mapper):
        series = {k: self.get_series(k) for k in self.keys}
        series[key] = mapper(series[key])
        return ConvertedTable(self, series)

    def to_df(self):
        return pandas.DataFrame({
            col_name: col.values for col_name, col in self.series.items()
        })
    def add_column(self,key:str,series:Series):
        return ConvertedTable(self,dict(**{key:series},**self.series))


class ConvertedTable(Table):
    @en_lazy
    def keys(self):
        return list(self.mapped_series.keys())

    def get_series(self, key):
        return self.mapped_series[key]

    def __init__(self, src: Table, mapped_series: dict):
        self.src = src
        self.mapped_series = mapped_series


class SeriesTable(Table):
    """
    dict of series
    """

    @property
    def keys(self):
        return list(self._columns.keys())

    def get_series(self, key):
        return self._columns[key]

    def __init__(self, columns: Mapping[str, Series]):
        self._columns: Mapping[str, Series] = dict(**columns)
