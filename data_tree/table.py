import abc
from typing import Mapping, List

from frozendict import frozendict
from lazy import lazy as en_lazy

import pandas
import numpy as np

from data_tree._series import Series


class Tagged:
    def __init__(self, value, **tags):
        self.value = value
        self.tags = tags

    def map(self, f):
        return Tagged(value=self.value.map(f), **self.tags)

    def all(self, **conditions):
        for k, v in conditions.items():
            if k not in self.tags or self.tags[k] != v:
                return False
        return True


class TaggedSpace:
    def __init__(self, *items: Tagged):
        self.items = items

    def split(self, **conditions):
        positives = []
        negatives = []
        for item in self.items:
            if item.all(**conditions):
                positives.append(item)
            else:
                negatives.append(item)
        return TaggedSpace(*positives), TaggedSpace(*negatives)

    def __add__(self, other):
        return TaggedSpace(*self.items, *other.items)

    def map(self, f, **conditions):
        tgts, others = self.split(**conditions)
        return TaggedSpace(*[item.map(f) for item in self.items]) + others


class TableBase(metaclass=abc.ABCMeta):
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
            return Table(**{k: v[item] for k, v in self.series.items()})
        elif isinstance(item, int):
            return frozendict({k: v[item] for k, v in self.series.items()})

    def map(self, key: str, mapper):
        series = {k: self.get_series(k) for k in self.keys}
        series[key] = series[key].map(mapper)
        return Table(**series)

    def to_df(self):
        return pandas.DataFrame({
            col_name: col.values for col_name, col in self.series.items()
        })

    def add_column(self, key: str, series: Series):
        return Table(**{key: series}, **self.series)

    def rename(self, key, new_name):
        new_series = self.series.copy()
        del new_series[key]
        new_series[new_name] = self.series[key]
        return Table(**new_series)


class Table(TableBase):
    """
    dict of series
    """

    @property
    def keys(self):
        return list(self._columns.keys())

    def get_series(self, key):
        return self._columns[key]

    def __init__(self, **columns):
        self._columns: Mapping[str, Series] = columns

    def __getitem__(self, item):
        return self._columns[item]

    def __add__(self, other: "Table"):
        assert len(set(self.keys) & set(
            other.keys)) == 0, "both table must have distinct names for each columns.no overwrapping"
        return Table(**self._columns, **other._columns)


class Tables:
    def __init__(self, **tables):
        self.tables = tables

    def map(self, col, f):
        return Tables(**{k: t.map(col, f) for k, t in self.tables.items()})

    def zipped(self, new_col_name, a, b):
        return Tables(
            **{tn: Table(**{new_col_name: self.tables[tn][a].zip(self.tables[tn][b])}) for tn in self.tables.keys()})

    def __getitem__(self, item):
        return self.tables[item]

    def __add__(self, other: "Tables"):
        assert set(other.tables.keys()) == set(self.tables.keys()), "added tables must have same set of tables"
        return Tables(**{k: (self[k] + other[k]) for k in self.tables.keys()})


def from_df(df: pandas.DataFrame):
    keys = list(df.keys())
    return Table({k: df[k].values for k in keys})
