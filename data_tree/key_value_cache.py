import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Callable

from returns.result import safe

from data_tree.kvc_errors import KVSFullError
from data_tree.lmdb_helper import LmdbHelper


class IKeyValueCache(ABC):
    @abstractmethod
    def keys(self) -> List[str]:
        pass

    @abstractmethod
    def __contains__(self, key):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    def write_all(self, pairs):
        for k, v in pairs:
            self[k] = v


class IFileBasedCache(IKeyValueCache):
    @abstractmethod
    def path(self):
        pass


@dataclass
class LmdbKvs(IFileBasedCache):
    lmdb_helper: LmdbHelper

    def keys(self) -> List[str]:
        return self.lmdb_helper.keys()

    def __contains__(self, key):
        return key in self.lmdb_helper

    def __getitem__(self, key):
        return self.lmdb_helper[key]

    def __setitem__(self, key, value):
        assert value is not None, "you cannot set None as a value for LMDB"
        self.lmdb_helper[key] = value

    def path(self):
        return self.lmdb_helper.target_lmdb_path

    def write_all(self, pairs):
        pairs = self.lmdb_helper.write_until_full(pairs)
        for item in pairs:
            raise RuntimeError("could not write all item to this lmdb..")


@dataclass
class MultiKvs(IFileBasedCache):
    root_dir: str
    kvs_factory: Callable[[str], IFileBasedCache]

    # kvs_factory is supposed to load the db at specified path ,or craete if it doesnt exist.

    def __post_init__(self):
        """
        find all dbs in root_dir.
        they are located as dirs each named 0 to N,
        :return:
        """
        self.dbs = dict()
        for entry in os.scandir(self.root_dir):
            entry: os.DirEntry
            if safe(int)(entry.name).value_or(None) is not None:
                self.dbs[entry.name] = self.kvs_factory(entry.path)

        self.current_db_name = self._get_youngest_db_name()
        self.key_db_mapping = self._get_key_db_mapping()

    def _get_key_db_mapping(self):
        mapping = dict()
        for name, db in self.dbs.items():
            for k in db.keys():
                assert k not in mapping, "db key %s already exists! overlapping!"
                mapping[k] = name
        return mapping

    def _get_youngest_db_name(self):
        if self.dbs:
            oldest = max([int(k) for k in self.dbs.keys()])
            return str(oldest)
        else:
            return self._get_new_db_name()

    def _get_new_db_name(self):
        if self.dbs:
            oldest = max([int(k) for k in self.dbs.keys()])
            next_name = str(oldest + 1)
        else:
            next_name = str(0)
        os.makedirs(os.path.join(self.root_dir, next_name), exist_ok=True)
        self.dbs[next_name] = self.kvs_factory(next_name)
        return next_name

    def path(self):
        return self.root_dir

    def keys(self) -> List[str]:
        return list(self.key_db_mapping.keys())

    def __contains__(self, key):
        return key in self.key_db_mapping

    def __getitem__(self, key):
        return self.dbs[self.key_db_mapping[key]][key]

    def __setitem__(self, key, value):
        db = self.dbs[self.current_db_name]
        try:
            db[key] = value
        except KVSFullError as e:
            self.current_db_name = self._get_new_db_name()
            db = self.dbs[self.current_db_name]
            db[key] = value

