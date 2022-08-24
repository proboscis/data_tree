import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import List, Callable

import lmdb
from lmdb import MapFullError
from loguru import logger
from returns.result import safe
from tqdm import tqdm

from data_tree.serialization_protocol import ISerializationProtocol


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
class LmdbHelper:
    target_lmdb_path: str
    max_size_in_bytes: int
    write_map: bool
    key_protocol: ISerializationProtocol
    value_protocol: ISerializationProtocol

    def create_if_nonexistent(self):
        logger.info(f"checking if lmdb exists...")
        if not os.path.exists(self.target_lmdb_path):
            logger.info(f"lmdb does not exist at {self.target_lmdb_path}. creating...")
            with self.write_transaction():
                pass
        else:
            logger.info(f"lmdb already exists at {self.target_lmdb_path}")

    def __post_init__(self):
        assert isinstance(self.target_lmdb_path,str)
        self.create_if_nonexistent()

    @contextmanager
    def write_env(self):
        with lmdb.open(self.target_lmdb_path, writemap=self.write_map,
                       map_size=self.max_size_in_bytes) as env:
            yield env

    @contextmanager
    def read_env(self):
        # map_size was 1e11
        # on lustre file system on abci you cannot use file lock. and you have to disable it for lmdb to work.
        with lmdb.open(self.target_lmdb_path, lock=False, readonly=True) as env:
            yield env

    @contextmanager
    def write_transaction(self):
        with self.write_env() as env:
            with env.begin(buffers=False, write=True) as txn:
                yield txn

    @contextmanager
    def read_transaction(self):
        with self.read_env() as env:
            with env.begin(write=False, buffers=True) as txn:
                yield txn

    def write_until_full(self, pairs):
        def continued(key, data):
            return chain([(key, data)], pairs)

        def get_used_size(stat):
            return 4096 * (stat['branch_pages'] + stat['leaf_pages'] + stat['overflow_pages'] + 2)

        for i, (key, data) in enumerate(pairs):
            try:
                with self.write_transaction() as txn:
                    key = self.key_protocol.dump(key)
                    data = self.value_protocol.dump(data)
                    assert isinstance(key,bytes)
                    assert isinstance(data,bytes)
                    pages = len(data) // 4096 + 1
                    to_use = pages * 4096 * 2  # for safety we multiply by 2
                    if (get_used_size(txn.stat()) + to_use) < self.max_size_in_bytes:
                        txn.put(key, data)
                    else:
                        return continued(key, data)  # return unfinished iterator to be used again
            except MapFullError as mfe:  # if you hit this you are done .
                logger.error(
                    f'reached max size of a db. written={get_used_size(txn.stat())}, limit={self.max_size_in_bytes}')
                raise mfe
            except Exception as e:
                logger.error(f"unknown exception: {e}")
                raise e

        return []

    def __getitem__(self, key):
        with self.read_transaction() as txn:
            return self.value_protocol.load(txn.get(self.value_protocol.dump(key)).tobytes())

    def __setitem__(self, key, value):
        assert value is not None, "you cannot set None as a value for LMDB"
        key = self.key_protocol.dump(key)
        value = self.value_protocol.dump(value)
        with self.write_transaction() as txn:
            try:
                txn.put(key, value)
            except MapFullError as mfe:
                raise KVSFullError() from mfe

    def key_iterator(self):
        with self.read_transaction() as txn:
            with txn.cursor() as cur:
                for t in tqdm(cur.iternext(values=False), desc="iterating all keys from lmdb"):
                    yield self.key_protocol.load(t.tobytes())

    def keys(self) -> List[str]:
        try:
            return list(self.key_iterator())
        except lmdb.Error as e:
            logger.warning(f"empty keys are returned since lmdb was not initialized")
            return []

    def __contains__(self, key):
        return self[key] is not None


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


class KVSFullError(Exception):
    pass


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