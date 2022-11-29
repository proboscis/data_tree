import os
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import List

import lmdb
from lmdb import MapFullError
from tqdm import tqdm

from data_tree import logger
from data_tree.kvc_errors import KVSFullError
from data_tree.serialization_protocol import ISerializationProtocol


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
