import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import lmdb
from tqdm import tqdm

from data_tree import logger
from data_tree.key_value_store import KeyValueStore
from data_tree.util import Pickled


@dataclass
class LmdbBytesKvs(KeyValueStore):
    target_lmdb_path: str
    sort_keys: bool

    def __post_init__(self):
        assert os.path.exists(self.target_lmdb_path), f"lmdb target path does not exist at {self.target_lmdb_path}!"
        self._keys = self._cached_keys()

    @property
    def txn(self):
        return self.create_txn()

    @contextmanager
    def create_env(self,readonly=True):
        # map_size was 1e11
        # on lustre file system on abci you cannot use file lock. and you have to disable it for lmdb to work.
        try:
            with lmdb.open(self.target_lmdb_path, lock=False, readonly=readonly) as env:
                yield env
        except Exception as e:
            import os
            if not os.path.exists(self.target_lmdb_path):
                logger.error(f"target_lmdb_path does not exist!:{e}")
            raise e

    @contextmanager
    def create_txn(self,readonly=True):
        with self.create_env(readonly) as env:
            with env.begin(write=not readonly, buffers=True) as txn:
                yield txn

    def num_keys(self):
        with self.create_txn() as txn:
            return txn.stat()['entries']

    def _cached_keys(self) -> List[str]:
        cache_path = os.path.join(self.target_lmdb_path, 'keys.pkl')
        pkl_keys = Pickled(cache_path, self._retrieve_keys)
        keys = pkl_keys.value
        if len(keys) != self.num_keys():
            pkl_keys.clear()
        return pkl_keys.value

    def keys(self) -> List[str]:
        return self._keys

    def _retrieve_keys(self) -> List[str]:
        _keys = list(self.key_iterator())
        if self.sort_keys:
            logger.info(f"sorting image store keys...")
            _keys = list(sorted(_keys))
        return _keys

    def key_iterator(self):
        with self.create_txn() as txn:
            with txn.cursor() as cur:
                for t in tqdm(cur.iternext(values=False), desc="retrieving keys from lmdb"):
                    yield t.tobytes().decode()

    def __getitem__(self, item):
        with self.create_txn() as txn:
            buf = txn.get(item.encode())
            assert buf is not None  # why the heck this is None?
            return buf.tobytes()

    def __delitem__(self, key):
        with self.create_txn(readonly=False) as txn:
            txn.delete(key.encode())
