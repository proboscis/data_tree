import abc
import os
from dataclasses import dataclass
from itertools import chain
from queue import Queue
from threading import Thread
from typing import Tuple, Iterable, Union, Optional

import lmdb as lmdb
import ray
from lmdb import MapFullError

from tqdm import tqdm

from data_tree import logger
from data_tree.util import ensure_path_exists


class DataStoreCreator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create(self, key_data_pairs: Tuple[str, bytes], dst: str, size_in_bytes: int):
        pass


def get_used_size(stat):
    return 4096 * (stat['branch_pages'] + stat['leaf_pages'] + stat['overflow_pages'] + 2)


@dataclass
class StatefulLmdbCreator:
    lmdb_creator_write_map: bool
    dst: str
    size_in_bytes: int

    def open(self):
        ensure_path_exists(self.dst)
        self.env = lmdb.open(self.dst, writemap=self.lmdb_creator_write_map, map_size=self.size_in_bytes)
        self.size_in_bytes = self.size_in_bytes
        logger.warning(f"creating lmdb with map_size={self.size_in_bytes / 2 ** 30} GB")

    def write(self, key, data):
        with self.env.begin(buffers=False, write=True) as txn:
            try:
                pages = len(data) // 4096 + 1
                to_use = pages * 4096 * 2  # for safety we multiply by 2
                if (get_used_size(txn.stat()) + to_use) < self.size_in_bytes:
                    txn.put(key.encode(), data)
                else:
                    raise RuntimeError(f"lmdb file reached the specified size")
            except MapFullError as mfe:  # if you hit this you are done .
                logger.error(
                    f'reached max size of a db. written={get_used_size(txn.stat())}, limit={self.size_in_bytes}')
                raise mfe

    def close(self):
        print(f"closing lmdb at {self.dst}")
        self.env.close()
        print(f"closed lmdb at {self.dst}")


@dataclass
class StatefullMultiLmdbCreator:
    """
    use this to write to lmdb from many processess using ray.
    """
    dst_path: str
    lmdb_creator_write_map: bool
    size_in_bytes_per_lmdb: int

    def __post_init__(self):
        self.next_i = 0
        self.open_next()

    def open_next(self):
        dst = os.path.join(self.dst_path, f"{self.next_i}")
        self.creator = StatefulLmdbCreator(self.lmdb_creator_write_map, dst, self.size_in_bytes_per_lmdb)
        self.creator.open()
        self.next_i += 1

    def _write(self, key, data, retry_count=0):
        try:
            self.creator.write(key, data)
        except Exception as e:
            logger.error(f"failed to write key={key} data={data} at {self.creator.dst}")
            self.creator.close()
            self.open_next()
            self._write(key, data, retry_count=retry_count + 1)
            if retry_count > 1000:
                raise RuntimeError(f"unexpected retry count:{retry_count}")

    def write(self, key, data):
        self._write(key, data, retry_count=0)

    def close(self):
        self.creator.close()


@dataclass
class LmdbCreator(DataStoreCreator):
    lmdb_creator_write_map: bool  # True for linux os for using sparse files.

    def create(self,
               key_data_pairs: Iterable[Tuple[str, bytes]],
               dst: str,
               size_in_bytes: int,
               total=None,
               bar=None) -> \
            Union[None, Iterable[Tuple[str, bytes]]]:
        """
        consumes given key data pairs until it consumes size_in_bytes.
        then returns the unfinished iterator to be reused.
        :param key_data_pairs:
        :param dst:
        :param size_in_bytes:
        :param total:
        :param bar:
        :return:
        """
        written_bytes = 0

        def continued(key, data):
            return chain([(key, data)], key_data_pairs)

        def get_used_size(stat):
            return 4096 * (stat['branch_pages'] + stat['leaf_pages'] + stat['overflow_pages'] + 2)

        with lmdb.open(dst, writemap=self.lmdb_creator_write_map, map_size=size_in_bytes) as env:
            logger.warning(f"creating lmdb with map_size={size_in_bytes / 2 ** 30} GB")

            with env.begin(buffers=False, write=True) as txn:
                bar = tqdm(desc=f"storing data to lmdb file:{dst}", total=total) if bar is None else bar
                for i, (key, data) in enumerate(key_data_pairs):
                    bar.update(1)
                    try:
                        pages = len(data) // 4096 + 1
                        to_use = pages * 4096 * 2  # for safety we multiply by 2
                        if (get_used_size(txn.stat()) + to_use) < size_in_bytes:
                            txn.put(key.encode(), data)
                        else:
                            return continued(key, data)  # return unfinished iterator to be used again
                    except MapFullError as mfe:  # if you hit this you are done .
                        logger.error(
                            f'reached max size of a db. written={get_used_size(txn.stat())}, limit={size_in_bytes}')
                        raise mfe


class ThreadedLmdbCreator(DataStoreCreator):

    def create(self, key_data_pairs: Iterable[Tuple[str, bytes]], dst: str, size_in_bytes):
        queue = Queue(maxsize=10000)

        def fetch():
            for key, data in tqdm(key_data_pairs, desc=f"storing data to lmdb file:{dst}"):
                queue.put((key, data))
            queue.put(None)

        def write():
            with lmdb.open(dst, writemap=True, map_size=size_in_bytes) as env:
                with env.begin(buffers=False, write=True) as txn:
                    while True:
                        t = queue.get()
                        if t is None:
                            break
                        key, data = t
                        # if writing nothing speeds up?
                        txn.put(key.encode(), data)

        fetcher = Thread(target=fetch)
        writer = Thread(target=write)
        fetcher.start()
        writer.start()
        fetcher.join()
        writer.join()
