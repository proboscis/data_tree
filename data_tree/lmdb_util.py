import abc
from dataclasses import dataclass
from itertools import chain
from queue import Queue
from threading import Thread
from typing import Tuple, Iterable, Union, Optional

import lmdb as lmdb
import ray
from lmdb import MapFullError
from loguru import logger
from tqdm import tqdm


class DataStoreCreator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create(self, key_data_pairs: Tuple[str, bytes], dst: str, size_in_bytes: int):
        pass


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
