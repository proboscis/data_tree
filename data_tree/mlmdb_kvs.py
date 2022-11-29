import os

from returns.result import safe, Success

from data_tree.key_value_store import MergedKvs
from data_tree.lmdb_bytes_kvs import LmdbBytesKvs


def load_mlmdb_kvs(mlmdb_root: str):
    kvs_list = []
    for entry in os.scandir(mlmdb_root):
        entry: os.DirEntry
        match safe(int)(entry.name):
            case Success(num):
                kvs = LmdbBytesKvs(entry.path, sort_keys=True)
                kvs_list.append(kvs)

    return MergedKvs(kvs_list)
