import os

import numpy as np
from tqdm import tqdm

from data_tree.key_value_store import IKeyValueStoreCreator, Hdf5KeyValueStoreCreator, Hdf5KeyValueStore


def test_create():
    os.remove("test.hdf5")
    kvsc:IKeyValueStoreCreator = Hdf5KeyValueStoreCreator()
    ary = np.arange(1000).reshape((100,2,5))
    def items():
        for i,a in enumerate(ary):
            yield "hello"+str(i),a
    kvsc.create(tqdm(items()),"test.hdf5")
    kvs = Hdf5KeyValueStore("test.hdf5")
    print(kvs.keys())
    print(kvs[kvs.keys()[3]])
