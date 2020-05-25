from loguru import logger
class Dummy:pass
import pickle
import numpy as np
def test_lmdb():
    import lmdb
    obj = dict(
        a=0,
        nested=dict(hello="world"),
        dummy=Dummy(),
        c=[0,10]
    )
    with lmdb.open("./test/test_db") as env:
        with env.begin(write=True) as txn:  # start transaction.
            # one thread at a time.
            txn.put("test".encode("utf-8"), "hello".encode("utf-8"))
            txn.put((0).to_bytes(4,"big"), "0 in 4 bytes".encode("utf-8"))
            txn.put((1).to_bytes(4, "big"), pickle.dumps(obj))
            #txn.put(bytes(10), pickle.dumps(obj)) # bytes(10) is not a representation of 10!
            logger.info("hello store with key:test")

    with lmdb.open("./test/test_db") as env:
        with env.begin(write=False) as txn:  # start transaction.
            # one thread at a time.
            value = txn.get("test".encode("utf-8"))
            value2 = txn.get((0).to_bytes(4, "big"))
            value3 = pickle.loads(txn.get((1).to_bytes(4, "big")))
            #value4 = pickle.loads(txn.get(bytes(10)))
            logger.info(f"value:{value} got from key:test")
            logger.info(f"value:{value2} got from key:0")
            logger.info(f"value:{value3} got from key:0")
            #logger.info(f"value:{value4} got from key:10")
    with lmdb.open("./test/test_db") as env:
        with env.begin(write=True,buffers=True) as txn:
            txn.put(b"a",b"hello")
            buf = txn.get(b"a")
            print(f"printing buffer:{buf}")


    with lmdb.open("./test/test_db") as env:
        with env.begin(write=True) as txn:
            print(txn.get(b"a"))
            print(txn.get(b"c"))
            print(txn.stat())
            print(list(txn.cursor()))

def test_lmdb_cache_series():
    from data_tree import series
    from tqdm import tqdm
    from data_tree.ops.cache import LMDBCachedSeries
    dummy = np.zeros(shape=(100,100,3),dtype="uint8")
    #src = series(np.arange(2)).map(lambda i: dict(i=i, obj=dict(), img=dummy))
    src = series(np.arange(1000)).map(lambda i: dict(i=i,img=dummy,img2=dummy))
    cached = LMDBCachedSeries(src=src,db_path="./test/lmdb_cache_series",map_size=1e9*3)
    cached.clear()
    a = src.values_progress(32)
    b = cached.values_progress(32)
    #assert src.values_progress(batch_size=32) == cached.values_progress(batch_size=32)
    #assert src.values_progress(batch_size=32) == cached.values_progress(batch_size=32)
    print(src.sps())
    print(cached.sps())
