
import numpy as np
from loguru import logger

from data_tree import Series
from tqdm.autonotebook import tqdm

def test_numpy_dataset():
    data = np.arange(100)
    s = Series.from_iterable(data)
    p = np.random.permutation(100)
    assert (s[:100].values == np.arange(100)).all()
    assert (s[10:100].values == np.arange(10, 100)).all()
    assert (s[10:100].values == np.arange(10, 100)).all()
    assert (s[:100][p].values == np.arange(100)[p]).all()
    for item in [s[10:], s[10:][20:], s[10:][20:][:-5]]:
        logger.info(np.array(item.values))
    left = s[10:][20:][:-5]
    right = np.arange(100)[30:-5]
    assert len(left) == len(right)
    logger.info(np.array(left.values))
    logger.info(right)
    assert (np.array(left.values) == right).all(), f"{left} != {right}"
    left_cache = left.hdf5("test.hdf5")
    left_cache.clear()
    left_val = left_cache.values
    right_val = left.values
    assert (left_val == right_val).all()
    assert (left_cache[:5].values == left[:5].values).all()


def test_shelve():
    data = np.arange(100)
    s = Series.from_iterable(data)
    c = s.shelve("test")[:10]
    assert (s[:10].values == c.values).all()


def test_hash():
    N = 10000
    series = Series.from_numpy(np.arange(N))
    indices = np.random.permutation(N)
    shuffled = series[indices]
    pkled = shuffled.with_hash(
        lambda hash, self: self.map(lambda item: item * 2).pkl("test_shuffled.pkl", src_hash=hash))

    assert pkled.values == shuffled.map(lambda item: item * 2).values, "shuffling and caching with hash"

def test_metadata():
    N = 10000
    series = Series.from_numpy(np.arange(N))
    s2 = series.update_metadata(name="hello world")
    s3 = s2.visualization(lambda:print)
    print(s3.trace(0))
    print(s3._trace(0))

def sum_accessors(t):
    res = []
    for item in t:
        #print(item.shape)
        #res.append(item[:])
        res.append(item)
    return res
def test_batch_generator():
    from multiprocessing import get_context
    logger.warning(f"testing batch generator with mp context:{get_context()}")
    N = 10000
    s1 = Series.from_numpy(np.arange(N))
    h1 = s1.hdf5("s1.hdf5")
    h2 = s1.hdf5("s2.hdf5")
    h3 = s1.hdf5("s3.hdf5")
    for h in [h1,h2,h3]:
        h.clear()
        h.ensure()
        pass

    zipped = h1.ensured_accessor_closed(4).zip(
        h2.ensured_accessor_closed(4),
        h3.ensured_accessor_closed(4)
    )
    bs=512
    for batch in zipped.batch_generator(batch_size=bs,progress_bar=tqdm):
        pass
    #assert batch[0] == (0,0,0),"batch_generator must return a batch of elements"
    #logger.info(f"zipped batch:{batch}")
    for batch in zipped.map(sum_accessors).batch_generator(batch_size=bs,progress_bar=tqdm):
        pass
    #assert batch[0] == 0,"mapped zipped must work right."
    #logger.info(f"mapped zipped batch:{batch}")

    for batch in zipped.mp_map(sum_accessors,num_process=None).batch_generator(batch_size=bs, preload=False,progress_bar=tqdm):
        pass
    #logger.info(f"mp_mapped zipped batch without preload :{batch}")

    #logger.info(f"starting to mp_map with preload")
    for batch in zipped.mp_map(sum_accessors).batch_generator(batch_size=bs, preload=10,progress_bar=tqdm):
        pass
    #logger.info(f"mp_mapped zipped batch with preload :{batch}")

def _work(i,value):
    import time
    from loguru import logger
    time.sleep(5)
    value.value += 1
    logger.info(f"hello:{i},{value.value}")

def test_mp_mac_os():
    from multiprocessing import Process,Manager
    m = Manager()
    v = m.Value(int,0)
    workers = []
    for i in range(100):
        workers.append(Process(target=_work,args=(i,v)))

    for p in workers:
        p.start()

    for p in workers:
        p.join()
