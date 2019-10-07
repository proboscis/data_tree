
import numpy as np
from logzero import logger

from data_tree import Series


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
