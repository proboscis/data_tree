import numpy as np
from data_tree import series
from loguru import logger
def test_hdf5_lock():

    s = series(np.arange(100))
    s1 = s.hdf5("test_s1.hdf5")
    s2 = s.hdf5("test_s2.hdf5")
    for s in [s1, s2]:
        s.clear()
        s.ensure()
    from itertools import islice
    # for some reason, islice stops execution
    logger.info(list(islice(zip(*(i.batch_generator(16,preload=5) for i in [s1, s2])),0,1))[0])

