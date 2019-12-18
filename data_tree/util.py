import pickle
from datetime import datetime
from hashlib import sha1
from queue import Queue, Full
from threading import Thread

import numpy as np
import pandas as pd
from frozendict import frozendict
from logzero import logger


def load_or_save(path, proc):
    try:
        with open(path, "rb") as f:
            logger.info(f"loading cache at {path}")
            start = datetime.now()
            res = pickle.load(f)
            end = datetime.now()
            dt = end - start
            logger.info(f"loading cache at {path} took {dt.total_seconds():.2f} seconds")
            return res
    except Exception as e:
        logger.info(f"failed to load cache at {path} for {proc.__name__} (cause:{e})")
        res = proc()
        ensure_path_exists(path)
        with open(path, "wb") as f:
            logger.info(f"caching at {path}")
            pickle.dump(res, f)
        return res


def load_or_save_df(path, proc):
    try:
        logger.info(f"loading df cache at {path}")
        start = datetime.now()
        res = pd.read_hdf(path, key="cache")
        end = datetime.now()
        dt = end - start
        logger.info(f"loading df cache at {path} took {dt.total_seconds():.2f} seconds")
        return res
    except Exception as e:
        logger.info(f"failed to load cache at {path} for {proc.__name__} (cause:{e})")
        df: pd.DataFrame = proc()
        logger.info(f"caching at {path}")
        ensure_path_exists(path)
        df.to_hdf(path, key="cache")
        return df

def batch_index_generator(start, end, batch_size):
    for i in range(start, end, batch_size):
        yield i, min(i + batch_size, end)


def ensure_path_exists(fileName):
    import os
    from os import path, makedirs
    parent = os.path.dirname(fileName)
    if not path.exists(parent) and parent:
        try:
            logger.info(f"making dirs for {fileName}")
            makedirs(parent)
        except FileExistsError as fee:
            pass


def prefetch_generator(gen, n_prefetch=5):
    """
    use this on IO intensive(non-cpu intensive) task
    :param gen:
    :param n_prefetch:
    :return:
    """

    item_queue = Queue(n_prefetch)
    active = True

    END_TOKEN="$$end$$"
    def loader():
        for item in gen:
            while active:
                try:
                    #logger.info(f"putting item to queue. (max {n_prefetch})")
                    item_queue.put(item, timeout=1)
                    break
                except Full:
                    pass
            if not active:
                break
        item_queue.put(END_TOKEN)

    t = Thread(target=loader)
    t.daemon = True
    t.start()
    try:
        while True:
            #logger.info(f"queue status:{item_queue.qsize()}")
            if item_queue.qsize() == 0:
                logger.warn(f"prefetching queue is empty! check bottlenecks")
            item = item_queue.get()
            if item is END_TOKEN:
                break
            else:
                yield item
    finally:
        active = False
        t.join()


def dict_hash(val):
    return sha1(str(freeze(val)).encode()).hexdigest()


def freeze(_item):
    def _freeze(item):
        if isinstance(item, dict):
            return sorted_frozendict({_freeze(k):_freeze(v) for k, v in item.items()})
        elif isinstance(item, list):
            return tuple(_freeze(i) for i in item)
        elif isinstance(item, np.ndarray):
            return tuple(_freeze(i) for i in item)
        return item

    return _freeze(_item)


def sorted_frozendict(_dict):
    return frozendict(sorted(_dict.items(), key=lambda item: item[0]))