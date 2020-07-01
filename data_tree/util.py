import abc
import os
import pickle
import shutil
from datetime import datetime
from hashlib import sha1
from queue import Queue, Full
from threading import Thread

import numpy as np
import pandas as pd
from frozendict import frozendict
from lazy import lazy
from loguru import logger
from easydict import EasyDict as edict
from tqdm.autonotebook import tqdm
from contextlib import contextmanager

WARN_SLOW_PREFETCH = False
import sys


# logger.remove()
# logger.add(sys.stderr,format="<green>{time}</green><lvl>\t{level}</lvl>\t{thread.name}\t{process.name}\t| <lvl>{message}</lvl>")

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
def ensure_dir_exists(dirname):
    import os
    from os import path, makedirs
    parent = os.path.dirname(fileName)
    if not path.exists(parent) and parent:
        try:
            logger.info(f"making dirs for {fileName}")
            makedirs(parent)
        except FileExistsError as fee:
            pass

def prefetch_generator(gen, n_prefetch=5, name=None):
    """
    use this on IO intensive(non-cpu intensive) task
    :param gen:
    :param n_prefetch:
    :return:
    """

    if n_prefetch <= 0:
        yield from gen
        return

    item_queue = Queue(n_prefetch)
    active = True

    END_TOKEN = "$$end$$"

    def loader():
        try:
            for item in gen:
                while active:
                    try:
                        # logger.debug(f"putting item to queue. (max {n_prefetch})")
                        item_queue.put(item, timeout=1)
                        break
                    except Full:
                        pass
                if not active:
                    # logger.info(f"break due to inactivity")
                    break
                # logger.debug("waiting for generator item")
            # logger.info("putting end token")
            item_queue.put(END_TOKEN)
        except Exception as e:
            import traceback
            logger.error(f"exception in prefetch loader:{e}")
            logger.error(traceback.format_exc())

    t = Thread(target=loader)
    t.daemon = True
    t.start()
    try:
        while True:
            # logger.info(f"queue status:{item_queue.qsize()}")
            if item_queue.qsize() == 0 and WARN_SLOW_PREFETCH:
                logger.warning(f"prefetching queue is empty! check bottleneck named:{name}")
            item = item_queue.get()
            if item is END_TOKEN:
                # logger.debug("an end token is fetched")
                break
            else:
                yield item
    except Exception as e:
        logger.error(e)
        raise e
    finally:
        # logger.info(f"trying to join loader {t.name}")
        active = False
        # consume all queue
        while item_queue.qsize() > 0:
            item_queue.get()
        t.join()
        # logger.info(f"loader {t.name} completed")


def dict_hash(val):
    return sha1(str(freeze(val)).encode()).hexdigest()


def freeze(_item):
    def _freeze(item):
        if isinstance(item, dict):
            return sorted_frozendict({_freeze(k): _freeze(v) for k, v in item.items()})
        elif isinstance(item, list):
            return tuple(_freeze(i) for i in item)
        elif isinstance(item, np.ndarray):
            return tuple(_freeze(i) for i in item)
        return item

    return _freeze(_item)


def sorted_frozendict(_dict):
    return frozendict(sorted(_dict.items(), key=lambda item: item[0]))


class PickledTrait:

    @property
    @abc.abstractmethod
    def value(self):
        pass

    @property
    @abc.abstractmethod
    def clear(self):
        pass


class Pickled(PickledTrait):
    def __init__(self, path, proc):
        self.loaded = False
        self._value = None
        self.path = path
        self.proc = proc

    @property
    def value(self):
        if not self.loaded:
            self._value = load_or_save(self.path, self.proc)
            self.loaded = True
        return self._value

    def clear(self):
        os.remove(self.path)
        self.loaded = False
        logger.info(f"deleted pickled file at {self.path}")


    def map(self, f):
        return MappedPickled(self, f)


class MappedPickled(PickledTrait):
    def __init__(self, src: PickledTrait, f):
        self.src = src
        self.f = f

    @lazy
    def value(self):
        return self.f(self.src.value)

    def clear(self):
        return self.src.clear()


def scantree(path,yield_dir = False):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            if yield_dir:
                yield entry
            yield from scantree(entry.path,yield_dir=yield_dir)  # see below for Python 2.x
        else:
            yield entry

def scanfiles(path):
    def gen():
        for item in tqdm(scantree(path), desc="scanning files.."):
            yield item
    from data_tree import series
    return series(gen())




def scan_images(path):
    from data_tree import series
    from PIL import Image
    EXTS = {"jpg", "png", "gif", "jpeg"}

    def gen():
        for item in tqdm(scantree(path), desc="scanning directory for images..."):
            ext = item.name.split(".")
            if len(ext):
                ext = ext[-1]
            if ext in EXTS:
                yield item

    return series(gen()).tag("dir_entries").map(
        lambda p: Image.open(p.path)
    ).tag("load_image")


def scan_images_cached(cache_path, scan_path) -> PickledTrait:
    """
    searches a given directory for images recursively and save its result as pkl.
    :param cache_path:
    :param scan_path:
    :return: PickledTrait[Series[Image]]
    """
    from data_tree import series
    from PIL import Image
    paths = Pickled(
        cache_path,
        lambda: series(
            scan_images(scan_path).tagged_value("dir_entries").map(
                lambda de:edict(path=de.path,name=de.name)
            ).values_progress(512, tqdm))
    )
    return paths.map(
        lambda ps: series(ps).tag("dir_entries").map(
            lambda d:d.path).tag("image_path").map(
            Image.open).tag("loaded_image"))


def save_images_to_path(series, path):
    """
    stores all Image instance in a series in to path with name:sha1(np.array(img)).hexdigest()+".png"
    :param series:Series[PIL.Image]
    :param path:destination dir
    :return:
    """
    from hashlib import sha1
    import numpy as np
    import os
    ensure_path_exists(path)
    for img in tqdm(series, desc=f"saving images"):
        _hash = sha1(np.array(img)).hexdigest()
        img_path = os.path.join(path, _hash + ".jpg")
        if not os.path.exists(img_path):
            img.save(img_path)


def shared_npy_array_like(ary: np.ndarray):
    import multiprocessing as mp

    buf = mp.RawArray(
        np.ctypeslib.as_ctypes_type(ary.dtype),
        ary.size)
    return np.frombuffer(buf, dtype=ary.dtype).reshape(ary.shape)




@contextmanager
def checktime(label="none"):
    start = datetime.now()
    yield
    end = datetime.now()
    dt = end - start
    print(f"time_{label}:{dt.total_seconds():.3f}")