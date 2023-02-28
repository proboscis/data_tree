import abc
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha1
from multiprocessing import Manager
from queue import Queue, Full, Empty
from threading import Thread
from typing import Callable

import numpy as np
import pandas as pd
from cytoolz import memoize
from easydict import EasyDict as edict
from frozendict import frozendict
from lazy import lazy
# from data_tree._series import Series
from tqdm import tqdm

from data_tree.get_callee import get_callee
from data_tree.picklable_file_lock import PicklableFileLock

WARN_SLOW_PREFETCH = False
import pickle


# logger.remove()
# logger.add(sys.stderr,format="<green>{time}</green><lvl>\t{level}</lvl>\t{thread.name}\t{process.name}\t| <lvl>{message}</lvl>")
def load_or_save_with_lock(path, proc, reader_lock: threading.RLock, writer_lock: threading.RLock, backend=pickle):
    # we need to share a reader counter across processes. but can we?
    try:
        with open(path, "rb") as f:
            return backend.load(f)  # this takes a lot of time.
    except Exception as e:
        from loguru import logger
        logger.info(f"failed to load cache at {path} for {proc.__name__} (cause:{e})")
        # ok, loading failed. we need to recalc.
        with writer_lock:  # wait for someone to complete writing
            # at this point the cache is ready
            pass

        res = proc()
        try:
            ensure_path_exists(path)
            with open(path, "wb") as f:
                logger.info(f"caching at {path}")
                backend.dump(res, f)  # actually now we can release the lock
        except Exception as e:
            logger.error(f"failed to cache:{e}")
            logger.error(f"failed to cache proc result at {path}. check the filesystem is writable.")
        logger.info(f"resolved cache load failure.")
        return res


def load_or_save(path, proc, backend=pickle):
    try:
        # logger.debug(f"opening file at:{path}")
        with open(path, "rb") as f:
            # logger.info(f"using backend:{backend}")
            # logger.debug(f"pid:|{os.getpid()}|loading cache at {path}")
            start = datetime.now()
            res = backend.load(f)
            end = datetime.now()
            dt = end - start
            # logger.debug(f"loading cache at {path} took {dt.total_seconds():.2f} seconds")
            return res
    except Exception as e:
        from loguru import logger
        logger.info(f"failed to load cache at {path} for {proc.__name__} (cause:{e})")
        res = proc()
        try:
            ensure_path_exists(path)
            with open(path, "wb") as f:
                logger.info(f"caching at {path}")
                backend.dump(res, f)  # actually now we can release the lock
        except Exception as e:
            logger.error(f"failed to cache:{e}")
            logger.error(f"failed to cache proc result at {path}. check the filesystem is writable.")
        logger.info(f"resolved cache load failure.")
        return res


def load_or_save_df(path, proc):
    try:
        from loguru import logger
        logger.info(f"loading df cache at {path}")
        start = datetime.now()
        res = pd.read_hdf(path, key="cache")
        end = datetime.now()
        dt = end - start
        logger.info(f"loading df cache at {path} took {dt.total_seconds():.2f} seconds")
        return res
    except Exception as e:
        from loguru import logger
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
            from loguru import logger
            logger.info(f"making dirs for {fileName}")
            makedirs(parent, exist_ok=True)
        except FileExistsError as fee:
            pass


def cached_ensure_path_exists(name):
    parent = os.path.dirname(name)
    return cached_makedirs(parent)


@memoize
def cached_makedirs(dir):
    if not os.path.exists(dir) and dir:
        try:
            from loguru import logger
            logger.info(f"making dirs for {dir}")
            os.makedirs(dir)
        except FileExistsError as fee:
            pass


def ensure_dir_exists(dirname):
    import os
    from os import path, makedirs
    parent = os.path.dirname(dirname)
    if not path.exists(parent) and parent:
        try:
            from loguru import logger
            logger.info(f"making dirs for {dirname}")
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
            from loguru import logger
            logger.error(f"exception in prefetch loader:{e}")
            logger.error(traceback.format_exc())
    from loguru import logger
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


class DefaultDict(dict):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def __missing__(self, key):
        from loguru import logger
        logger.debug(f"missing:{key}")
        res = self.f(key)
        self[key] = res
        return res


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
    def __init__(self, path, proc, backend=pickle):
        self.loaded = False
        self._value = None
        self.path = path
        self.proc = proc
        self.lock = PicklableFileLock(self.path + ".lock")
        ensure_path_exists(self.lock.lock.lock_file)
        self.backend = backend

    def __getstate__(self):
        state = dict(
            path=self.path,
            proc=self.proc,  # I think this is preventing pickling.. but it is a must to pickle this.
            backend=self.backend
        )
        # at this point self.proc requires recursive pickling
        return state

    def __setstate__(self, state):
        self.loaded = False
        self._value = None
        self.path = state["path"]
        self.proc = state["proc"]
        self.lock = PicklableFileLock(self.path + ".lock")
        self.backend = state['backend']

    @property
    def value(self):
        callee = get_callee()
        # logger.debug(f"cache value access from {callee}")
        # logger.debug(f"{threading.current_thread().name}:trying to aqquire lock:{self.lock.path}")
        with self.lock:
            if not self.loaded:
                # logger.debug(f"{threading.current_thread().name}:aqquired lock:{self.lock.path}")
                self._value = load_or_save(self.path, self.proc, backend=self.backend)
                self.loaded = True
        # logger.debug(f"{threading.current_thread().name}:released lock:{self.lock.path}")
        return self._value

    def clear(self):
        from loguru import logger
        with self.lock:
            try:
                # this try has no effect at all?
                ensure_dir_exists(self.path)
                os.remove(self.path)
                self.loaded = False
                logger.info(f"deleted pickled file at {self.path}")
            except FileNotFoundError as e:
                self.loaded = False
                logger.warning(f"no cache found at {self.path}")
                # embed()

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


def scantree(path, yield_dir=False):
    """Recursively yield DirEntry objects for given directory."""
    from loguru import logger
    try:
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                if yield_dir:
                    yield entry
                yield from scantree(entry.path, yield_dir=yield_dir)  # see below for Python 2.x
            else:
                yield entry
    except PermissionError as e:
        logger.warning(f"permission error at {path}. ignoring...")


def scantree_filtered(path, ignore: Callable, yield_dir=False):
    from loguru import logger
    try:
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                if yield_dir:
                    yield entry
                if not ignore(entry.path):
                    logger.info(f"digging {entry.path}")
                    if len(os.listdir(entry.path)) > 100:
                        logger.warning(f"directory {entry.path} has more than 100 entries")
                    yield from scantree_filtered(entry.path, ignore=ignore,
                                                 yield_dir=yield_dir)  # see below for Python 2.x
            else:
                yield entry
    except PermissionError as e:
        logger.warning(f"permission error at {path}. ignoring...")


def scantree_leafcut(path, to_go_deepr: Callable):
    from loguru import logger
    try:
        for entry in os.scandir(path):
            entry: os.DirEntry
            if entry.is_dir(follow_symlinks=False):
                yield entry
                if to_go_deepr(entry.path):
                    yield from scantree_leafcut(entry.path, to_go_deepr)

            else:
                yield entry
    except PermissionError as e:
        logger.warning(f"permission error at {path}. ignoring...")


def scanfiles(path) -> "Series[os.DirEntry]":
    def gen():
        for item in tqdm(scantree(path), desc="scanning files.."):
            yield item

    from data_tree import series
    return series(gen())


def scan_images(path):
    from PIL import Image
    return scan_image_paths(path).tag("dir_entries").update_metadata(scan_path=path).tag("path").map(
        Image.open).tag("load_image")


def scan_image_paths(path):
    from data_tree import series

    EXTS = {"jpg", "png", "gif", "jpeg"}

    def gen():
        for item in tqdm(scantree(path), desc=f"scanning {path} for images..."):
            ext = item.name.split(".")
            if len(ext):
                ext = ext[-1]
            if ext.lower() in EXTS:
                yield item.path

    return series(gen())

def scan_images_piped(path):
    EXTS = {"jpg", "png", "gif", "jpeg"}
    queue = Queue()
    def gen():
        for item in tqdm(scantree(path), desc=f"scanning {path} for images..."):
            queue.put(item)
        queue.put(None)

    t= Thread(target=gen)
    t.start()
    while True:
        item = queue.get()
        if item is None:
            break
        else:
            if item.name.split(".")[-1] in EXTS:
                yield item.path
    t.join()



def scan_dir_mp(path, n_worker=8):
    from loguru import logger
    logger.info("Scanning directory with %d workers", n_worker)
    manager = Manager()
    target_dirs = manager.Queue()
    finished = manager.Event()
    results = manager.Queue()
    worker_states = manager.dict()
    target_dirs.put(path)

    def worker(id):
        while not finished.is_set():
            try:
                worker_states[id] = "waiting"
                item: str = target_dirs.get(timeout=1)
                worker_states[id] = "working"
                for entry in os.scandir(item):
                    if entry.is_dir():
                        target_dirs.put(entry.path)
                    else:
                        results.put(entry.path)
            except Empty:
                continue
        worker_states[id] = "finished"

    workers = [Thread(target=worker, args=(i,)) for i in range(n_worker)]
    for w in workers:
        w.start()

    def check_workers():
        while True:
            time.sleep(1)
            if all([state == "waiting" for state in worker_states.values()]) and target_dirs.empty():
                break
        finished.set()

    checker = Thread(target=check_workers)
    checker.start()

    bar=tqdm(desc=f"scanning {path} with {n_worker} threads...")

    while not finished.is_set():
        try:
            item = results.get(timeout=1)
            yield item
            bar.update(1)
        except Empty:
            continue
    bar.close()
    for w in workers:
        w.join()
    checker.join()


def scan_image_paths_mp(path, num_workers):
    EXTS = {"jpg", "png", "gif", "jpeg"}
    for item in scan_dir_mp(path, n_worker=num_workers):
        if item.split(".")[-1] in EXTS:
            yield item


def scan_image_paths_cached(path, cache_path) -> "Series[str]":
    return Pickled(cache_path, lambda: scan_image_paths(path)).value
def scan_image_paths_pipe_cached(path, cache_path) -> "Series[str]":
    return Pickled(cache_path, lambda: list(scan_images_piped(path))).value
def scan_image_paths_mp_cached(path, cache_path,n_worker=8) -> "Series[str]":
    return Pickled(cache_path, lambda: list(scan_image_paths_mp(path,num_workers=n_worker))).value


def scan_images_sorted(path):
    scanned = scan_images(path)
    indices = scanned.tagged_value("path").argsort()
    scanned = scanned[indices]
    return scanned


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
                lambda de: edict(path=de.path, name=de.name)
            ).values_progress(512, tqdm))
    )
    return paths.map(
        lambda ps: series(ps).meta(cache_path=cache_path).tag("dir_entries").map(
            lambda d: d.path).tag("image_path").map(
            Image.open).tag("loaded_image")
    )


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


def embed_qt(locals):
    from qtpy.QtWidgets import QApplication
    from pyqtconsole.console import PythonConsole
    from pyqtconsole.highlighter import format
    app = QApplication([])

    console = PythonConsole(formats={
        'keyword': format('darkBlue', 'bold')
    }, locals=locals
    )
    console.show()

    console.eval_queued()

    return app.exec_()


def next_without_exception(iterator):
    from loguru import logger
    try:
        return next(iterator)
    except StopIteration as e:
        logger.info("iterator stop iteration")
        return None
