import multiprocessing
import multiprocessing
import numbers
import os
import pickle
import queue
import shelve
import threading
from contextlib import contextmanager
# from threading import RLock
from datetime import time
from multiprocessing import RLock
from multiprocessing.dummy import Value

import h5py
import numpy as np
from PIL import Image
from filelock import FileLock
from lazy import lazy as en_lazy

from lru import LRU
from tqdm import tqdm

from proboscis_image_rules.rulebook import legacy_auto as auto

from data_tree import logger
from data_tree._series import SourcedSeries, NumpySeries, MappedSeries, Series
from data_tree.coconut.astar import Conversion
from data_tree.indexer import IdentityIndexer, Indexer
from data_tree.custom_resource import ContextResource
from data_tree.util import ensure_path_exists, batch_index_generator, prefetch_generator


# from loguru import logger


class CachedSeries(SourcedSeries):
    def slice_generator(self, slices, preload=5, en_numpy=False):
        # do not propagate batch generation to parent
        def batches():
            for _slice in slices:
                batch = self[_slice].values
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")


class PickledSeries(CachedSeries):

    def clone(self, parents):
        return PickledSeries(src=parents[0], pickle_path=self.pickle_path, src_hash=self.src_hash)

    def __init__(self, src: Series, pickle_path: str, src_hash=None):
        self._src = src
        self._indexer = IdentityIndexer(self.src.total)
        self.pickle_path = pickle_path
        self.src_hash = src_hash

    @en_lazy
    def cache(self):
        if os.path.exists(self.pickle_path):
            with open(self.pickle_path, "rb") as f:
                failed = False
                try:
                    loaded_hash, data = pickle.load(f)
                    if data is not None and len(data) == self.total and loaded_hash == self.src_hash:
                        return data
                    else:
                        logger.warning(f"found data is corrupt:{loaded_hash} != {self.src_hash}")
                        failed = True
                except Exception as e:
                    logger.warning(f"failed to load pkl:{e}")
                    failed = True
                if failed:
                    os.remove(self.pickle_path)
                    logger.warning(f"removed corrupt cache at {self.pickle_path}")

        with open(self.pickle_path, "wb") as f:
            data = self.src.values_progress(batch_size=128, progress_bar=tqdm)
            pickle.dump((self.src_hash, data), f)
            logger.info(f"pickled at {self.pickle_path} with src_hash {self.src_hash}")
        return data

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def total(self):
        return self.src.total

    def _get_item(self, index):
        return self.cache[index]

    def _get_slice(self, _slice):
        return self.cache[_slice]

    def _get_indices(self, indices):
        return [self.cache[i] for i in indices]

    def _get_mask(self, mask):
        return self._get_indices(np.arange(self.total)[mask])


class ShelveSeries(CachedSeries):

    def __init__(self, src: Series, cache_path: str, src_hash: str = None):
        self._src = src
        self._indexer = IdentityIndexer(total=self.src.total)
        self.cache_path = cache_path
        self.flag_path = cache_path + ".flags.hdf5"
        self.shelve_file = ContextResource(lambda: shelve.open(self.cache_path))
        self.src_hash = src_hash
        if self.shelve_file.map(lambda db: "src_hash" in db and db["src_hash"] != src_hash).get():
            os.remove(self.cache_path)
        with self.shelve_file.to_context() as db:
            if "src_hash" not in db:
                db["src_hash"] = src_hash

    def clone(self, parents):
        return ShelveSeries(src=parents[0], cache_path=self.cache_path, src_hash=self.src_hash)

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def total(self):
        return self._src.total

    def _get_item(self, index):
        with self.shelve_file.to_context() as f:
            key = str(index)
            if key not in f:
                f[key] = self.src._get_item(index)
            return f[key]

    def _get_slice(self, _slice):
        current, stop, step = _slice.indices(len(self))
        with self.shelve_file.to_context() as f:
            if not all([str(i) in f for i in range(current, stop, step)]):
                src_vals = self.src._get_slice(_slice)
                for i in range(current, stop, step):
                    f[str(i)] = src_vals[i]
            res = []
            for i in range(current, stop, step):
                key = str(i)
                res.append(f[key])
            return res

    def _get_indices(self, indices):
        with self.shelve_file.to_context() as f:
            res = []
            for i in indices:
                key = str(i)
                if key not in f:
                    f[key] = self.src._get_item(i)
                res.append(f[key])
            return res

    def _get_mask(self, mask):
        return self._get_indices(np.arange(self.total)[mask])

    def clear(self):
        return os.remove(self.cache_path)


class FlagCacheProvider:
    def create_flags(self, total: int):
        pass

    def craete_cache(self, total: int):
        pass

    def create_coint(self):
        pass


class MpFlagCache(FlagCacheProvider):
    def __init__(self):
        self.manager = multiprocessing.Manager()

    def create_flags(self, total: int):
        return self.manager.list([False] * total)

    def create_cache(self, total: int):
        return self.manager.list([None] * total)

    def create_count(self):
        return self.manager.Value("int", 0)


class LocalFlagCache(FlagCacheProvider):
    def create_flags(self, total: int):
        return [False] * total

    def create_cache(self, total: int):
        return [None] * total

    def create_coutn(self):
        return Value("int", 0)


class LazySeries(CachedSeries):
    try:
        provider = MpFlagCache()
        logger.warning(f"created lazy series with multi processing manager")
        # each manager creation causes a new process to be spawned. beware!
    except Exception as e:
        logger.warning(f"could not create process safe cache. using non-process safe cache")
        provider = LocalFlagCache()

    def clone(self, parents):
        return LazySeries(src=parents[0], prefer_slice=self.prefer_slice)

    @property
    def total(self):
        return len(self.src)

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    def __init__(self, src: Series, prefer_slice=True):
        # this class will be accessed by multiple processes so I need to make it process safe.
        ctx = multiprocessing.get_context("fork")  # DO NOT USE SPAWNING
        # this may be causing a dead lock
        self._src = src
        self.count_lock = ctx.RLock()
        # self.locks = [ctx.RLock() for i in range(self.src.total)]
        # hmm, we need so many locks.
        # actually we only need lock to share lock state.

        self._indexer = IdentityIndexer(self.src.total)
        self.states = self.provider.create_flags(self.total)
        self.state_lock = ctx.Lock()
        self.cache = self.provider.create_cache(self.total)
        self.num_loaded = self.provider.create_count()
        self._index = np.arange(src.total)
        self.prefer_slice = prefer_slice  # not used currently
        self.ctx = ctx

    def _get_item(self, index):
        #logger.debug(f"trying to get lock pid:{os.getpid()} idx:{index}")

        # first check the state of target index,
        # wait if it is now loading... but how long do you wait?
        # you can use an event for this so that the waiter can wait on event.

        self.ctx.Event()

        with self.state_lock:
            state = self.states[index]
            if not state:
                logger.debug(f"cache not hit. pid:{os.getpid()} at {index}.")
                self.states[index] = "loading"#self.ctx.Event() #
                #logger.debug(f"obtained a lock:{os.getpid()}")
                #well.. should I do a polling to check if this index is released?

        if not state:
            self._update_cache(index)

        while state == "loading":
            time.sleep(0.1) # I dont like the way this is done though...
            state = self.states[index]
        return self.cache[index]

    def _update_cache(self,index):
        self.cache[index] = self.src._values(index)
        self.states[index] = True
        with self.count_lock:
            self.num_loaded.value += 1
            logger.debug(f"written data to {index} from pid: {os.getpid()} \t progress:{self.num_loaded}/{len(self.states)}")
        with self.state_lock:
            self.states[index] = True

    def _ensure_and_return_smart(self, smart_indexer):
        if not all(self.states[smart_indexer]):
            non_cached_indices = self._index[smart_indexer][~np.array(self.states[smart_indexer])]
            for v, i in zip(self.src._values(non_cached_indices), non_cached_indices):
                self._get_item(i)
        else:
            pass
            # logger.debug(f"hit")
        if isinstance(smart_indexer, slice):
            return self.cache[smart_indexer]
        else:
            return [self.cache[i] for i in smart_indexer]

    def _get_slice(self, _slice):
        return self._ensure_and_return_smart(_slice)

    def _get_indices(self, indices):
        return self._ensure_and_return_smart(indices)

    def _get_mask(self, mask):
        return self._ensure_and_return_smart(mask)


class NumpyCache(NumpySeries, CachedSeries):

    def clone(self, parents):
        return NumpyCache(parents[0])

    @property
    def src(self) -> Series:
        return self._src

    @property
    def parents(self):
        return [self.src]

    def __init__(self, src: Series):
        self._src = src
        super().__init__(self.src.values)


class Hdf5CachedSeries(CachedSeries):
    @property
    def src(self) -> Series:
        return self._src

    def clone(self, parents):
        return Hdf5CachedSeries(src=parents[0], cache_path=self.cache_path, src_hash=self.src_hash, **self.dataset_opts)

    def __init__(self, src: Series, cache_path: str, src_hash=None, **dataset_opts):
        self.cache_path = cache_path
        logger.info(f"initialized hdf5 cache series at {cache_path}")
        self._src = src
        self._indexer = IdentityIndexer(self.total)
        self.src_hash = "None" if src_hash is None else src_hash
        self.dataset_opts = dataset_opts
        # self.lock = RLock()  # FileLock(self.cache_path + ".lock")
        self.lock = FileLock(self.cache_path + ".lock")

    def prepared(self):
        ensure_path_exists(self.cache_path)
        with self.lock:  # dont let multiple thread check this.
            # why is this repeatedly called in batch gen?
            # logger.info(
            #    f"trying to open hdf5 for preparation at thread:{threading.currentThread().name} |pid:{os.getpid()} ")
            with h5py.File(self.cache_path, mode="a") as f:  # tries to open lock even though it is locked..
                # sometimes the file is already open after ensured.
                if "src_hash" in f.attrs and f.attrs["src_hash"] != self.src_hash:
                    os.remove(self.cache_path)
                    logger.warning(f"deleted cache due to inconsistent hash of source. {self.cache_path}")

            with h5py.File(self.cache_path, mode="a") as f:
                if "src_hash" not in f.attrs:
                    f.attrs["src_hash"] = self.src_hash
                if not all(map(lambda item: item in f, "value flag".split())):
                    sample = self.src[0]
                    if isinstance(sample, int):
                        sample = np.int64(sample)
                    elif isinstance(sample, float):
                        sample = np.float64(sample)
                    if "value" not in f:
                        if hasattr(sample, "shape"):
                            shape = (self.total, *sample.shape)
                        else:
                            shape = (self.total,)
                        if sample.dtype == np.float64:
                            logger.warning(f"this dataset({self.__class__.__name__}) returns value with dtype:float64 ")

                        if self.dataset_opts.get("chunks") is True:
                            items_per_chunk = max(1024 * 1024 * 1 // sample.nbytes, 1)  # chunk is set to 1 MBytes

                            self.dataset_opts["chunks"] = (items_per_chunk, *sample.shape)
                        logger.info(f"dataset created with options:{self.dataset_opts}")
                        logger.warning(f"dataset dtype: {sample.dtype}")
                        f.create_dataset("value", shape=shape, dtype=sample.dtype, **self.dataset_opts)
                        logger.info(f"created value in hdf5")
                    if "flag" not in f:
                        f.create_dataset("flag", data=np.zeros((self.total,), dtype=np.bool))
                        logger.info(f"created flag in hdf5")
                    logger.info(f"created hdf5 cache at {self.cache_path}.")
                    f.flush()
                    logger.info(f"{list(f.keys())}")
        return True

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._src.total

    def check_zeros_in_cache(self, _slice):
        with self.lock:
            with h5py.File(self.cache_path, mode="r+") as f:
                flags = f["flag"]
                flags_on_mem = flags[:]
                xs = f["value"]
                for i in tqdm(range(*_slice.indices(_slice.stop))):
                    if flags_on_mem[i]:
                        x = xs[i]
                        if (x == 0).all():
                            flags[i] = False
                            flags_on_mem[i] = False
            return flags_on_mem

    def ensure(self, _slice=None, batch_size=32, check_non_zero=False, preload=5):
        if batch_size is None:
            batch_size = 4
        if _slice is None:
            _slice = slice(0, self.total)
            logger.info(f"ensuring whole dataset")
            progress_bar = tqdm
        else:
            progress_bar = lambda _it, **kwargs: _it
        if self.prepared():
            item_queue = queue.Queue(preload)
            if check_non_zero:
                flags = self.check_zeros_in_cache(_slice)
            else:
                with self.lock:
                    with h5py.File(self.cache_path, mode="r") as f:
                        flags = f["flag"][:]

            def missing_batches():
                start, end, step = _slice.indices(len(self))
                batch_indices = list(batch_index_generator(start, end, batch_size))
                missing_slices = []
                for bs, be in batch_indices:
                    if not flags[bs:be].all():
                        missing_slices.append(slice(bs, be, 1))
                yield from progress_bar(zip(missing_slices,
                                            self.src.slice_generator(missing_slices, preload=preload, en_numpy=True)),
                                        desc=f"{self.__class__.__name__}:filling missing cache batches",
                                        total=len(missing_slices))

            def yielder():
                for item in missing_batches():
                    item_queue.put(item)
                item_queue.put(None)

            t = threading.Thread(target=yielder)
            t.start()
            """
            def saver():
                with self.lock:
                    with h5py.File(self.cache_path, mode="r+") as f:
                        flags = f["flag"]
                        xs = f["value"]
                        observed = 0
                        while True:
                            item = item_queue.get()
                            if item is None:
                                break
                            _s, x = item
                            # logger.info(f"item inside: {item}")
                            xs[_s] = x
                            saved_count = len(x)
                            flags[_s] = True
                            observed += saved_count
                            # if observed > 1000:
                            f.flush()
                            observed = 0
            """
            with self.lock:
                with h5py.File(self.cache_path, mode="r+") as f:
                    flags = f["flag"]
                    xs = f["value"]
                    observed = 0
                    while True:
                        item = item_queue.get()
                        if item is None:
                            break
                        _s, x = item
                        # logger.info(f"item inside: {item}")
                        try:
                            assert isinstance(x, np.ndarray)
                            xs[_s] = x
                            saved_count = len(x)
                            flags[_s] = True
                            observed += saved_count
                            # if observed > 1000:
                            f.flush()
                            observed = 0
                        except Exception as e:
                            logger.info(f"failed to write data at slice {_s}. data:{x}")
                            raise e
            t.join()

    def ensured(self, batch_size=None):
        try:
            self.ensure(batch_size=batch_size)
        except OSError as ose:
            logger.warning(f"failed to open source hdf5 for ensuring. assuming it is already ensured.")
            logger.warning(ose)
        return Hdf5OpenFileAdapter(
            hdf5_initializer=lambda: None,
            open_file=h5py.File(self.cache_path, mode="r"),
            key="value",
            parents=[self]
        )

    def ensured_accessor(self, batch_size=None):
        try:
            self.ensure(batch_size=batch_size)
        except OSError as ose:
            logger.warning(f"failed to open source hdf5 for ensuring. assuming it is already ensured.")
            logger.warning(ose)
        return Hdf5OpenFileAccessor(
            open_file=h5py.File(self.cache_path, mode="r"),
            key="value",
            parents=[self]
        )

    def ensured_accessor_closed(self, batch_size=None):
        try:
            pass
            # self.ensure(batch_size=batch_size)
        except OSError as ose:
            logger.warning(f"failed to open source hdf5 for ensuring. assuming it is already ensured.")
            logger.warning(ose)
        return Hdf5ClosedFileAccessor(
            file_path=self.cache_path,
            key="value",
            parents=[self]
        )

    def clear(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            logger.warning(f"deleted cache at {self.cache_path}")

    def _get_item(self, index):
        if self.prepared():
            with self.lock:
                with h5py.File(self.cache_path, mode="r") as f:
                    if f["flag"][index]:
                        return f["value"][index]

                with h5py.File(self.cache_path, mode="r+") as f:
                    x = self.src._values(index)
                    f["value"][index] = x
                    f["flag"][index] = True
                    return x

    def _get_slice(self, _slice):
        if self.prepared():
            with self.lock:
                with h5py.File(self.cache_path, mode="r") as f:  # opening file takes 20ms
                    if f["flag"][_slice].all():
                        # logger.info(f"cache is ready for this slice")
                        return f["value"][_slice]
                self.ensure(_slice)
                with h5py.File(self.cache_path, mode="r") as f:
                    return f["value"][_slice]

    def slice_generator(self, slices, preload=5, en_numpy=False):
        # do not propagate batch generation to parent
        self.ensure()  # checking all values before iteration.

        def batches():
            for _slice in slices:
                # logger.info(f"get lock:{threading.currentThread().name}")
                with self.lock, h5py.File(self.cache_path, mode="r") as f:
                    values = f["value"][_slice]  # do not yield while holding a lock!
                    # logger.info(f"yield:{threading.currentThread().name}")
                yield values
                # logger.info(f"release lock:{threading.currentThread().name}")

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")

    def _get_indices(self, indices):
        logger.warning(f"indices access on hdf5 is slow")
        if self.prepared():
            with h5py.File(self.cache_path, mode="r+") as f:
                flags = f["flag"]
                values = f["value"]
                res_values = []
                for i in indices:
                    if flags[i]:
                        res_values.append(values[i])
                    else:
                        v = self.src._values(i)
                        f["value"][i] = v
                        f["flag"][i] = True
                        res_values.append(v)
                return res_values

    def _get_mask(self, mask):
        return self._get_indices(np.arange(self.total)[mask])


class Hdf5OpenFileAdapter(Series):
    def clone(self, parents):
        pass

    @property
    def parents(self):
        return self._parents

    def __init__(self, hdf5_initializer, open_file, key, parents=None):
        if key not in open_file:
            hdf5_initializer()
        self._parents = [] if parents is None else parents
        self.hdf5_file = open_file
        self.data = self.hdf5_file[key]
        self._indexer = IdentityIndexer(total=len(self.data))
        self.lock = multiprocessing.Lock()

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._indexer.total

    def _get_item(self, index):
        with self.lock:
            return self.data[index]

    def _get_slice(self, _slice):
        with self.lock:
            return self.data[_slice]

    def _get_indices(self, indices):
        with self.lock:
            return self.data[indices]

    def _get_mask(self, mask):
        with self.lock:
            return self.data[mask]


class MappedHdf5Series(Hdf5CachedSeries):
    def __init__(self, src: Series, mapper, cache_path: str, **dataset_opts):
        super().__init__(src, cache_path, **dataset_opts)
        self.mapper = mapper

    def _get_item(self, index):
        return self.mapper(super()._get_item(index))

    def _get_slice(self, _slice):
        return [self.mapper(i) for i in super()._get_slice(_slice)]

    def _get_indices(self, indices):
        return [self.mapper(i) for i in super()._get_indices(indices)]

    def _get_mask(self, mask):
        return [self.mapper(i) for i in super()._get_mask(mask)]


class AutoSeries(MappedSeries):
    def __init__(self, src: Series, format):
        self.format = format
        mapper = auto(format)
        super().__init__(src, single_mapper=mapper)

    def hdf5(self, cache_path, src_hash=None, **dataset_opts):
        converter: Conversion = self[0].converter(type="numpy")
        end = converter.edges[-1].dst
        return MappedHdf5Series(
            self.map(lambda a: a.to(end)),
            mapper=auto(end),
            cache_path=cache_path,
            src_hash=src_hash, **dataset_opts)


class Hdf5Accessor:
    def __init__(self, hdf5_database, slicer, lock):
        self.slicer = (slicer,)
        self.data = hdf5_database
        self.shape = self.data.shape[1:]
        self.dtype = self.data.dtype
        self.lock = lock

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            pass
        else:
            idx = (idx,)
        index = self.slicer + idx
        try:
            # with self.lock:
            data = self.data[index]
            return data
        except Exception as e:
            logger.error(f"failed to access hdf5 data with index:{index}")
            raise e

    def __iter__(self):
        pass


class MappedHdf5Accessor(Hdf5Accessor):
    def __init__(self, src: Hdf5Accessor, f):
        self.src = src
        super().__init__(src.data, src.slicer, src.lock)
        self.f = f

    def __getitem__(self, item):
        values = self.src[item]
        return self.f(values)


class Hdf5ClosedAccessor:
    def __init__(self, file_opener, key, slicer, shape, dtype):
        self.slicer = (slicer,)
        self.opener = file_opener
        self.key = key
        self.shape = shape
        self.dtype = dtype
        with self.opener() as data:
            self.shape = data[self.key].shape[1:]

    def __getitem__(self, idx):
        with self.opener() as hdf5:
            if self.shape == ():
                index = self.slicer
            elif isinstance(idx, tuple):
                index = self.slicer + idx
            else:
                idx = (idx,)
                index = self.slicer + idx
            try:
                data = hdf5[self.key][index]
                return data
            except Exception as e:
                logger.error(f"failed to access hdf5 data with index:{index}")
                raise e

    def __iter__(self):
        pass


class Hdf5OpenFileAccessor(Series):
    def clone(self, parents):
        pass

    @property
    def parents(self):
        return self._parents

    def __init__(self, open_file, key, parents=None):
        self._parents = [] if parents is None else parents
        self.hdf5_file = open_file
        self.data = self.hdf5_file[key]
        self._indexer = IdentityIndexer(total=len(self.data))
        self.indices = np.arange(self.total)
        self.shape = self.data.shape[1:]
        self.dtype = self.data.dtype
        self.lock = multiprocessing.Lock()

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._indexer.total

    def _get_item(self, index):
        return Hdf5Accessor(self.data, index, self.lock)

    def _get_slice(self, _slice):
        return [Hdf5Accessor(self.data, i, self.lock) for i in self.indices[_slice]]

    def _get_indices(self, indices):
        return [Hdf5Accessor(self.data, i, self.lock) for i in indices]

    def _get_mask(self, mask):
        return [Hdf5Accessor(self.data, i, self.lock) for i in self.indices[mask]]

    def close(self):
        self.hdf5_file.close()


def _h5py_opener(path):
    def _inner():
        return h5py.File(path, "r")

    return _inner


class Hdf5ClosedFileAccessor(Series):
    def clone(self, parents):
        return Hdf5ClosedFileAccessor(file_path=self.file_path, key=self.key, parents=self.parents)

    @property
    def parents(self):
        return self._parents

    @contextmanager
    def opener(self):
        with self.lock, h5py.File(name=self.file_path, mode="r") as f:
            yield f

    def __init__(self, file_path, key, parents=None):
        from functools import partial
        self._parents = [] if parents is None else parents
        self.key = key
        self.file_path = file_path
        self.lock = FileLock(self.file_path + ".lock")

        with self.opener() as hdf5:
            self.shape = hdf5[self.key].shape[1:]
            self.dtype = hdf5[self.key].dtype
            self._indexer = IdentityIndexer(total=len(hdf5[self.key]))
        self.indices = np.arange(self.total)
        # self.lock = multiprocessing.Lock()

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._indexer.total

    def _get_item(self, index):
        return Hdf5ClosedAccessor(self.opener, self.key, index, shape=self.shape, dtype=self.dtype)

    def _get_slice(self, _slice):
        return [Hdf5ClosedAccessor(self.opener, self.key, i, shape=self.shape, dtype=self.dtype) for i in
                self.indices[_slice]]

    def _get_indices(self, indices):
        return [Hdf5ClosedAccessor(self.opener, self.key, i, shape=self.shape, dtype=self.dtype) for i in indices]

    def _get_mask(self, mask):
        return [Hdf5ClosedAccessor(self.opener, self.key, i, shape=self.shape, dtype=self.dtype) for i in
                self.indices[mask]]


int2bytes = lambda a: a.to_bytes(4, "big")


def putter(item_queue, result_queue, termination_event, db_path, exception_event):
    import lmdb
    with lmdb.open(db_path) as env:
        with env.begin(write=True) as txn:
            while not termination_event.is_set():
                try:
                    key, value = item_queue.get(timeout=1)
                    txn.put(key, value)
                except (TimeoutError, queue.Empty) as e:
                    pass


class LMDBCachedSeries(CachedSeries):
    """
    problem: not process safe.
    mmaped numpy is process safe. for both write and read?
    hdf5 write is not process safe.
    """

    def __init__(self, src, db_path, src_hash=None, **options):
        self.db_path = db_path
        self.flag_path = os.path.join(self.db_path, "flag.mmap")
        ensure_path_exists(self.flag_path)
        self._src = src
        self._indexer = IdentityIndexer(self.total)
        self.db_opts = options
        # keep lazy flags in memory,save it on lmdb.
        # I just want to edit a single bit, but do I have to write the whole buffer?
        self.create_flags()

    def create_flags(self):
        if os.path.exists(self.flag_path):
            mode = "r+"
        else:
            mode = "w+"
        self.flags = np.memmap(self.flag_path, dtype="bool", mode=mode, shape=(self.total,))

    def clone(self, parents):
        return LMDBCachedSeries(parents[0], self.db_path)

    def clear(self):
        os.remove(self.flag_path)
        self.create_flags()
        self.flags[:] = False
        for tgt in ["data.mdb", "lock.mdb"]:
            p = os.path.join(self.db_path, tgt)
            if os.path.exists(p):
                os.remove(p)

    @contextmanager
    def open_txn(self, write=False, buffers=False):
        import lmdb
        with lmdb.open(self.db_path, **self.db_opts) as env:
            with env.begin(buffers=buffers, write=write) as txn:
                yield txn

    @en_lazy
    def index_helper_array(self):
        return np.arange(self.total)

    def put(self, index: int, data):
        key = int2bytes(int(index))
        with self.open_txn(write=True) as txn:
            res = txn.put(key, pickle.dumps(data))
        self.flags[index] = True
        return res

    def get(self, index: int):
        key = int2bytes(int(index))
        with self.open_txn(buffers=True, write=False) as txn:
            return pickle.loads(txn.get(key))

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def total(self):
        return self.src.total

    def _get_item(self, index):
        if not self.flags[index]:
            self.put(index, self.src[index])
        return self.get(index)

    def _get_slice(self, _slice):
        return [self._get_item(int(i)) for i in self.index_helper_array[_slice]]

    def _get_indices(self, indices):
        return [self._get_item(int(i)) for i in self.index_helper_array[indices]]

    def _get_mask(self, mask):
        return [self._get_item(int(i)) for i in self.index_helper_array[mask]]


class PNGCachedSeries(CachedSeries):
    def __init__(self, src: Series, dir: str):
        super().__init__()
        self._src = src
        self._indexer = IdentityIndexer(total=src.total)
        self.dir = dir

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def total(self):
        return self.src.total

    @en_lazy
    def index_helper_array(self):
        return np.arange(self.total)

    def _get_item(self, index):
        img_path = os.path.join(self.dir, f"{index}.png")
        if os.path.exists(img_path):
            return Image.open(img_path)
        else:
            img: Image = self.src[index]
            img.save(img_path)
            return img

    def _get_slice(self, _slice):
        return [self._get_item(i) for i in self.index_helper_array[_slice]]

    def _get_indices(self, indices):
        return [self._get_item(i) for i in self.index_helper_array[indices]]

    def _get_mask(self, mask):
        return [self._get_item(i) for i in self.index_helper_array[mask]]


class Hdf5AutoSeries(CachedSeries):
    @property
    def src(self):
        return self._src

    def clone(self, parents):
        return Hdf5AutoSeries(src=parents[0], cache_path=self.cache_path, format=self.format)

    def __init__(self, src, cache_path, format, src_hash):
        self._src = src
        self.cache_path = cache_path
        self.format = format  # format ?? src[0].format
        self.cache = Hdf5CachedSeries(self._src.map(lambda a: a.value), cache_path, src_hash=src_hash)
        from proboscis_image_rules.rulebook import legacy_auto as auto
        self.auto = auto(self.format)
        self._indexer = IdentityIndexer(self.src.total)

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._src.total

    def ensure(self, _slice=None, batch_size=32, check_non_zero=False, preload=5):
        return self.cache.ensure(_slice, batch_size, check_non_zero, preload)

    def _get_item(self, index):
        return self.auto(self.cache[index])

    def _get_mask(self, mask):
        return [self.auto(a) for a in self.cache._get_mask(mask)]

    def _get_indices(self, indices):
        return [self.auto(a) for a in self.cache._get_indices(indices)]

    def _get_slice(self, _slice):
        return [self.auto(a) for a in self.cache._get_slice(_slice)]


class LRUSeries(SourcedSeries):
    def __init__(self, src, max_memo=1000):
        self._src = src
        self._indexer = IdentityIndexer(self._src.total)
        self.indices = np.arange(self._src.total)
        self.max_memo = 1000
        self.cache = LRU(size=max_memo)

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def total(self):
        return self._src.total

    def _get_item(self, index):
        i = int(index)
        if i not in self.cache:
            self.cache[i] = self.src[i]
        return self.cache[i]

    def _smart_get(self, smart):
        data = [self._get_item(i) for i in self.indices[smart]]
        if isinstance(data[0], np.ndarray):
            return np.array(data)
        else:
            return data

    def _get_slice(self, _slice):
        return self._smart_get(_slice)

    def _get_indices(self, indices):
        return self._smart_get(indices)

    def _get_mask(self, mask):
        return self._smart_get(mask)


class StructuredHdf5:
    def __init__(self, hdf5file, key, size):
        self.size = size
        self.hdf5file = hdf5file
        self.key = key

    def set_value(self, index, value):
        pass

    def smart_get(self, smart):
        pass


def sample2SHdf5(hdf5file, key, sample, size):
    if isinstance(sample, numbers.Number):
        return NumberSHdf5(hdf5file, key, sample, size=size)
    elif isinstance(sample, np.ndarray):
        return NumberSHdf5(hdf5file, key, sample, size=size)
    elif isinstance(sample, dict):
        return DictSHdf5(hdf5file, key, sample, size)
    elif isinstance(sample, tuple):
        return TupleSHdf5(hdf5file, key, sample, size)
    else:
        raise RuntimeError(f"Structured Hdf5 does not support ")


class NumberSHdf5(StructuredHdf5):
    def __init__(self, hdf5file, key, sample, size):
        super().__init__(hdf5file, key, size)
        # TODO create dataset
        if isinstance(sample, int):
            sample = np.int64(sample)
        elif isinstance(sample, float):
            sample = np.float64(sample)
        f = hdf5file
        f.create_dataset("value", shape=shape, dtype=sample.dtype, **self.dataset_opts)

        self.dataset = hdf5file[key]
        ensure_path_exists(self.cache_path)
        with self.lock:  # dont let multiple thread check this.
            # why is this repeatedly called in batch gen?
            # logger.info(
            #    f"trying to open hdf5 for preparation at thread:{threading.currentThread().name} |pid:{os.getpid()} ")
            with h5py.File(self.cache_path, mode="a") as f:  # tries to open lock even though it is locked..
                # sometimes the file is already open after ensured.
                if "src_hash" in f.attrs and f.attrs["src_hash"] != self.src_hash:
                    os.remove(self.cache_path)
                    logger.warning(f"deleted cache due to inconsistent hash of source. {self.cache_path}")

            with h5py.File(self.cache_path, mode="a") as f:
                if "src_hash" not in f.attrs:
                    f.attrs["src_hash"] = self.src_hash
                if not all(map(lambda item: item in f, "value flag".split())):
                    sample = self.src[0]
                    if isinstance(sample, int):
                        sample = np.int64(sample)
                    elif isinstance(sample, float):
                        sample = np.float64(sample)
                    if "value" not in f:
                        if hasattr(sample, "shape"):
                            shape = (self.total, *sample.shape)
                        else:
                            shape = (self.total,)
                        if sample.dtype == np.float64:
                            logger.warning(f"this dataset({self.__class__.__name__}) returns value with dtype:float64 ")

                        if self.dataset_opts.get("chunks") is True:
                            items_per_chunk = max(1024 * 1024 * 1 // sample.nbytes, 1)  # chunk is set to 1 MBytes

                            self.dataset_opts["chunks"] = (items_per_chunk, *sample.shape)
                        logger.info(f"dataset created with options:{self.dataset_opts}")
                        logger.warning(f"dataset dtype: {sample.dtype}")
                        f.create_dataset("value", shape=shape, dtype=sample.dtype, **self.dataset_opts)
                        logger.info(f"created value in hdf5")
                    if "flag" not in f:
                        f.create_dataset("flag", data=np.zeros((self.total,), dtype=np.bool))
                        logger.info(f"created flag in hdf5")
                    logger.info(f"created hdf5 cache at {self.cache_path}.")
                    f.flush()
                    logger.info(f"{list(f.keys())}")


class TupleSHdf5(StructuredHdf5):
    def __init__(self, hdf5file, key, sample, size):
        super().__init__(hdf5file, key, size)


class DictSHdf5(StructuredHdf5):
    def __init__(self, hdf5file, key, sample, size):
        super().__init__(hdf5file, size)
