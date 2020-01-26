import os
import pickle
import queue
import shelve
import threading

import h5py
import numpy as np
from lazy import lazy as en_lazy
from logzero import logger
from tqdm._tqdm_notebook import tqdm_notebook

from data_tree import Series, IdentityIndexer, Indexer
from data_tree.resource import ContextResource
from data_tree._series import SourcedSeries, IndexedSeries, NumpySeries
from data_tree.util import ensure_path_exists, batch_index_generator, prefetch_generator
import data_tree._series as series
from tqdm.autonotebook import tqdm


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
                        logger.warn(f"found data is corrupt:{loaded_hash} != {self.src_hash}")
                        failed = True
                except Exception as e:
                    logger.warn(f"failed to load pkl:{e}")
                    failed = True
                if failed:
                    os.remove(self.pickle_path)
                    logger.warn(f"removed corrupt cache at {self.pickle_path}")

        with open(self.pickle_path, "wb") as f:
            data = self.src.values
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
        if self.shelve_file.map(lambda db: "src_hash" in db and db["src_hash"] != src_hash).get():
            os.remove(self.cache_path)
        with self.shelve_file.to_context() as db:
            if "src_hash" not in db:
                db["src_hash"] = src_hash

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


class LazySeries(CachedSeries):

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
        self._src = src
        self._indexer = IdentityIndexer(self.src.total)
        self.flags = np.zeros(src.total, dtype=bool)
        self.cache = [None] * self.total
        self._index = np.arange(src.total)
        self.prefer_slice = prefer_slice

    def _get_item(self, index):
        if not self.flags[index]:
            self.cache[index] = self.src._values(index)
            self.flags[index] = True
        return self.cache[index]

    def _ensure_and_return_smart(self, smart_indexer):
        if not self.flags[smart_indexer].all():
            non_cached_indices = self._index[smart_indexer][~self.flags[smart_indexer]]
            if self.prefer_slice and isinstance(smart_indexer, slice):
                self.cache[smart_indexer] = self.src._values(smart_indexer)
                self.flags[smart_indexer] = True

            else:
                for v, i in zip(self.src._values(non_cached_indices), non_cached_indices):
                    self.cache[i] = v
                self.flags[non_cached_indices] = True
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

    def __init__(self, src: Series, cache_path: str, src_hash=None, **dataset_opts):
        self.cache_path = cache_path
        logger.info(f"initialized hdf5 cache series at {cache_path}")
        self._src = src
        self._indexer = IdentityIndexer(self.total)
        self.src_hash = "None" if src_hash is None else src_hash

        self.dataset_opts = dataset_opts

    def prepared(self):
        ensure_path_exists(self.cache_path)
        with h5py.File(self.cache_path, mode="a") as f:
            if "src_hash" in f.attrs and f.attrs["src_hash"] != self.src_hash:
                os.remove(self.cache_path)
                logger.warn(f"deleted cache due to inconsistent hash of source. {self.cache_path}")

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
                        logger.warn(f"this dataset({self.__class__.__name__}) returns value with dtype:float64 ")

                    if self.dataset_opts.get("chunks") is True:
                        items_per_chunk = max(1024 * 1024 * 1 // sample.nbytes, 1)  # chunk is set to 1 MBytes

                        self.dataset_opts["chunks"] = (items_per_chunk, *sample.shape)
                    logger.info(f"dataset created with options:{self.dataset_opts}")
                    logger.warn(f"dataset dtype: {sample.dtype}")
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
        with h5py.File(self.cache_path, mode="r+") as f:
            flags = f["flag"]
            flags_on_mem = flags[:]
            with h5py.File(self.cache_path, mode="r") as f:
                xs = f["value"]
                for i in tqdm(range(*_slice.indices(_slice.stop))):
                    if flags_on_mem[i]:
                        x = xs[i]
                        if (x == 0).all():
                            flags[i] = False
                            flags_on_mem[i] = False
            return flags_on_mem

    def ensure(self, _slice=None, batch_size=1000, check_non_zero=False, preload=5):
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
                with h5py.File(self.cache_path, mode="r+") as f:
                    flags = f["flag"][:]

            def missing_batches():
                start, end, step = _slice.indices(len(self))
                batch_indices = list(batch_index_generator(start, end, batch_size))
                missing_slices = []
                for bs, be in batch_indices:
                    if not flags[bs:be].all():
                        missing_slices.append(slice(bs, be, 1))
                yield from progress_bar(zip(missing_slices, self.src.slice_generator(missing_slices, preload=preload)),
                                        desc=f"{self.__class__.__name__}:filling missing cache batches",
                                        total=len(missing_slices))

            def saver():
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

            t = threading.Thread(target=saver)
            t.start()
            for item in missing_batches():
                item_queue.put(item)
            item_queue.put(None)
            t.join()

    def clear(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
            logger.warn(f"deleted cache at {self.cache_path}")

    def _get_item(self, index):
        if self.prepared():
            with h5py.File(self.cache_path, mode="r+") as f:
                if f["flag"][index]:
                    return f["value"][index]
                else:
                    x = self.src._values(index)
                    f["value"][index] = x
                    f["flag"][index] = True
                    return x

    def _get_slice(self, _slice):
        if self.prepared():
            with h5py.File(self.cache_path, mode="r+") as f:  # opening file takes 20ms
                if f["flag"][_slice].all():
                    # logger.info(f"cache is ready for this slice")
                    return f["value"][_slice]
                else:
                    self.ensure(_slice)
                    return f["value"][_slice]

    def _get_indices(self, indices):
        logger.warn(f"indices access on hdf5 is slow")
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
