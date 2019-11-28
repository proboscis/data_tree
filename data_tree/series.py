import abc
import multiprocessing
import os
import pickle
import queue
import shelve
import threading
from concurrent.futures import Executor, Future
from datetime import datetime
from hashlib import sha1
from typing import Iterable, NamedTuple, Union, Mapping, Callable, List

import PIL
import h5py
import numpy as np
from IPython.core.display import display
from PIL import Image
from frozendict import frozendict
from lazy import lazy as en_lazy
from logzero import logger
from tqdm.autonotebook import tqdm

from data_tree.indexer import Indexer, IdentityIndexer
from data_tree.mp_util import GlobalHolder
from data_tree.resource import ContextResource, Resource
from data_tree.util import ensure_path_exists, batch_index_generator, load_or_save, prefetch_generator
from ipywidgets import interactive


def en_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        t = object
        if data:
            sample = data[0]
            if isinstance(sample, int):
                t = np.int64
            elif isinstance(sample, float):
                t = np.float64
        buf = np.empty(len(data), dtype=t)
        buf[:] = data
        return buf


class Trace(NamedTuple):
    parents: List["Trace"]
    series: "Series"
    index: int
    metadata: dict

    def get_value(self):
        return self.series[self.index]

    def trace_string(self, metadata_generator=None):

        def _print_trace(trace, indent):
            indent_str = '-' * indent
            meta = trace.metadata.copy()
            if metadata_generator is not None:
                meta.update(metadata_generator(trace))

            buf = f"{indent_str}:{str(trace.series)} | {meta}\n"
            if not trace.parents:
                buf += indent_str + ">" + str(trace.get_value()) + "\n"
            else:
                for p in trace.parents:
                    buf += _print_trace(p, indent + 1)
            return buf

        return _print_trace(self, 0)

    def print_trace(self):
        print(self.trace_string())
        """
        def _print_trace(trace, indent):
            indent_str = '-' * indent
            print(f"{indent_str}:{str(trace.series.__class__.__name__)} | {trace.metadata}")
            if not trace.parents:
                print(indent_str + ">" + str(trace.get_value()))
            else:
                for p in trace.parents:
                    _print_trace(p, indent + 1)

        return _print_trace(self, 0)
        """

    def check_time(self):
        def _time(t: Trace):
            start = datetime.now()
            res = t.get_value()
            end = datetime.now()
            return dict(time=(end - start))

        print(self.trace_string(metadata_generator=_time))

    def traverse(self):
        yield self
        for p in self.parents:
            yield from p.traverse()

    # you cannot use lazy with namedtuple!
    def ancestors(self):
        return Series.from_iterable(list(self.traverse()))


    def find_by_tag(self, tag: str):
        return [t for t in self.compiled().traverse() if "tags" in t.metadata and tag in t.metadata["tags"]]

    def compiled(self):
        if isinstance(self.series, MetadataOpsSeries):
            m = self.metadata
            return self.parents[0]._replace(metadata=self.series.operator(m)).compiled()
        else:
            return self._replace(parents=[p.compiled() for p in self.parents])

    def __repr__(self):
        return self.trace_string()

    def _repr_html(self):
        return self.__repr__()

    def inferred_visualization(self):
        value = self.get_value()
        if isinstance(value, np.ndarray):
            if len(value.shape) == 3:
                return Image.fromarray(value, mode="RGB")
            elif len(value.shape) == 2:
                return Image.fromarray(value, mode="L")
            else:
                return value
        elif isinstance(value, PIL.Image.Image):
            return value
        else:
            return value

    def visualization(self):
        viz = self.metadata.get("visualization")
        if viz is None:
            viz = self.inferred_visualization()
        elif callable(viz):
            viz = viz(self.get_value())
        return viz

    def show_traceback_html(self, skipping_tags=("slow",)):
        for t in self.traverse():
            tags = t.metadata.get("tags")
            ignored = False
            if tags is not None:
                for tag in skipping_tags:
                    if tag in tags:
                        ignored = True
                        break
            if ignored:
                continue
            display(t.visualization())


class Series(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def indexer(self) -> Indexer:
        pass

    @property
    @abc.abstractmethod
    def total(self):
        pass

    def __len__(self):
        return self.total

    @abc.abstractmethod
    def _get_item(self, index):
        pass

    @abc.abstractmethod
    def _get_slice(self, _slice):
        pass

    @abc.abstractmethod
    def _get_indices(self, indices):
        pass

    @abc.abstractmethod
    def _get_mask(self, mask):
        pass

    def _trace(self, index) -> Trace:
        return Trace(parents=[], series=self, index=index, metadata=dict())

    def trace(self, index) -> Trace:
        return self._trace(index).compiled()

    def _values(self, item):
        # logger.debug(f"{self} _values at:{item}")
        smart_indices = self.indexer[item]
        if isinstance(smart_indices, slice):
            return self._get_slice(smart_indices)
        elif isinstance(smart_indices, np.ndarray):
            if smart_indices.dtype == np.bool:
                return self._get_mask(smart_indices)
            else:
                return self._get_indices(smart_indices)
        else:
            return self._get_item(smart_indices)

    def __getitem__(self, item) -> Union["Series", object]:
        if isinstance(item, slice):
            return self.indexed(IdentityIndexer(self.total).slice(item))
        elif isinstance(item, np.ndarray):
            if item.dtype == np.bool:
                return self.indexed(IdentityIndexer(self.total).mask(item))
            else:
                return self.indexed(IdentityIndexer(self.total).transform(item))
        elif isinstance(item, Iterable):
            return self[en_numpy(item)]
        else:
            return self._get_item(self.indexer.get_index(item))

    def indexed(self, indexer: Indexer):
        return Indexed(self, indexer)

    def map(self, f, batch_f=None):
        return MappedSeries(self, f, batch_f)

    def mp_map(self, f, global_resources: Union[None, dict] = None, num_process=None):
        return MPMappedSeries(self, f, resources_kwargs=global_resources, num_process=num_process).lazy()

    @property
    def values(self):
        return self._values(slice(None))

    def hdf5(self, cache_path, src_hash=None, **dataset_opts):
        return Hdf5CachedSeries(self, cache_path=cache_path, src_hash=src_hash, **dataset_opts)

    def shelve(self, cache_path, src_hash=None):
        return ShelveSeries(self, cache_path=cache_path, src_hash=src_hash)

    def pkl(self, cache_path, src_hash=None):
        return PickledSeries(self, pickle_path=cache_path, src_hash=src_hash)

    def lazy(self):
        return LazySeries(self)

    def flatten(self):
        return FlattenedSeries(self)

    def tag(self, *tags):
        def _add_tag(m: dict):
            copied = m.copy()
            old_tags = copied.get("tags")
            if old_tags is None:
                old_tags = set()
            new_tags = old_tags | set(tags)
            copied["tags"] = new_tags
            return copied

        return MetadataOpsSeries(self, operator=_add_tag)

    def update_metadata(self, **metadata):
        def _update_metadata(m: dict):
            # print(f"incoming:{m}")
            copied = m.copy()
            copied.update(metadata)
            # print(f"result:{copied}")
            return copied

        return MetadataOpsSeries(self, operator=_update_metadata)

    def visualization(self, visualizer):
        return self.update_metadata(visualization=visualizer)

    @staticmethod
    def from_numpy(values: np.ndarray):
        return NumpySeries(values)

    @staticmethod
    def from_iterable(values):
        if isinstance(values, np.ndarray):
            return Series.from_numpy(values)
        else:
            return ListSeries(list(values))

    @staticmethod
    def from_lambda(f, total):
        return LambdaAdapter(f, total)

    @staticmethod
    def from_numpy_like(data):
        return LambdaAdapter(lambda s: data[s], len(data))

    @en_lazy
    def traces(self):
        return TraceSeries(self)

    def zip(self, tgt: "Series"):
        return ZippedSeries(self, tgt)

    def zip_with_index(self):
        return ZippedSeries(self, NumpySeries(np.arange(self.total)))

    @en_lazy
    def hash(self):
        def _freeze(item):
            if isinstance(item, dict):
                return frozendict((_freeze(k), _freeze(v)) for k, v in item.items())
            elif isinstance(item, list):
                return tuple(_freeze(i) for i in item)
            elif isinstance(item, np.ndarray):
                return tuple(_freeze(i) for i in item)
            return item

        def _hash():
            return sha1(str(_freeze(self.values)).encode()).hexdigest()

        ha, hb = _hash(), _hash()
        assert ha == hb, f"{self}=> hash changed! elements may contain mutable values"
        return ha

    def with_hash(self, f):
        """
        :param f: (hash,self) => other series
        :return:
        """
        return f(self.hash, self)

    def with_(self, f, file_path: str, kind: str = "pkl"):
        return f(self).__getattribute__(kind)(file_path, src_hash=self.hash)

    @staticmethod
    def from_constructor(self, initializer: Callable, cache_path: str, cache_kind: str = "hdf5"):
        lazy_src = None

        def get_src():
            nonlocal lazy_src
            lazy_src = initializer()
            return lazy_src

        total = load_or_save(cache_path + ".total.pkl", lambda: get_src().total)
        return LambdaAdapter(slicer=lambda s: get_src()._values(s), total=total).__getattribute__(self.cache_kind)(
            self.cache_path + "." + cache_kind)

    def cached_transform(self, transformer_constructor: Callable, cache_path: str):
        transformer = load_or_save(cache_path, transformer_constructor)
        return self[transformer]

    def interact(self):
        def _l(i):
            self.trace(i).show_traceback_html()

        return interactive(_l, i=(0, len(self) - 1, 1))

    def shuffle(self):
        permutation = np.random.permutation(self.total)
        return self[permutation].tag("shuffle")

    @staticmethod
    def split3_indices(train_amount, valid_amount, total):
        train_end = int(total * train_amount)
        train_start = 0
        valid_start = train_end
        valid_end = int(total * (train_amount + valid_amount))
        test_start = valid_end
        test_end = total - 1
        return [
            (train_start, train_end),
            (valid_start, valid_end),
            (test_start, test_end)
        ]

    def split(self, ratio):
        left = int(self.total * ratio)
        return self[0:left], self[left:]

    def split3(self, train_amount, valid_amount):
        return [self[start:end] for start, end in Series.split3_indices(train_amount, valid_amount, self.total)]

    def __iter__(self):
        def gen():
            for i in range(len(self)):
                yield self[i]

        return gen

    def slice_generator(self, slices, preload=5, en_numpy=False):
        def batches():
            for _slice in slices:
                batch = self[_slice].values
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        if preload:
            assert preload > 0, "preload must be positive integer"
            yield from prefetch_generator(batches(), preload)
        else:
            yield from batches()
        pass

    def batch_generator(self, batch_size, preload=5, offset=0, en_numpy=False):
        def _slices():
            for bs, be in batch_index_generator(offset, self.total, batch_size):
                yield slice(bs, be, 1)

        yield from self.slice_generator(_slices(), preload=preload, en_numpy=en_numpy)

    def schedule_on(self, executor: Executor):
        return ScheduledSeries(self, executor)

    def mbps(self, mbytes=100):
        """
        :param mbytes:
        :return: mega bytes per second
        """
        sample = np.array(self[0])
        assert np.issubdtype(sample.dtype, np.number), f"you cannot check mbps on non numeric series. {sample}"
        items_to_read = min(max(mbytes * 1024 * 1024 // sample.nbytes, 1), self.total)
        logger.info(f"checking read mbps by accessing {items_to_read} items")
        start = datetime.now()
        read_values = self[:items_to_read].values
        end = datetime.now()
        read_values = np.array(read_values)
        dt = end - start
        return read_values.nbytes / 1024 / 1024 / dt.total_seconds()

    def sps(self,batch_size=512):
        """
        :param batch_size:
        :return: sample per second
        """
        start = datetime.now()
        _ = self[:batch_size].values
        end = datetime.now()
        dt = end - start
        return batch_size / dt.total_seconds()

    def __str__(self):
        return f"{self.__class__.__name__} #{id(self)}"

    def __repr__(self):
        return self.trace(0).trace_string()


class ZippedSeries(Series):
    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def total(self):
        return self._total

    def _get_item(self, index):
        return self.a._values(index), self.b._values(index)

    def _get_slice(self, _slice):
        return list(zip(self.a._values(_slice), self.b._values(_slice)))

    def _get_indices(self, indices):
        return list(zip(self.a._values(indices), self.b._values(indices)))

    def _get_mask(self, mask):
        return list(zip(self.a._values(mask), self.b._values(mask)))

    def __init__(self, a: Series, b: Series):
        self._total = min(a.total, b.total)
        self._indexer = IdentityIndexer(self._total)
        self.a = a
        self.b = b

    def _trace(self, index) -> Trace:
        i = self.indexer[index]
        return Trace(
            parents=[self.a._trace(i), self.b._trace(i)],
            index=i,
            series=self,
            metadata=dict()
        )


class LambdaAdapter(Series):
    def __init__(self, slicer, total):
        self.slicer = slicer
        self._indexer = IdentityIndexer(total=total)

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._indexer.total

    def _get_item(self, index):
        return self.slicer(index)

    def _get_slice(self, _slice):
        return self.slicer(_slice)

    def _get_indices(self, indices):
        return self.slicer(indices)

    def _get_mask(self, mask):
        return self.slicer(mask)


class Hdf5Adapter(Series):
    def __init__(self, hdf5_initializer, hdf5_file_name, key):
        if not os.path.exists(hdf5_file_name):
            hdf5_initializer()
        self.hdf5_file = h5py.File(hdf5_file_name, "r")
        self.data = self.hdf5_file[key]
        self._indexer = IdentityIndexer(total=len(self.data))

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._indexer.total

    def _get_item(self, index):
        return self.data[index]

    def _get_slice(self, _slice):
        return self.data[_slice]

    def _get_indices(self, indices):
        return self.data[indices]

    def _get_mask(self, mask):
        return self.data[mask]


class Hdf5OpenFileAdapter(Series):
    def __init__(self, hdf5_initializer, open_file, key):
        if key not in open_file:
            hdf5_initializer()
        self.hdf5_file = open_file
        self.data = self.hdf5_file[key]
        self._indexer = IdentityIndexer(total=len(self.data))

    @property
    def indexer(self):
        return self._indexer

    @property
    def total(self):
        return self._indexer.total

    def _get_item(self, index):
        return self.data[index]

    def _get_slice(self, _slice):
        return self.data[_slice]

    def _get_indices(self, indices):
        return self.data[indices]

    def _get_mask(self, mask):
        return self.data[mask]


class SourcedSeries(Series):
    @property
    @abc.abstractmethod
    def src(self) -> Series:
        pass

    def _trace(self, index):
        i = self.indexer[index]
        return Trace(
            [self.src._trace(i)],
            index=index,
            series=self,
            metadata=dict()
        )


class TraceSeries(SourcedSeries):
    def __init__(self, src: Series):
        self._indexer = IdentityIndexer(src.total)
        self._src = src

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
        return self._src.trace(index)

    def _get_slice(self, _slice: slice):
        start, stop, step = _slice.indices(self.total)
        return [self.src.trace(i) for i in range(start, stop, step)]

    def _get_indices(self, indices):
        return [self.src.trace(i) for i in indices]

    def _get_mask(self, mask):
        return self._get_indices(np.arange(self.total)[mask])


class PickledSeries(SourcedSeries):
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


class FlattenedSeries(SourcedSeries):
    def __init__(self, src):
        self._src = src
        self._index = IdentityIndexer(total=sum([s.total for s in src.values]))
        self.mapping = np.zeros(self.total, dtype=int)
        self.offsets = np.zeros(self.total, dtype=int)
        self._range = np.arange(self.total)
        i = 0
        count = 0
        for s in src.values:
            self.mapping[count:count + s.total] = i
            self.offsets[count:count + s.total] = np.arange(s.total)
            i += 1
            count += s.total

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._index

    @property
    def total(self):
        return self.indexer.total

    def _slice_to_slices(self, _slice: slice):
        # TODO
        raise NotImplementedError()

    def _get_item(self, index):
        return self.src.values[self.mapping[index]]._get_item(self.offsets[index])

    def _get_slice(self, _slice):
        if _slice.start is None and _slice.stop is None:
            buf = []
            for s in self.src.values:
                buf += s.values
            return buf
        else:
            return [self._get_item(i) for i in self._range[_slice]]

    def _get_indices(self, indices):
        return [self._get_item(i) for i in self._range[indices]]

    def _get_mask(self, mask):
        return [self._get_item(i) for i in self._range[mask]]

    def _trace(self, index) -> Trace:
        src2 = self.src.values[self.mapping[index]]
        return Trace(
            parents=[self.src._trace(self.mapping[index]), src2._trace(self.offsets[index])],
            series=self,
            index=index,
            metadata=dict()
        )


class ShelveSeries(SourcedSeries):

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


class IndexedSeries(SourcedSeries):
    # TODO fix these accesors to respect indices
    @property
    def total(self):
        return self.indexer.total

    def _get_item(self, index):
        return self.src._values(index)

    def _get_slice(self, _slice):
        return self.src._values(_slice)

    def _get_indices(self, indices):
        return self.src._values(indices)

    def _get_mask(self, mask):
        return self.src._values(mask)


class ScheduledSeries(IndexedSeries):

    def __init__(self, src: Series, executor: Executor):
        self.scheduler = executor
        self._src = src
        self._indexer = IdentityIndexer(self.src.total)

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    def slice_generator(self, slices, preload=5, en_numpy=False):
        q = queue.Queue(preload)

        DONE = "$$DONE$$"

        def getter(_s):
            return self.src[_s]

        def waiter():
            for s in slices:
                values = self.scheduler.submit(getter, args=(s,))
                q.put(values.result())
            q.put(DONE)

        waiting_thread = threading.Thread(target=waiter)
        waiting_thread.start()
        while True:
            if q.qsize() == 0:
                logger.warning(f"preload queue is empty!")
            item = q.get()
            if item is DONE:
                break
            yield item

        return super().slice_generator(slices, preload, en_numpy)


class TaggedSeries(IndexedSeries):
    def __init__(self, src: Series, *tags):
        self.tags = tags
        self._src = src
        self._indexer = IdentityIndexer(self.src.total)

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    def __str__(self):
        return f"TaggedSeries<{self.tags}> at {id(self)}"


class MetadataOpsSeries(IndexedSeries):
    def __init__(self, src: Series, operator: Callable):
        self.operator = operator
        self._src = src
        self._indexer = IdentityIndexer(self.src.total)

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    def __str__(self):
        return f"MetadataOpsSeries<{self.operator.__name__}> at {id(self)}"


class Indexed(IndexedSeries):
    def __init__(self, src, indexer):
        self._src = src
        self._indexer = indexer

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self):
        return self._indexer


class MappedSeries(IndexedSeries):
    @property
    def indexer(self):
        return self._indexer

    @property
    def src(self) -> Series:
        return self._src

    def __init__(self, src: Series, single_mapper=None, slice_mapper=None):
        self._src = src
        self._indexer = IdentityIndexer(self.total)
        self.single_mapper = single_mapper
        self.slice_mapper = slice_mapper
        super().__init__()

    @property
    def total(self):
        return self._src.total

    def _get_item(self, index):
        return self.single_mapper(self.src._values(index))

    def _map_values(self, values):
        if self.slice_mapper is not None:
            return self.slice_mapper(values)
        else:
            return [self.single_mapper(x) for x in values]

    def _get_slice(self, _slice):
        src_vals = self.src._values(_slice)
        return self._map_values(src_vals)

    def _get_indices(self, indices):
        src_vals = self.src._values(indices)
        return self._map_values(src_vals)

    def _get_mask(self, mask):
        src_vals = self.src._values(mask)
        return self._map_values(src_vals)

    def slice_generator(self, slices, preload=5, en_numpy=False):
        def src_batches():
            for _slice in slices:
                batch = self.src[_slice].values
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        if preload:
            assert preload > 0, "preload must be positive integer"
            _src_batches = prefetch_generator(src_batches(), preload)
        else:
            _src_batches = src_batches()
        for batch in _src_batches:
            yield self._map_values(batch)


def mp_mapper(f, global_resources: Mapping[str, GlobalHolder], **kwargs):
    resources = {k: g.value for k, g in global_resources.items()}
    result = f(**resources, **kwargs)
    return result


class MPMappedSeries(IndexedSeries):
    @property
    def indexer(self):
        return self._indexer

    @property
    def src(self) -> Series:
        return self._src

    def __init__(self, src: Series, single_mapper, resources_kwargs: dict = Union[None, Mapping[str, Resource]],
                 num_process=None):
        self._src = src
        self._indexer = IdentityIndexer(self.total)
        self.single_mapper = single_mapper
        if resources_kwargs is None:
            resources_kwargs = dict()
        self.resources = resources_kwargs
        self.num_process = num_process
        super().__init__()

    @property
    def total(self):
        return self._src.total

    def _map_values(self, values):
        resources = {k: r.prepare() for k, r in self.resources.items()}
        with multiprocessing.Pool(processes=self.num_process) as pool:
            futures = [pool.apply_async(mp_mapper, args=(self.single_mapper, resources), kwds=v) for v in values]
            results = [f.get() for f in tqdm(futures, desc="waiting for mp map results")]
        for k, r in self.resources.items():
            r.release(r)
        return results

    def _get_item(self, index):
        return self._map_values([self.src._values(index)])[0]

    def _values(self, item):
        src_vals = self.src._values(item)
        if isinstance(item, slice) or isinstance(item, np.ndarray) or isinstance(item, Iterable):
            return self._map_values(src_vals)
        else:
            return self._map_values([src_vals])[0]

    def _get_slice(self, _slice):
        src_vals = self.src._values(_slice)
        return self._map_values(src_vals)

    def _get_indices(self, indices):
        src_vals = self.src._values(indices)
        return self._map_values(src_vals)

    def _get_mask(self, mask):
        src_vals = self.src._values(mask)
        return self._map_values(src_vals)


class LazySeries(IndexedSeries):

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


class NumpySeries(Series):
    @property
    def indexer(self):
        return self._indexer

    def __init__(self, data: np.ndarray):
        self.data = data
        self._indexer = IdentityIndexer(len(data))
        super().__init__()

    @property
    def total(self):
        return len(self.data)

    def _get_item(self, index):
        return self.data[index]

    def _get_slice(self, _slice):
        return self.data[_slice]

    def _get_indices(self, indices):
        return self.data[indices]

    def _get_mask(self, mask):
        return self.data[mask]

    def _values(self, item):
        return self.data[item]


class ListSeries(Series):
    @property
    def indexer(self):
        return self._indexer

    def __init__(self, data: list):
        self.data = data
        self._indexer = IdentityIndexer(len(data))
        super().__init__()

    @en_lazy
    def _indexer(self):
        return np.arange(self.total)

    @property
    def total(self):
        return len(self.data)

    def _get_item(self, index):
        return self.data[index]

    def _get_slice(self, _slice):
        return self.data[_slice]

    def _get_indices(self, indices):
        return [self.data[i] for i in indices]

    def _get_mask(self, mask):
        return self._values(self._indexer[mask])


class NumpyCache(NumpySeries, SourcedSeries):

    @property
    def src(self) -> Series:
        return self._src

    def __init__(self, src: Series):
        self._src = src
        super().__init__(self.src.values)


class Hdf5CachedSeries(SourcedSeries):
    @property
    def src(self) -> Series:
        return self._src

    def __init__(self, src: Series, cache_path: str, src_hash=None, **dataset_opts):
        self.cache_path = cache_path
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
            item_queue = queue.Queue(1000)
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
                    i = 0
                    while True:
                        item = item_queue.get()
                        if item is None:
                            break
                        _s, x = item
                        # logger.info(f"item inside: {item}")
                        xs[_s] = x
                        flags[_s] = True
                        if i % 10000 == 0:
                            f.flush()
                        i += 1

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
