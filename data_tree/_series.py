import abc
import multiprocessing
import os
import queue
import threading
from collections import OrderedDict
from concurrent.futures import Executor
from datetime import datetime
from hashlib import sha1
from typing import Iterable, NamedTuple, Union, Mapping, Callable, List

import h5py
import numpy as np
from IPython.core.display import display
from frozendict import frozendict
from ipywidgets import interactive
from lazy import lazy as en_lazy
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from tqdm.autonotebook import tqdm

from data_tree.cache import ConditionedFilePathProvider
from data_tree.coconut.visualization import infer_widget
from data_tree.indexer import Indexer, IdentityIndexer
from data_tree.mp_util import GlobalHolder, SequentialTaskParallel2

from data_tree.resource import Resource
from data_tree.util import batch_index_generator, load_or_save, prefetch_generator, Pickled, shared_npy_array_like
import ipywidgets as widgets
from pprintpp import pformat
from pyvis.network import Network


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


def series_to_tree(s):
    got_net = Network(notebook=True, directed=True)
    # got_net.barnes_hut()#physics
    got_net.add_node(str(s), str(s), title=str(s))
    for a in s.ancestors.values:
        got_net.add_node(str(a), str(a), title=str(a))

    def _dfs(s):
        for p in s.parents:
            got_net.add_edge(str(s), str(p))
            _dfs(p)

    _dfs(s)
    # got_net.show_buttons()
    return got_net.show("ancestors.html")


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

    def find_one_by_tag(self, tag: str):
        return self.find_by_tag(tag)[0]

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

    def visualization(self):
        viz = self.metadata.get("visualization")
        data = self.get_value()
        if viz is not None:
            viz = viz(data)
        elif isinstance(data, Trace):
            viz = widgets.HTML(value=repr(data).replace("\n", "<br>"))
        else:
            viz = infer_widget(data)
        return viz

    def show_traceback_html(self,
                            skipping_tags=("slow",),
                            showing_tags=None,
                            depth=None,
                            skip_indices=None,
                            show_indices=None):
        # import ipywidgets as widgets
        skipping_tags = set(skipping_tags) if skipping_tags is not None else ()
        showing_tags = set(showing_tags) if showing_tags is not None else set()

        traces = list(self.traverse())
        skip_mask = np.zeros(len(traces), dtype=bool)
        show_mask = np.zeros(len(traces), dtype=bool)
        skip_mask[skip_indices if skip_indices is not None else []] = True
        show_mask[show_indices if show_indices is not None else []] = True
        mask = np.ones(len(traces), dtype=bool)  # defaults to show everything
        if depth is None:
            depth = len(traces)
        for i, t in enumerate(traces):
            tags = t.metadata.get("tags")
            tags = set(tags) if tags is not None else set()
            ignored = bool(tags & skipping_tags)  # or skip_mask[i]
            shown = bool(tags & showing_tags)  # or show_mask[i]
            skip_mask[i] = skip_mask[i] or ignored
            show_mask[i] = show_mask[i] or shown

        # blacklist and whitelist
        # default is to show everything. so if show is specified, switch to not showing default
        if sum(show_mask):
            mask[:] = False
            mask[show_mask] = True
        else:
            mask[skip_mask] = False

        for i, t in enumerate(traces[:depth]):
            if mask[i]:
                vis = t.visualization()
                assert isinstance(vis,
                                  widgets.Widget), f"visualization must be an instance of widget. viz from {t} is {vis}"
                hbox = widgets.VBox([
                    widgets.Label(value=f"value at depth {i}"),
                    widgets.Label(value=str(t.metadata)),
                    vis
                ])
                hbox.layout.border = "solid 2px"
                display(hbox)


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

    @property
    def _metadata(self):
        return dict()

    @en_lazy
    def metadata(self):
        return dict(series=self, metadata=self._metadata, parents=[p.metadata for p in self.parents])

    @property
    @abc.abstractmethod
    def parents(self) -> List["Series"]:
        pass

    @en_lazy
    def ancestors(self):
        result = []
        for p in self.parents:
            result.append(p)
            result += p.ancestors
        return Series.from_iterable(result)

    @en_lazy
    def traversed(self):
        return Series.from_iterable([self] + list(self.ancestors))

    @property
    def traversed_metadata(self):
        return self.traversed.map(lambda m: m._metadata)

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

    def star_map(self, f):
        return MappedSeries(self, single_mapper=lambda args: f(*args))

    def map_b(self, batch_f):
        return MappedSeries(self, lambda b: batch_f(b[None])[0], batch_f)

    def mp_map(self, f, global_resources: Union[None, dict] = None, num_process=None,max_pending_result=8):
        """
        :param f: picklable function (Item,Dict(str,value from resources) => value)
        :param global_resources: Mapping[Str,Resource[GlobalHolder]]
        :param num_process:
        :return: beware f must be picklable and each element comes to first argument of f, and global resources will be given as kwargs.
        """
        return MPMappedSeries(self, f, resources_kwargs=global_resources, num_process=num_process,max_pending_result=max_pending_result)  # .lazy()

    @property
    def values(self):
        return self._values(slice(None))

    def hdf5(self, cache_path, src_hash=None, **dataset_opts):
        from data_tree.ops.cache import Hdf5CachedSeries
        # when you replace anything in the tree, this cache must be invalidated.
        # you need a renamed cache when the replaced tree was new
        # but naming is hard.. how about you provide names?
        # sin you know that all the cache files have distinct names, you need just one prefix.
        return Hdf5CachedSeries(self, cache_path=cache_path, src_hash=src_hash, **dataset_opts)

    def lmdb(self, cache_path, src_hash=None, **dataset_opts):
        from data_tree.ops.cache import LMDBCachedSeries
        # when you replace anything in the tree, this cache must be invalidated.
        # you need a renamed cache when the replaced tree was new
        # but naming is hard.. how about you provide names?
        # sin you know that all the cache files have distinct names, you need just one prefix.
        return LMDBCachedSeries(self,db_path=cache_path,src_hash=src_hash, **dataset_opts)

    def auto_hdf5(self, cache_path, src_hash=None, **dataset_opts):
        sample = self[0]
        # check sample and split until it becomes an array
        if isinstance(sample, str):
            raise RuntimeError("cannot make hdf5 cache for string values.")
        if isinstance(sample, Iterable):
            def converted():
                split_serires = [
                    self.map((lambda n: (lambda item: item[n]))(i)).auto_hdf5(cache_path=cache_path + f"_auto{i}",
                                                                              src_hash=src_hash,
                                                                              **dataset_opts)
                    for i in range(len(sample))]
                return ZippedSeries(*split_serires)

            try:
                numpied = np.array(sample)
            except ValueError as e:
                logger.info(f"element cannot be converted to numpy array")
                return converted()
            if numpied.dtype == np.object:  #
                return converted()
            else:
                return self.hdf5(cache_path, src_hash=src_hash, **dataset_opts)

    def condition(self, **conds: dict):
        def _add_condition(m: dict):
            copied: dict = m.copy()
            old_conds = copied.get("conditions")
            if old_conds is None:
                copied["conditions"] = conds
            else:
                old_conds.update(conds)
                copied["conditions"] = old_conds.copy()
            return copied

        return MetadataOpsSeries(self, operator=_add_condition)

    @en_lazy
    def conditions(self):
        conditions = dict()
        for m in self.traversed_metadata:
            conds = m.get("conditions")
            if conds is not None:
                conditions.update(conds)
        return conditions

    def managed_cache_path_provider(self, cache_root=None):
        if cache_root is None:
            for meta in reversed(list(self.traversed_metadata)):
                if "managed_cache_root" in meta:
                    cache_root = meta["managed_cache_root"]
                    logger.info(f"managed cache root is set to {cache_root} from series metadata")
                    break
        assert cache_root is not None, "no cache root is specified. you must set this in either parameter or in metadata"
        cfpp = ConditionedFilePathProvider(cache_root)
        return cfpp

    def managed_cache_path(self, file_name, cache_root=None):
        cfpp = self.managed_cache_path_provider(cache_root)
        return cfpp.get_managed_file_path(self.conditions, file_name)

    def managed_cache(self, cache_root=None, kind="auto_hdf5"):
        managed_path = self.managed_cache_path("managed_cache." + kind, cache_root=cache_root)
        if kind == "hdf5":
            return self.hdf5(managed_path)
        elif kind == "pkl":
            return self.pkl(managed_path)
        elif kind == "shelve":
            return self.shelve(managed_path)
        elif kind == "auto_hdf5":
            return self.auto_hdf5(managed_path)
        raise RuntimeError(f"kind :{kind} is not a supported cache type! use hdf5/pkl/shelve")

    def shelve(self, cache_path, src_hash=None):
        from data_tree.ops.cache import ShelveSeries  # Local imports to avoid circular imports
        return ShelveSeries(self, cache_path=cache_path, src_hash=src_hash)

    def pkl(self, cache_path, src_hash=None):
        from data_tree.ops.cache import PickledSeries
        return PickledSeries(self, pickle_path=cache_path, src_hash=src_hash)

    def lazy(self):
        from data_tree.ops.cache import LazySeries
        return LazySeries(self)

    def numpy(self):
        from data_tree.ops.cache import NumpyCache
        return NumpyCache(self)

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

    def zip(self, *tgts: "Series"):
        return ZippedSeries(self, *tgts)

    def zip_with_index(self):
        return ZippedSeries(self, NumpySeries(np.arange(self.total)))

    def unzip(self):
        sample = self[0]
        return tuple(self.map((lambda n: (lambda item: item[n]))(i)) for i in range(len(sample)))

    def sorted(self, key=None):
        """make sure everything can be loaded on to memory"""
        if key is not None:
            values_for_sort = self.map(key).values
        else:
            values_for_sort = self.values
        indices = np.argsort(values_for_sort)
        return self[indices]

    def filter(self, _filter):
        # filter must check whole element since a series requires predetermined length...
        # I can make this length lazy though
        mask = np.array(self.map(_filter).values, dtype=bool)
        return self[mask]

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
        return self[Pickled(cache_path, transformer_constructor).value]

    def shuffle(self, seed=42):
        idx = np.random.RandomState(seed=seed).permutation(self.total)
        return self[idx].tag("shuffle")

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

        return gen()

    def slice_generator(self, slices, preload=5, en_numpy=False):
        def batches():
            for _slice in slices:
                batch = self[_slice].values
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")

    def batch_generator(self, batch_size, preload=5, offset=0, en_numpy=False, progress_bar=None):
        def _slices():
            for bs, be in batch_index_generator(offset, self.total, batch_size):
                yield slice(bs, be, 1)

        slices = list(_slices())

        if progress_bar is None:
            progress_bar = lambda i, **kwargs: i

        yield from progress_bar(self.slice_generator(slices, preload=preload, en_numpy=en_numpy), total=len(slices))

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

    def sps(self, batch_size=512):
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

    @en_lazy
    def list(self):
        return list(self)

    def batch_scan(self, init_state, scan_function, **batch_generator_kwargs):
        state = init_state
        for batch in self.batch_generator(**batch_generator_kwargs):
            state = scan_function(state, batch)
        return state

    def __repr__(self):
        def _str_(s: Series):
            return OrderedDict(
                series=str(s),
                metadata=s._metadata,
                parents=[_str_(p) for p in s.parents]
            )

        return pformat(_str_(self), indent=2)  # self.trace(0).trace_string()

    def acc(self, identifier, op):
        path = self.managed_cache_path(identifier)
        pickled = Pickled(path, lambda: op(self))
        return pickled

    def scan(self, init, scanner, batch_scanner=None, show_progress=True, batch_size=512):
        wrapper = tqdm if show_progress else (lambda i: i)
        generator = self if batch_scanner is None else self.batch_generator(batch_size=batch_size)
        scanner = scanner if batch_scanner is None else batch_scanner
        state = init
        for item in wrapper(generator):
            state = scanner(state, item)
        return state

    def normalization_scaler(self, feature_range=(-1, 1), batch_size=128, progress_bar=tqdm):
        # need to cache used scaler and store that in a history

        sample = self[0]
        assert isinstance(sample, np.ndarray), "normalize must be run on numpy array"
        ch = sample.shape[-1]
        scaler = MinMaxScaler(feature_range=feature_range)

        def _scanner(s, batch):
            s.partial_fit(batch.reshape(-1, ch))
            return s

        scaler = self.batch_scan(scaler, _scanner, batch_size=128, progress_bar=progress_bar, en_numpy=True)
        return scaler
        # return self.map(scaler.transform).update_metadata(scaler=scaler)

    def tagged_value(self, tag):
        return self.traces.map(lambda t: t.find_by_tag(tag)[0].get_value())

    def interact(self, depth=None,
                 skip_tags=("slow",),
                 show_tags=None,
                 skip_indices=None,
                 show_indices=None
                 ):
        def _l(i):
            self.trace(i).show_traceback_html(
                depth=depth,
                skipping_tags=skip_tags,
                showing_tags=show_tags,
                skip_indices=skip_indices,
                show_indices=show_indices
            )

        return interactive(_l, i=(0, len(self) - 1, 1))

    def widget(self):

        max_depth = len(self.trace(0).ancestors()) + 1

        def _interactive(**kwargs):
            def wrapper(f):
                return interactive(f, **kwargs)

            return wrapper

        @_interactive()
        def show_tree():
            print(repr(self))

        @_interactive(depth=widgets.IntSlider(value=1, min=1, max=max_depth))
        def show_trace(depth):
            display(self.interact(depth=depth))

        @_interactive()
        def show_network():
            display(series_to_tree(self))

        @_interactive()
        def show_trace2():
            display(self.traces[0])

        @_interactive()
        def control():
            display(self.control())

        @_interactive(
            visualization=dict(
                trace=show_trace,
                raw_trace=show_trace2,
                tree=show_tree,
                network=show_network,
                control=control
            )
        )
        def inner(visualization):
            display(visualization)

        return inner

    def _ipython_display_(self):
        return display(self.widget())

    def replace(self, replacer):
        replaced = replacer(self, self.parents)
        if replaced is not None:
            return replaced
        else:
            return self

    @abc.abstractmethod
    def clone(self, parents):
        pass

    def control(self):
        from data_tree.coconut.ops.control import control_series
        return control_series(self)

    def cmap(self, f, batch_f, defaults):
        """
        either f or batch_f can be None
        defaults: = dict(a=dict(schem=(1,128,0),value=128,description="test a ")))

        example:
        pixiv224.rescaled_images.cmap(
            f = (a,size,scale_method)->a |> Image.fromarray |> .resize((size,size),Image.LANCZOS),
            batch_f=None,
            defaults=dict(
                size=(128,dict(schem=(16,512,16),value=128,description="size")),
                scale_method=(Image.LANCZOS,dict(schem=[Image.LANCZOS,Image.BILINEAR],description="scaling method"))
            )
        )
        """
        from data_tree.coconut.ops.control import CMapTemplate
        return CMapTemplate(
            src=self,
            mapper=f,
            slice_mapper=batch_f,
            param_controllers=defaults
        )

    def values_progress(self, batch_size=16, progress_bar=tqdm):
        res = []
        for batch in self.batch_generator(batch_size=batch_size, preload=0, progress_bar=progress_bar):
            res += batch
        return res


class MultiSourcedSeries(Series):

    def __init__(self, parents: list):
        self._parents = parents

    @property
    def parents(self):
        return self._parents

    def replace(self, replacer):
        new_parents = [p.replace(replacer) for p in self.parents]
        replaced = replacer(self, new_parents)
        if replaced is not None:
            return replaced
        else:
            return self.clone(new_parents)


class ZippedSeries(MultiSourcedSeries):

    def clone(self, parents):
        return ZippedSeries(self, *parents)

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    @property
    def total(self):
        return self._total

    def _get_item(self, index):
        return tuple(s._values(index) for s in self.parents)

    def _get_slice(self, _slice):
        return list(zip(*(s._values(_slice) for s in self.parents)))

    def _get_indices(self, indices):
        return list(zip(*(s._values(indices) for s in self.parents)))

    def _get_mask(self, mask):
        return list(zip(*(s._values(mask) for s in self.parents)))

    def __init__(self, *sources: Iterable[Series]):
        super().__init__(parents=list(sources))
        self._total = min([s.total for s in self.parents])
        self._indexer = IdentityIndexer(self._total)

    def _trace(self, index) -> Trace:
        i = self.indexer[index]
        return Trace(
            parents=[s._trace(i) for s in self.parents],
            index=i,
            series=self,
            metadata=dict()
        )

    def slice_generator(self, slices, preload=5, en_numpy=False):
        def batches():
            src_slices = [self.indexer[_s] for _s in slices]
            for batch in zip(*[s.slice_generator(src_slices, preload=preload, en_numpy=False) for s in self.parents]):
                # batch is now batch of slices. so .. you need to extract it
                batch = tuple(zip(*batch))
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")


class LambdaAdapter(Series):
    @en_lazy
    def parents(self):
        return []

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
    @en_lazy
    def parents(self):
        return []

    def clone(self, parents):
        return Hdf5Adapter(self.hdf5_initializer,self.hdf5_file_name,self.key)

    def __init__(self, hdf5_initializer, hdf5_file_name, key):
        self.hdf5_initializer = hdf5_initializer
        self.hdf5_file_name = hdf5_file_name
        self.key = key
        if not os.path.exists(hdf5_file_name):
            hdf5_initializer()
        self.hdf5_file = h5py.File(hdf5_file_name, "r")
        if key not in self.hdf5_file:
            hdf5_initializer()
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

    def replace(self, replacer):
        new_parents = [p.replace(replacer) for p in self.parents]
        replaced = replacer(self, new_parents)
        if replaced is not None:
            return replaced
        else:
            return self.clone(new_parents)

    def clone(self, parents):
        return self.__class__(parents[0])

    @en_lazy
    def parents(self):
        return [self.src]

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

    def slice_generator(self, slices, preload=5, en_numpy=False):
        def batches():
            src_slices = [self.indexer[_s] for _s in slices]
            for batch in self.src.slice_generator(src_slices, preload=preload, en_numpy=False):
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")


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

    def slice_generator(self, slices, preload=5, en_numpy=False):
        def batches():
            for _slice in slices:
                batch = self[_slice].values
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")


class FlattenedSeries(MultiSourcedSeries):

    def clone(self, parents):
        return FlattenedSeries(Series.from_iterable(parents))

    @property
    def parents(self):
        return self._parents

    def __init__(self, src):
        super().__init__(list(src.values))
        self._src = src
        self._index = IdentityIndexer(total=sum([s.total for s in self.parents]))
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

    def clone(self, parents):
        return TaggedSeries(src=parents[0], *self.tags)

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

    def clone(self, parents):
        return MetadataOpsSeries(src=parents[0], operator=self.operator)

    def __init__(self, src: Series, operator: Callable):
        self.operator = operator
        self._src = src
        self._indexer = IdentityIndexer(self.src.total)

    @en_lazy
    def _metadata(self):
        return self.operator(dict())

    @property
    def src(self) -> Series:
        return self._src

    @property
    def indexer(self) -> Indexer:
        return self._indexer

    def __str__(self):
        return f"MetadataOpsSeries<{self.operator.__name__}> at {id(self)}"


class Indexed(IndexedSeries):

    def clone(self, parents):
        return Indexed(parents[0], self.indexer)

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

    def clone(self, parents):
        return MappedSeries(src=parents[0], single_mapper=self.single_mapper, slice_mapper=self.slice_mapper)

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
        def batches():
            src_slices = [self.indexer[_s] for _s in slices]
            for batch in self.src.slice_generator(src_slices, preload=preload, en_numpy=False):
                batch = self._map_values(batch)
                if en_numpy and not isinstance(batch, np.ndarray):
                    batch = np.array(batch)
                yield batch

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")


def mp_mapper(f, global_resources: Mapping[str, GlobalHolder], value):
    resources = {k: g.value for k, g in global_resources.items()}
    result = f(value, **resources)
    return result


def stp_worker_generator(mapper, global_resources):
    def _worker_generator():
        def _worker(value):
            v, token = value
            resources = {k: g.value for k, g in global_resources.items()}
            return mapper(v, **resources), token

        return _worker

    return _worker_generator


class MPMappedSeries(IndexedSeries):
    def clone(self, parents):
        return MPMappedSeries(src=parents[0], single_mapper=self.single_mapper, resources_kwargs=self.resources,
                              num_process=self.num_process)

    @property
    def indexer(self):
        return self._indexer

    @property
    def src(self) -> Series:
        return self._src

    def __init__(self, src: Series, single_mapper,
                 resources_kwargs: dict = Union[None, Mapping[str, Resource]],
                 num_process=None,
                 max_pending_result=8
                 ):
        self._src = src
        self._indexer = IdentityIndexer(self.total)
        self.single_mapper = single_mapper
        if resources_kwargs is None:
            resources_kwargs = dict()
        self.resources = resources_kwargs
        self.num_process = num_process
        self.max_pending_result = max_pending_result
        super().__init__()

    @property
    def total(self):
        return self._src.total

    def _map_values(self, values, show_progress=True):
        resources = {k: r.prepare() for k, r in self.resources.items()}
        if show_progress:
            bar = tqdm
        else:
            bar = lambda seq, *args: seq
        with multiprocessing.Pool(processes=self.num_process) as pool:
            futures = [pool.apply_async(mp_mapper, args=(self.single_mapper, resources, v)) for v in values]
            results = [f.get() for f in bar(futures, desc="waiting for mp map results")]
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

    def slice_generator(self, slices, preload=5, en_numpy=False):
        slices = list(slices)
        termination_signal = threading.Event()
        src_slices = [self.indexer[_s] for _s in slices]

        def batches():
            resources = {k: r.prepare() for k, r in self.resources.items()}
            stp = SequentialTaskParallel2(
                worker_generator=stp_worker_generator(self.single_mapper, global_resources=resources),
                num_worker=multiprocessing.cpu_count() if self.num_process is None else self.num_process,
                max_pending_result=self.max_pending_result)

            def _fetcher():
                logger.info(f"multiprocessing fetcher started")
                for batch in self.src.slice_generator(src_slices, preload=preload, en_numpy=False):
                    if not termination_signal.is_set():
                        # logger.debug(f"task queue size:{stp.task_queue.qsize()}")
                        for item in batch[:-1]:
                            stp.enqueue((item, None))
                        stp.enqueue((batch[-1], "end"))
                    else:
                        logger.warning(f"multiprocessing fetcher stopped due to signal")
                        break
                stp.enqueue_termination()
                logger.warning(f"multiprocessing fetcher stopped ")
            fetch_thread = threading.Thread(target=_fetcher)
            fetch_thread.start()
            try:
                with stp.managed_start() as gen:
                    batch_buffer = []
                    for item, token in tqdm(gen, desc="multi process mapping progress.."):
                        batch_buffer.append(item)
                        if token == "end":
                            if en_numpy and not isinstance(batch, np.ndarray):
                                batch = np.array(batch)
                            #logger.info("yielding batch")
                            yield batch_buffer
                            # logger.debug("yield batch")
                            batch_buffer = []
            finally:
                logger.debug(f"setting fetcher termination signal")
                termination_signal.set()
                for k, r in self.resources.items():
                    r.release(r)
                logger.debug(f"mp user resources are released")

        yield from prefetch_generator(batches(), preload, name=f"{self.__class__.__name__} #{id(self)}")


class NumpySeries(Series):
    def clone(self, parents):
        return NumpySeries(self.data)

    @property
    def parents(self):
        return []

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
    def parents(self):
        return []

    def clone(self, parents):
        return ListSeries(self.data)

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
