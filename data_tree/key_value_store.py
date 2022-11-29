import abc
import os
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Generic, List, Callable, Tuple, Iterable, Any, TypeVar

import h5py
import numpy as np
import ray
from ray.actor import ActorHandle
from tqdm import tqdm

from data_tree import series, logger
from data_tree._series import Series
from data_tree.periodic import PeriodicReopenV2
from data_tree.wandb_util.artifact_identifier import ArtifactMetadata, ArtifactIdentifier
from data_tree.wandb_util.path_artifact import ManagedPathArtifactLoader, ManagedPathArtifactLogger

T = TypeVar("T")
U = TypeVar("U")


class KeyValueStore(Generic[T], metaclass=abc.ABCMeta):
    """
    read only interface of a kvs
    """

    @abc.abstractmethod
    def keys(self) -> List[str]:
        pass

    @abc.abstractmethod
    def __getitem__(self, item) -> T:
        pass

    def map(self, f: Callable[[T], U]) -> "MappedKVS[U]":
        return MappedKVS(self, f)

    def __call__(self, item):
        return self[item]

    def sorted_items(self) -> Series[Tuple[str, T]]:
        """
        sorted series of tuple[key,img]
        :return:
        """
        return series(self.keys()).sorted().map(lambda k: (k, self[k]))

    def sorted_values(self) -> Series[T]:
        return series(self.keys()).sorted().map(self)

    def __len__(self):
        return len(self.keys())

    def subset(self, keys):
        return JustKVS(lambda: keys, lambda k: self[k])

    def blacklist_keys(self,keys):
        org_keys = set(self.keys())
        result_keys = org_keys - set(keys)
        return self.subset(result_keys)

    def take(self, n):
        return self.subset(self.keys()[:n])


@dataclass
class JustKVS(KeyValueStore):
    _keys: Callable
    _getitem: Callable

    def keys(self) -> List[str]:
        return self._keys()

    def __getitem__(self, item):
        return self._getitem(item)

    def __call__(self, key):
        return self[key]


@dataclass
class RamKvs(KeyValueStore):
    src: dict

    def keys(self) -> List[str]:
        return list(self.src.keys())

    def __getitem__(self, item) -> T:
        return self.src[item]


@dataclass
class MappedKVS(KeyValueStore):
    src: KeyValueStore[T]
    mapping: Callable

    def keys(self) -> List[str]:
        return self.src.keys()

    def __getitem__(self, item):
        return self.mapping(self.src[item])


class IKeyValueStoreCreator(abc.ABC):
    @abc.abstractmethod
    def create(self, key_data_pairs: Iterable[Tuple[str, Any]], dest: str):
        pass


class IKVSArtifactCreator(abc.ABC):
    @abc.abstractmethod
    def create(self, key_data_pairs: Iterable[Tuple[str, Any]], dest: ArtifactMetadata):
        pass


@dataclass
class Hdf5KeyValueStoreCreator(IKeyValueStoreCreator):
    def create(self, key_data_pairs: Iterable[Tuple[str, np.ndarray]], dest: str):
        logger.info(f"creating hdf5 kvs at {dest}")
        if os.path.exists(dest):
            logger.error(f"the target file already exists at {dest}")
            raise RuntimeError(f"the target hdf5 file already exists {dest}")

        with PeriodicReopenV2(dest) as pr:
            keys = []
            for i, (k, data) in enumerate(key_data_pairs):
                f = pr.update()
                if "key_value_store" not in f:
                    f.create_dataset("key_value_store", shape=(0, *data.shape), dtype=data.dtype,
                                     maxshape=(None, *data.shape), chunks=(1, *data.shape))
                ds = f["key_value_store"]
                keys.append(k)
                if i >= ds.shape[0]:
                    ds.resize((ds.shape[0] + 1, *ds.shape[1:]))
                ds[i] = data
            data = np.array(keys).astype("S")
            f.create_dataset("keys", data=data)
        logger.info(f"done hdf5 kvs at {dest}")


@dataclass
class Hdf5KeyValueStore(KeyValueStore):
    hdf5_path: str

    def __post_init__(self):
        with h5py.File(self.hdf5_path, "r") as f:
            self._keys = list(np.array(f["keys"]))
            self.key_to_index = {k: i for i, k in enumerate(self._keys)}

    def keys(self) -> List[str]:
        return self._keys

    def __getitem__(self, item) -> T:
        with h5py.File(self.hdf5_path, "r") as f:
            return f["key_value_store"][self.key_to_index[item]]


# now we can create hdf5 with artifact
@dataclass
class Hdf5ArtifactKeyValueStoreCreator(IKVSArtifactCreator):
    managed_path_artifact_logger: ManagedPathArtifactLogger
    hdf5_key_value_store_creator: Hdf5KeyValueStoreCreator

    def create(self, key_data_pairs: Iterable[Tuple[str, Any]], dest: ArtifactMetadata):
        path = self.managed_path_artifact_logger.log_artifact_path(dest)
        self.hdf5_key_value_store_creator.create(key_data_pairs, path)


@dataclass
class ManagedHdf5KvsFactory:
    managed_path_artifact_loader: ManagedPathArtifactLoader

    def get(self, idt: ArtifactIdentifier) -> Hdf5KeyValueStore:
        path = self.managed_path_artifact_loader.get_path_identifier(idt)
        return Hdf5KeyValueStore(path)


@dataclass
class LocalHdf5ArtifactKvsCreator:
    creator: IKVSArtifactCreator
    destination: ArtifactMetadata

    def __post_init__(self):
        self.finished = False
        self.queue = Queue()

    def start(self):
        def impl():
            def yielder():
                while not self.finished:
                    item = self.queue.get()
                    if item is None:
                        self.finished = True
                        return
                    yield from item

            self.creator.create(yielder(), self.destination)

        self.thread = Thread(target=impl)
        self.thread.start()

    def insert_batch(self, batch):
        self.queue.put(batch)

    def signal_end(self):
        self.queue.put(None)
        self.thread.join()

    def __repr__(self):
        return f"{self.__class__.__name__}"


@ray.remote
@dataclass
class RemoteHdf5ArtifactKvsCreator:
    creator: IKVSArtifactCreator
    destination: ArtifactMetadata

    def __post_init__(self):
        self.queue = Queue()
        self.finished = False

    def start(self):
        def impl():
            def yielder():
                while not self.finished:
                    item = self.queue.get()
                    if item is None:
                        self.finished = True
                        return
                    yield from item

            self.creator.create(yielder(), self.destination)

        self.thread = Thread(target=impl)
        self.thread.start()

    def insert_batch(self, batch):
        self.queue.put(batch)

    def signal_end(self):
        self.queue.put(None)
        self.thread.join()

    def __repr__(self):
        return f"{self.__class__.__name__}"


@dataclass
class RemoteKeyValueStoreWrapper(KeyValueStore):
    src: ActorHandle  # handle of KeyValueStoreActor

    def keys(self) -> List[str]:
        return ray.get(self.src.keys.remote())

    def __getitem__(self, item) -> T:
        return ray.get(self.src.__getitem__.remote(item))


@dataclass
class MergedKvs(KeyValueStore):
    sources: List[KeyValueStore]

    def __post_init__(self):
        # build key to src mapping
        self.key_to_src = dict()
        for i, s in tqdm(enumerate(self.sources)):
            for k in tqdm(s.keys()):
                self.key_to_src[k] = i

    def keys(self) -> List[str]:
        return list(self.key_to_src.keys())

    def __getitem__(self, item) -> T:
        return self.sources[self.key_to_src[item]][item]


