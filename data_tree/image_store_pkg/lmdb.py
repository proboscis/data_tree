import abc
import io
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import Pool
from queue import Queue
from threading import Thread
from typing import Iterable, Tuple, List, Callable, Iterator, Any

import PIL
import lmdb
import ray
from PIL import Image

from ray.actor import ActorHandle
from rx import Observable
from rx.core import typing
from rx.core.typing import T_in
from rx.subject import Subject
from tqdm import tqdm

from data_tree import series, logger

from data_tree.dt_wandb.json_artifact import JsonArtifactLoader
from data_tree.format_converter import FormatConverter
from data_tree.image_store import ImageStoreCreator, convert_images_to_bytes_mp, ImageStore, MillionImageFileCreator, \
    MergedImageStore
from data_tree.lmdb_bytes_kvs import LmdbBytesKvs
from data_tree.lmdb_util import LmdbCreator
from data_tree.storage_manager import FileStorageManager
from data_tree.util import ensure_path_exists, Pickled
from data_tree.wandb_util.artifact_identifier import ArtifactMetadata, ArtifactIdentifier
from data_tree.wandb_util.path_artifact import ManagedPathArtifactLoader, ManagedPathArtifactLogger
from pinjected.di.util import check_picklable


@dataclass
class LmdbImageStoreCreator(ImageStoreCreator):
    lmdb_creator: LmdbCreator
    lmdb_output_destination: str
    lmdb_image_format: str
    lmdb_size_in_bytes: int

    def create(self, key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]], total=None):
        def gen():
            for k, img in key_image_pairs:
                bytes = auto("image,RGB,RGB")(img).to(f"{self.lmdb_image_format}_bytes")
                yield k, bytes

        self.lmdb_creator.create(gen(), self.lmdb_output_destination, size_in_bytes=self.lmdb_size_in_bytes,
                                 total=total)


@dataclass
class MultiLmdbCreator:
    lmdb_creator: LmdbCreator

    def create(self, key_image_pairs: Iterable[Tuple[str, bytes]], dst_path, total=None, max_size_per_lmdb=2 ** 24,
               bar=None):
        # 16GB is the limitation on abci lustre fs
        i = 0

        def get_path():
            nonlocal i
            res = os.path.join(dst_path, f"{i}")
            i += 1
            return res

        bar = tqdm(desc=f"creating multi lmdb storage at {dst_path}", total=total) if bar is None else bar
        while key_image_pairs is not None:
            dst = get_path()
            ensure_path_exists(dst)
            key_image_pairs = self.lmdb_creator.create(
                key_image_pairs,
                dst=dst,
                size_in_bytes=max_size_per_lmdb,
                total=total,
                bar=bar
            )


@dataclass
class MessageGenerator:
    queue_size: int = field(default=None)

    def __post_init__(self):
        self.queue = Queue(self.queue_size)

    def add(self, item):
        self.queue.put(item)

    def finish(self):
        self.queue.put("__end__")

    def iterate_on_new_thread(self, f):
        assert not hasattr(self, "thread"), "consumer is already started for this queue"

        def yielder():
            while True:
                item = self.queue.get()
                if item == "__end__":
                    return
                yield item

        def impl():
            return f(yielder())

        self.thread = Thread(target=impl)
        self.thread.start()
        return self.thread


@ray.remote
class MultiLmdbCreatorActor:
    def __init__(self,
                 lmdb_creator: LmdbCreator,
                 dst_path,
                 total=None,
                 max_size_per_lmdb=2 ** 24,
                 ):
        self.lmdb_creator = lmdb_creator
        # I guess I should have used rx.Observable instead of generator.
        # because it is more versatile for multi processing
        # self.msg_generator = MessageGenerator()
        self.messages = Subject()

        # if Subject is thread safe, this should work fine.

        def on_gen(key_image_pairs):
            i = 0

            def get_path():
                nonlocal i
                res = os.path.join(dst_path, f"{i}")
                i += 1
                # hmm, so this is not supporting continuation.
                return res

            bar = tqdm(desc=f"creating multi lmdb storage at {dst_path}", total=total)
            while key_image_pairs is not None:
                dst = get_path()
                ensure_path_exists(dst)
                key_image_pairs = self.lmdb_creator.create(  # suppports continuation.
                    key_image_pairs,
                    dst=dst,
                    size_in_bytes=max_size_per_lmdb,
                    total=total,
                    bar=bar
                )

        self.thread = subscribe_as_generator(self.messages, on_gen, 10000)
        # self.msg_generator.iterate_on_new_thread(on_gen)

    def insert(self, key, value):
        self.messages.on_next((key, value))
        # self.msg_generator.add((key, value))

    def finish(self):
        self.messages.on_completed()
        # self.msg_generator.finish()
        self.thread.join()

    def on_next(self, item):
        self.insert(item)

    def on_completed(self):
        self.finish()

    def on_error(self, e):
        logger.error(e)
        raise e


class IImageStoreArtifactCreator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create(self, metadata: ArtifactMetadata, key_image_pairs, total=None):
        pass


class IMultiLmdbImageStoreCreator(abc.ABC):
    @abc.abstractmethod
    def create(self,
               metadata: ArtifactMetadata,
               key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]],
               bar=None,
               max_size_per_lmdb: int = 2 ** 34,
               total=None
               ):
        pass


def subscribe_as_generator(o: Observable, worker: Callable[[Iterator], Any], queue_size: int):
    msg_gen = MessageGenerator(queue_size)
    t = msg_gen.iterate_on_new_thread(worker)
    o.subscribe(
        on_next=msg_gen.add,
        on_completed=msg_gen.finish,
        on_error=logger.error
    )
    return t


@dataclass
class ActorObserver(typing.Observer):
    handle: ActorHandle

    def on_next(self, value: T_in) -> None:
        self.handle.on_next.remote(value)

    def on_error(self, error: Exception) -> None:
        self.handle.on_error.remote(error)

    def on_completed(self) -> None:
        self.handle.on_completed.remote()


@ray.remote
class SubjectActor:
    def __init__(self):
        self.queue = Queue()
        self.subscribers: List[ActorObserver] = []

    def subscribe(self, actor: ActorObserver):
        self.subscribers.append(actor)

    def on_next(self, value: T_in) -> None:
        for s in self.subscribers:
            s.on_next(value)

    def on_error(self, error: Exception) -> None:
        for s in self.subscribers:
            s.on_error(error)

    def on_completed(self) -> None:
        for s in self.subscribers:
            s.on_completed()


@dataclass
class RemoteSubject:
    handle: SubjectActor

    def subscribe(self, actor: ActorObserver):
        # can I bind a callback to an actor?
        # this is possible if self is an actor,
        # but can I do it from outside of an actor?
        # ray.remote returns a object ref.
        # so we need a way to create an object ref without value.

        self.subscribers.append(actor)

    def on_next(self, value: T_in) -> None:
        for s in self.subscribers:
            s.on_next(value)

    def on_error(self, error: Exception) -> None:
        for s in self.subscribers:
            s.on_error(error)

    def on_completed(self) -> None:
        for s in self.subscribers:
            s.on_completed()


@dataclass
class RemoteMultiLmdbImageStoreCreator(IMultiLmdbImageStoreCreator):
    lmdb_creator: LmdbCreator

    def create(self,
               key_image_pairs: Iterable[Tuple[str, bytes]],
               dst_path, total=None,
               max_size_per_lmdb=2 ** 24,
               bar=None):
        creator_actor = MultiLmdbCreatorActor.remote(
            self.lmdb_creator,
            dst_path=dst_path,
            total=total,
            max_size_per_lmdb=max_size_per_lmdb
        )
        # the problem is that we cannot pass an observable to an actor.
        # so we need to use an adapter
        # since we cannot pass an observable, why not pass an actor handle?
        # so first convert observable to an ActorHandle which acts like a observable.
        # then the target actor should receive this handle and use it as an observable
        for k, v in key_image_pairs:
            creator_actor.insert.remote(k, v)
        creator_actor.finish.remote()


@dataclass
class MultiLmdbImageStoreArtifactCreator(IMultiLmdbImageStoreCreator):
    multi_lmdb_creator: MultiLmdbCreator
    managed_path_artifact_logger: ManagedPathArtifactLogger
    format_converter: FormatConverter

    def create(self,
               metadata: ArtifactMetadata,
               key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]],
               bar=None,
               max_size_per_lmdb: int = 2 ** 34,
               total=None
               ):
        check_picklable(dict(
            format_converter=self.format_converter,
        ))  # pairs doesnt need to be picklable, right?
        return self.create_from_bytes(
            metadata=metadata,
            key_bytes_pairs=convert_images_to_bytes_mp(key_image_pairs, self.format_converter),
            bar=bar,
            max_size_per_lmdb=max_size_per_lmdb,
            total=total
        )

    def create_from_bytes(self,
                          metadata: ArtifactMetadata,
                          key_bytes_pairs: Iterable[Tuple[str, bytes]],
                          bar=None,
                          max_size_per_lmdb: int = 2 ** 34,
                          total=None
                          ):
        dst_path = self.managed_path_artifact_logger.log_artifact_path(metadata)
        self.multi_lmdb_creator.create(
            key_image_pairs=key_bytes_pairs,
            dst_path=dst_path,
            bar=bar,
            max_size_per_lmdb=max_size_per_lmdb,
            total=total
        )
        return dst_path


@dataclass
class MultiLmdbImageStoreArtifactCreatorTwo(IImageStoreArtifactCreator):
    multi_lmdb_image_store_artifact_creator: MultiLmdbImageStoreArtifactCreator

    def create(self, metadata: ArtifactMetadata, key_image_pairs, total=None):
        return self.multi_lmdb_image_store_artifact_creator.create(metadata, key_image_pairs=key_image_pairs,
                                                                   total=total)


@dataclass
class CreateLmdbImageStoreWithArtifactFromIterables:
    managed_path_artifact_logger: ManagedPathArtifactLogger
    lmdb_creator: LmdbCreator

    def create(self,
               key_imgs_pair: Iterable,
               output_artifact_name: str,
               size_in_bytes: int,
               format_converter: FormatConverter,
               total=None
               ):
        with Pool() as pool:
            converted = pool.imap_unordered(format_converter, key_imgs_pair)
            return self.create_from_bytes(
                converted, output_artifact_name, size_in_bytes, total=total
            )

    def create_from_bytes(self,
                          key_bytes_pair: Iterable,
                          output_artifact_name: str,
                          size_in_bytes: int,
                          total: int = None
                          ):
        dst_path = self.managed_path_artifact_logger.log_path(
            conditions=dict(name=output_artifact_name),
            name=output_artifact_name,
            type="lmdb_image_store",
        )

        self.lmdb_creator.create(key_bytes_pair, dst_path, size_in_bytes=size_in_bytes, total=total)
        return dst_path


@dataclass
class CreateLmdbImageStoreWithArtifact:
    managed_path_artifact_logger: ManagedPathArtifactLogger
    create_lmdb_image_store_with_artifact_from_iterables: CreateLmdbImageStoreWithArtifactFromIterables
    format_converter: FormatConverter

    def create(self,
               src: ImageStore,
               output_artifact_name: str,
               format_converter: FormatConverter,
               size_in_bytes: int
               ):
        keys = series(src.keys())
        images = keys.map(src.__getitem__)
        return self.create_lmdb_image_store_with_artifact_from_iterables.create(
            key_imgs_pair=keys.zip(images),
            output_artifact_name=output_artifact_name,
            format_converter=format_converter,
            size_in_bytes=size_in_bytes
        )


@dataclass
class LmdbImageStoreCreationWithArtifact(ImageStoreCreator):
    lmdb_output_artifact_name: str
    lmdb_size_in_bytes: int
    create_lmdb_image_store_with_artifact_from_iterables: CreateLmdbImageStoreWithArtifactFromIterables
    lmdb_format_converter: FormatConverter

    def create(self, key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]], total=None):
        return self.create_lmdb_image_store_with_artifact_from_iterables.create(
            key_imgs_pair=key_image_pairs,
            output_artifact_name=self.lmdb_output_artifact_name,
            size_in_bytes=self.lmdb_size_in_bytes,
            format_converter=self.lmdb_format_converter,
            total=total
        )


@dataclass
class ImageLMDBCreator(MillionImageFileCreator):
    lmdb_creator: LmdbCreator

    def create(self, paths, dst: str):
        def keys_and_data():
            for p in tqdm(paths, desc=f"storing images to lmdb file:{dst}"):
                with open(p, "rb") as f:
                    data = f.read()
                    yield os.path.basename(p), data

        self.lmdb_creator.create(keys_and_data(), dst)


@dataclass
class LmdbImageStore2(ImageStore):
    target_lmdb_path: str
    sort_image_store_keys: bool

    def __post_init__(self):
        self.src = LmdbBytesKvs(self.target_lmdb_path, self.sort_image_store_keys)

    def keys(self) -> List[str]:
        return self.src.keys()

    def __getitem__(self, item):
        return Image.open(io.BytesIO(self.src[item]))


@dataclass
class LmdbImageStore(ImageStore):
    target_lmdb_path: str
    sort_image_store_keys: bool

    def __post_init__(self):
        assert os.path.exists(self.target_lmdb_path), f"lmdb target path does not exist at {self.target_lmdb_path}!"
        self._keys = self._cached_keys()

    @property
    def txn(self):
        return self.create_txn()

    @contextmanager
    def create_env(self):
        # map_size was 1e11
        # on lustre file system on abci you cannot use file lock. and you have to disable it for lmdb to work.
        try:
            with lmdb.open(self.target_lmdb_path, lock=False, readonly=True) as env:
                yield env
        except Exception as e:
            import os
            logger.error(f"error on host:{socket.gethostname()}")
            # ah,oh, for some reason this error is happening on macbookpro.
            if not os.path.exists(self.target_lmdb_path):
                logger.error(f"paths in target_lmdb_path:{os.listdir(self.target_lmdb_path)}")
                logger.error(f"target_lmdb_path does not exist!:{e}")
            raise e

    @contextmanager
    def create_txn(self):
        with self.create_env() as env:
            with env.begin(write=False, buffers=True) as txn:
                yield txn

    def num_keys(self):
        with self.create_txn() as txn:
            return txn.stat()['entries']

    def _cached_keys(self) -> List[str]:
        cache_path = os.path.join(self.target_lmdb_path, 'keys.pkl')
        pkl_keys = Pickled(cache_path, self._keys)
        keys = pkl_keys.value
        if len(keys) != self.num_keys():
            pkl_keys.clear()
        return pkl_keys.value

    def keys(self) -> List[str]:
        return self._keys

    def _keys(self) -> List[str]:
        _keys = list(self.key_iterator())
        if self.sort_image_store_keys:
            logger.info(f"sorting image store keys...")
            _keys = list(sorted(_keys))
        return _keys

    def key_iterator(self):
        with self.create_txn() as txn:
            with txn.cursor() as cur:
                for t in tqdm(cur.iternext(values=False), desc="retrieving keys from lmdb"):
                    yield t.tobytes().decode()

    def __getitem__(self, item):
        with self.create_txn() as txn:
            buf = txn.get(item.encode())
            assert buf is not None  # why the heck this is None?
            return Image.open(io.BytesIO(buf.tobytes()))


@dataclass
class MultiLmdbImageStoreArtifactGetter:
    managed_path_artifact_loader: ManagedPathArtifactLoader
    sort_image_store_keys: bool

    def get(self, idt: ArtifactIdentifier):
        root = self.managed_path_artifact_loader.get_path_identifier(idt)
        lmdb_paths = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        assert lmdb_paths, f"lmdb_paths is empty! wtf?? {os.listdir(root)}"
        lmdbs = [LmdbImageStore(os.path.join(root, p), sort_image_store_keys=self.sort_image_store_keys) for p in
                 tqdm(lmdb_paths, desc='loading multi lmdb artifact')]
        return MergedImageStore(lmdbs, sort_image_store_keys=self.sort_image_store_keys)


@dataclass
class MultiLmdbImageStoreArtifact(ImageStore):
    identifier: ArtifactIdentifier
    multi_lmdb_image_store_artifact_getter: MultiLmdbImageStoreArtifactGetter

    def __post_init__(self):
        self._cache = None

    @property
    def cache(self):
        if self._cache is None:
            self._cache = self.multi_lmdb_image_store_artifact_getter.get(self.identifier)
        return self._cache

    def keys(self) -> List[str]:
        return self.cache.keys()

    def __getitem__(self, item):
        return self.cache.__getitem__(item)


@dataclass
class LmdbImageStoreArtifactFactory:
    """use this to get an ImageStore using artifact. so that we can track its usage."""
    json_artifact_loader: "JsonArtifactLoader"
    storage_manager: FileStorageManager
    sort_image_store_keys: bool

    def get_image_store(self, artifact_name):
        key = self.json_artifact_loader.load(artifact_name, "million_image_metadata")
        tgt_path = self.storage_manager.find(**key)
        return LmdbImageStore(tgt_path, sort_image_store_keys=self.sort_image_store_keys)


@dataclass
class ManagedLmdbImageStoreArtifact:
    managed_path_artifact_loader: ManagedPathArtifactLoader
    sort_image_store_keys: bool

    def get_image_store(self, identifier: str) -> ImageStore:
        path = self.managed_path_artifact_loader.get_path(identifier, "lmdb_image_store")
        return LmdbImageStore(path, sort_image_store_keys=self.sort_image_store_keys)


@dataclass
class LmdbImageStoreSplitter:
    multi_lmdb_image_store_artifact_creator: MultiLmdbImageStoreArtifactCreator

    def split_and_log(self, src: ImageStore, dst_metadata: ArtifactMetadata, max_bytes_per_lmdb=2 ** 24):
        return self.multi_lmdb_image_store_artifact_creator.create_from_bytes(
            metadata=dst_metadata,
            key_bytes_pairs=src.mp_bytes_pair_iterator(),
            total=len(src.keys()),
            max_size_per_lmdb=max_bytes_per_lmdb
        )
