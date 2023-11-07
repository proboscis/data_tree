import abc
import asyncio
import hashlib
import os
import pickle
from concurrent.futures import Future
from dataclasses import dataclass
from glob import glob
from threading import Thread
from typing import Callable, Generic, TypeVar, List, Awaitable, Any

import cloudpickle
import jsonpickle
from cytoolz import valmap
from returns.primitives.exceptions import UnwrapFailedError
from wandb import CommError
from wandb.apis.public import ArtifactType

from data_tree import logger
from data_tree.disk_cache import StaticDiskCache
from data_tree.dt_wandb.json_artifact import JsonArtifactLogger, JsonArtifactLoader
from data_tree.key_value_cache import IKeyValueCache
from data_tree.key_value_store import KeyValueStore
from data_tree.wandb_util import random_artifact_file_path
from data_tree.wandb_util.artifact_cache_type_memo import ARTIFACT_TYPE_MEMO
from data_tree.wandb_util.artifact_identifier import ArtifactIdentifier, ArtifactMetadata
from data_tree.wandb_util.artifact_logger import ArtifactLogger
from data_tree.wandb_util.artifact_wrapper import RemoteArtifactFactory, PublicRemoteArtifactFactory, \
    RemoteArtifactMetadataFactory, RemoteArtifactMetadata
from data_tree.wandb_util.cache import IRead, ReadFacory, LambdaReadFacory
from data_tree.wandb_util.file_read_write import FileRead, PickleRead, CloudPickleRead, TorchRead, Hdf5DataFrameRead, \
    FileWrite, Hdf5DataFrameWrite
from pinjected.di.injected import Injected
from pinjected import injected_function

T = TypeVar("T")
U = TypeVar("U")


class ArtifactWrite(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write(self, data: T, meta: ArtifactMetadata):
        pass


class ArtifactRead(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self, idt: ArtifactIdentifier) -> T:
        pass


class AsyncArtifactRead(Generic[T], abc.ABC):
    @abc.abstractmethod
    async def aread(self, idt: ArtifactIdentifier) -> T:
        pass


class AsyncArtifactWrite(Generic[T], abc.ABC):
    @abc.abstractmethod
    async def awrite(self, data: T, meta: ArtifactMetadata):
        pass


class AsyncArtifactDelete(abc.ABC):
    @abc.abstractmethod
    async def adelete(self, idt: ArtifactIdentifier):
        pass


@dataclass
class ArtifactIo(ArtifactRead, ArtifactWrite):
    reader: ArtifactRead
    writer: ArtifactWrite

    def read(self, idt: ArtifactIdentifier) -> T:
        return self.reader.read(idt)

    def write(self, data: T, meta: ArtifactMetadata):
        return self.writer.write(data, meta)


@dataclass
class JsonArtifactRead(ArtifactRead):
    json_artifact_loader: JsonArtifactLoader

    def read(self, idt: ArtifactIdentifier) -> T:
        return self.json_artifact_loader.load_identifier(idt)


@dataclass
class JsonArtifactWrite(ArtifactWrite):
    json_artifact_logger: JsonArtifactLogger

    def write(self, data: T, meta: ArtifactMetadata):
        return self.json_artifact_logger.log2(data, meta)


@dataclass
class GenericArtifactWrite(ArtifactWrite):
    wandb: "wandb"
    artifact_logger: ArtifactLogger
    file_write: FileWrite
    file_extension: str

    def write(self, data: T, meta: ArtifactMetadata):
        path = random_artifact_file_path(self.file_extension, self.wandb)
        self.file_write.write(data, path)
        self.artifact_logger.save2(path, meta)


@dataclass
class ExtensionArtifactRead(ArtifactRead):
    remote_artifact_factory: RemoteArtifactFactory
    extension: str
    file_read: FileRead

    def read(self, idt: ArtifactIdentifier) -> T:
        rart = self.remote_artifact_factory.from_identifier(idt.identifier_str(), type=idt.type)
        path = rart.download()
        tgt = glob(path + f"/*.{self.extension}")[0]
        return self.file_read.read(tgt)


async def proc_as_async_in_thread(proc: Callable):
    """Threows exception if the proc throws exception ineternally"""
    fut = Future()

    def impl():
        try:
            res = proc()
        except Exception as e:
            fut.set_exception(e)
            # raising this will print the stacktrace somehow.
            # raise e
            return
        fut.set_result(res)

    t = Thread(target=impl, daemon=True)
    t.start()
    res = await asyncio.wrap_future(fut)
    return res


@dataclass
class ThreadedAsyncArtifactRead(AsyncArtifactRead):
    artifact_read: ArtifactRead

    async def aread(self, idt: ArtifactIdentifier) -> T:
        return await proc_as_async_in_thread(lambda: self.artifact_read.read(idt))


@dataclass
class ThreadedAsyncArtifactWrite(AsyncArtifactWrite):
    artifact_write: ArtifactWrite

    async def awrite(self, data: T, meta: ArtifactMetadata):
        return await proc_as_async_in_thread(lambda: self.artifact_write.write(data, meta))


@dataclass
class ExtensionArtifactReadFactory:
    remote_artifact_factory: RemoteArtifactFactory

    def get(self, extension: str, file_read: FileRead):
        return ExtensionArtifactRead(
            remote_artifact_factory=self.remote_artifact_factory,
            extension=extension,
            file_read=file_read
        )


PickleArtifactRead: Injected[ArtifactRead] = Injected.bind(ExtensionArtifactRead,
                                                           extension=Injected.pure("pkl"),
                                                           file_read=PickleRead
                                                           )
CloudPickleArtifactRead: Injected[ArtifactRead] = Injected.bind(ExtensionArtifactRead,
                                                                extension=Injected.pure("pkl"),
                                                                file_read=CloudPickleRead
                                                                )
TorchArtifactRead: Injected[ArtifactRead] = Injected.bind(ExtensionArtifactRead,
                                                          extension=Injected.pure("pth"),
                                                          file_read=TorchRead
                                                          )
Hdf5DataFrameArtifactRead: Injected[ArtifactRead] = Injected.bind(ExtensionArtifactRead,
                                                                  extension=Injected.pure("h5f"),
                                                                  file_read=Hdf5DataFrameRead
                                                                  )
Hdf5DataFrameArtifactWrite: Injected[ArtifactWrite] = Injected.bind(GenericArtifactWrite,
                                                                    file_extension=Injected.pure("h5f"),
                                                                    file_write=Hdf5DataFrameWrite,
                                                                    )


@dataclass
class PublicPickleArtifactRead(ArtifactRead):
    public_remote_artifact_getter: PublicRemoteArtifactFactory

    def read(self, idt: ArtifactIdentifier) -> T:
        rart = self.public_remote_artifact_getter.get(idt)
        path = rart.download()
        tgt = glob(path + "/*.pkl")[0]
        with open(tgt, 'rb') as f:
            return pickle.load(f)


@dataclass
class PickleArtifactWrite(ArtifactWrite):
    artifact_logger: ArtifactLogger
    wandb: "wandb"

    def write(self, data: T, meta: ArtifactMetadata):
        path = random_artifact_file_path("pkl", self.wandb)
        logger.info(f"logging pickle artifact at {path}")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"written file at {path}. size == {os.path.getsize(path)}")
        self.artifact_logger.save2(path, meta)


@dataclass
class CloudPickleArtifactWrite(ArtifactWrite):
    artifact_logger: ArtifactLogger
    wandb: "wandb"

    def write(self, data: T, meta: ArtifactMetadata):
        path = random_artifact_file_path("pkl", self.wandb)
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)
        self.artifact_logger.save2(path, meta)


def investigate_cache_error(e, identifier, caller_info: str):
    logger.error(f"investigate_cache_error from {caller_info}")
    logger.error(f"error while loading cache from artifact")
    logger.error(e)
    logger.error(f"error type:{type(e)}")
    logger.error(f"failed to load cache: {identifier}")
    if isinstance(e, UnwrapFailedError):
        logger.error(f"causing container:{e.halted_container}")


@dataclass
class JsonArtifactCache(IRead):
    json_artifact_logger: JsonArtifactLogger
    json_artifact_loader: JsonArtifactLoader
    name: str
    type: str
    alias: str
    f: Callable

    def __post_init__(self):
        self.identifier = self.name + ":" + self.alias
        self._value = None

    @property
    def value(self):
        if self._value is None:
            try:
                loaded = self.json_artifact_loader.load(self.identifier, self.type)
                self._value = loaded
            except Exception as e:
                investigate_cache_error(e, self.identifier, caller_info="JsonArtifactCache")
                self._value = self.f()
                self.json_artifact_logger.log(self.name, self.type, self._value, metadata=dict(kind="cache"),
                                              aliases=[self.alias])
        return self._value


@dataclass
class ArtifactCache:
    read: ArtifactRead
    write: ArtifactWrite
    meta: ArtifactMetadata
    f: Callable

    def __post_init__(self):
        self._value = None

    @property
    def value(self):
        if self._value is None:
            try:
                loaded = self.read.read(self.meta.identifier)
                self._value = loaded
            except Exception as e:
                investigate_cache_error(e, self.meta.identifier, caller_info="ArtifactCache")
                self._value = self.f()
                self.write.write(self._value, self.meta)
        return self._value


@dataclass
class ArtifactMemoHashProvider:
    remote_artifact_metadata_factory: RemoteArtifactMetadataFactory

    def get_hash_meta(self, name, args, kwargs):
        dumped = jsonpickle.dumps((args, kwargs))
        input_hash = hashlib.sha256(dumped.encode()).hexdigest()
        meta = ArtifactMetadata(
            identifier=ArtifactIdentifier(
                name=name,
                type=ARTIFACT_TYPE_MEMO,
                version=input_hash
            ),
            metadata=dict(
                input=dumped[:10000]
            )
        )
        return input_hash, meta

    def exists(self, name, args, kwargs):
        ihash, meta = self.get_hash_meta(name, args, kwargs)
        rmeta = self.remote_artifact_metadata_factory.get2(meta.identifier)
        try:
            size = rmeta.size
            return True
        except Exception as e:
            logger.error(f"failed to get size for {meta}")
        return False


@dataclass
class MemoCallResult:
    value: Any
    hit: bool


@dataclass
class ArtifactMemo:
    read: ArtifactRead
    write: ArtifactWrite
    artifact_memo_hash_provider: ArtifactMemoHashProvider
    name: str
    f: Callable

    def __post_init__(self):
        self.cache = dict()

    def exists(self, *args, **kwargs):
        return self.artifact_memo_hash_provider.exists(self.name, args, kwargs)

    def __call__(self, *args, **kwargs):
        return self.call_with_hit(*args, **kwargs).value

    def call_with_hit(self, *args, **kwargs):
        input_hash, meta = self.artifact_memo_hash_provider.get_hash_meta(self.name, args, kwargs)
        try:
            # logger.info(f"checking artifact memo for \n{pformat(dumped)}")
            hash, loaded = self.read.read(meta.identifier)
            if input_hash == hash:
                self.cache[input_hash] = loaded
                return MemoCallResult(value=loaded, hit=True)
            else:
                raise RuntimeError(
                    f"input hash is different from cached value.:before:{hash},now:{input_hash}")
        except Exception as e:
            investigate_cache_error(e, meta.identifier, caller_info="ArtifactMemo")
            value = self.f(*args, **kwargs)
            obj = input_hash, value
            self.write.write(obj, meta)
            self.cache[input_hash] = value
        return MemoCallResult(self.cache[input_hash], hit=False)


@dataclass
class AsyncArtifactMemo:
    read: AsyncArtifactRead
    write: AsyncArtifactWrite
    name: str
    f: Callable[[Any], Awaitable]
    artifact_memo_hash_provider: ArtifactMemoHashProvider

    def __post_init__(self):
        self.cache = dict()

    def exists(self, *args, **kwargs):
        return self.artifact_memo_hash_provider.exists(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return (await self.call_with_hit(*args, **kwargs)).value

    def artifact_metadata(self, *args, **kwargs):
        input_hash, meta = self.artifact_memo_hash_provider.get_hash_meta(self.name, args, kwargs)
        return meta

    async def update(self, *args, **kwargs):
        input_hash, meta = self.artifact_memo_hash_provider.get_hash_meta(self.name, args, kwargs)
        value = await self.f(*args, **kwargs)
        obj = input_hash, value
        await self.write.awrite(obj, meta)
        self.cache[input_hash] = value
        return value

    async def call_with_hit(self, *args, **kwargs):
        input_hash, meta = self.artifact_memo_hash_provider.get_hash_meta(self.name, args, kwargs)
        try:
            # logger.info(f"checking artifact memo for \n{pformat(dumped)}")
            hash, loaded = await self.read.aread(meta.identifier)
            if input_hash == hash:
                self.cache[input_hash] = loaded
                return MemoCallResult(value=loaded, hit=True)
            else:
                raise RuntimeError(
                    f"input hash is different from cached value.:before:{hash},now:{input_hash}")
        except (CommError, ValueError):
            logger.debug(f"failed to load cache for {meta.identifier} (CommError/ValueError == cache miss)")
            # I thought the chain ends here,,, yes it does.?
            # then why is this exception being printed???
        except Exception as e:
            investigate_cache_error(e, meta.identifier, caller_info="AsyncArtifactMemo")
        # we are here since we failed to load the cache
        await self.update(*args, **kwargs)
        return MemoCallResult(self.cache[input_hash], hit=False)


def get_memo_meta_hash(name, args, kwargs):
    dumped = jsonpickle.dumps((name, args, kwargs))
    input_hash = hashlib.sha256(dumped.encode()).hexdigest()
    meta = ArtifactMetadata(
        identifier=ArtifactIdentifier(
            name=name,
            type=ARTIFACT_TYPE_MEMO,
            version=input_hash[:10]
        ),
        metadata=dict(
            comment="args and kwargs are not saved due to large size",
            args_types=[type(i) for i in args],
            kwargs_types=valmap(lambda v: type(v), kwargs)
        )
    )
    return meta, input_hash


@dataclass
class WandbArtifactApi:
    wandb: "wandb"
    project: str

    def artifact_type(self, type_name: str):
        api = self.wandb.Api()
        return api.artifact_type(type_name=type_name, project=self.project)


@dataclass
class PickleArtifactMemoCheck:
    pickle_artifact_read: PickleArtifactRead
    wandb_artifact_api: WandbArtifactApi
    remote_artifact_metadata_factory: RemoteArtifactMetadataFactory

    def names(self):
        return [i.name for i in list(self.wandb_artifact_api.artifact_type("memo").collections())]

    def memos(self, name) -> List[RemoteArtifactMetadataFactory]:
        at: ArtifactType = self.wandb_artifact_api.artifact_type("memo")
        col = at.collection(name)
        to_meta = self.remote_artifact_metadata_factory.get
        metas = [RemoteReadableArtifact(
            to_meta(a), self.pickle_artifact_read
        ) for a in col.versions()]
        return metas


@dataclass
class RemoteReadableArtifact:
    meta: RemoteArtifactMetadata
    read: ArtifactRead

    def load(self):
        return self.read.read(self.meta.to_artifact_identifier())


# memo assumes that we return the same thing on any situation
# but what we want is a memo that returns an injected that returns same thing.

@dataclass
class ArtifactInjectedMemo:
    """
    given T->U,
    this becomes T->Injected[U] which is an Injected that returns U while memoizing T. It may better be expressed as
    T=>Injected[Memoized[U]] anyways, it actually is T=>Injected[U]
    """
    read: ArtifactRead
    write: ArtifactWrite
    name: str
    f: Injected[Callable]

    def __post_init__(self):
        self.cache = dict()

    def __call__(self, *args, **kwargs) -> Injected:
        # some how None is passed here.
        meta, input_hash = get_memo_meta_hash(self.name, args, kwargs)
        if input_hash not in self.cache:
            try:
                hash, loaded = self.read.read(meta.identifier)
                if input_hash == hash:
                    logger.info(f"memo hit:{loaded}")
                    self.cache[input_hash] = loaded
                else:
                    raise RuntimeError(
                        f"input hash is different from cached value.:before:{hash},now:{input_hash}")
            except Exception as e:
                investigate_cache_error(e, meta.identifier, caller_info="ArtifactInjectedMemo")

                def _assign_cache(task_impl):
                    # res is an instance of task so it's not cachable
                    # what we do is wrap this res again
                    value = task_impl(*args, **kwargs)
                    obj = input_hash, value
                    self.write.write(obj, meta)
                    self.cache[input_hash] = value
                    return value

                return self.f.map(_assign_cache)
        to_be_injected = lambda: self.cache[input_hash]
        return Injected.bind(to_be_injected)


@dataclass
class PickledArtifactInjectedMemoFactory:
    pickle_artifact_read: ArtifactRead
    pickle_artifact_write: ArtifactWrite

    def get(self, name, f: Injected[Callable]):
        return ArtifactInjectedMemo(
            read=self.pickle_artifact_read,
            write=self.pickle_artifact_write,
            name=name,
            f=f
        )


@dataclass
class JsonArtifactCache2(IRead):
    json_artifact_write: JsonArtifactWrite
    json_artifact_read: JsonArtifactRead
    meta: ArtifactMetadata
    f: Callable

    def __post_init__(self):
        self._value = None

    @property
    def value(self):
        if self._value is None:
            try:
                loaded = self.json_artifact_read.read(self.meta.identifier)
                self._value = loaded
            except Exception as e:
                investigate_cache_error(e, self.meta.identifier, caller_info="JsonArtifactCache2")
                self._value = self.f()
                self.json_artifact_write.write(self._value, self.meta)
        return self._value


# data_tree/data_tree/wandb_util/artifact_cache.py
class IArtifactCacheFactory(abc.ABC):
    @abc.abstractmethod
    def get(self, meta: ArtifactMetadata) -> ReadFacory:
        pass


@dataclass
class ArtifactCacheFactory(IArtifactCacheFactory):
    read: ArtifactRead
    write: ArtifactWrite

    def get(self, meta: ArtifactMetadata) -> ReadFacory:
        def factory(f):
            return ArtifactCache(
                read=self.read,
                write=self.write,
                meta=meta,
                f=f
            )

        return LambdaReadFacory(factory)


@dataclass
class ArtifactMemoFactory:
    read: ArtifactRead
    write: ArtifactWrite
    artifact_memo_hash_provider: ArtifactMemoHashProvider

    def get(self, name, f: Callable) -> ArtifactMemo:
        return ArtifactMemo(
            name=name,
            read=self.read,
            write=self.write,
            artifact_memo_hash_provider=self.artifact_memo_hash_provider,
            f=f
        )


@dataclass
class AsyncMemoFactory:
    read: ArtifactRead
    write: ArtifactWrite

    def get(self, name, f: Callable) -> ArtifactMemo:
        return ArtifactMemo(
            name=name,
            read=self.read,
            write=self.write,
            f=f
        )


@dataclass
class PickleArtifactMemoFactory:
    cloud_pickle_artifact_read: ArtifactRead
    cloud_pickle_artifact_write: ArtifactWrite
    artifact_memo_hash_provider: ArtifactMemoHashProvider

    def __post_init__(self):
        self.factory = ArtifactMemoFactory(read=self.cloud_pickle_artifact_read, write=self.cloud_pickle_artifact_write,
                                           artifact_memo_hash_provider=self.artifact_memo_hash_provider)

    def get(self, name: str, f: Callable) -> ArtifactMemo:
        return self.factory.get(name=name, f=f)


@injected_function
def cloud_pickle_async_artifact_memo(
        _cloud_pickle_artifact_read,
        _cloud_pickle_artifact_write,
        _artifact_memo_hash_provider,
        name,
):
    def impl(func: Callable[[Any], Awaitable]):
        return AsyncArtifactMemo(
            read=ThreadedAsyncArtifactRead(_cloud_pickle_artifact_read),
            write=ThreadedAsyncArtifactWrite(_cloud_pickle_artifact_write),
            artifact_memo_hash_provider=_artifact_memo_hash_provider,
            name=name,
            f=func
        )

    return impl


@dataclass
class PickleArtifactAsyncMemoFactory:
    cloud_pickle_artifact_read: ArtifactRead
    cloud_pickle_artifact_write: ArtifactWrite

    def __post_init__(self):
        pass

    def get(self, name, f: Callable):
        pass


@dataclass
class PickleArtifactCacheFactory(IArtifactCacheFactory):
    cloud_pickle_artifact_read: ArtifactRead
    cloud_pickle_artifact_write: ArtifactWrite

    def __post_init__(self):
        self.factory = ArtifactCacheFactory(
            read=self.cloud_pickle_artifact_read,
            write=self.cloud_pickle_artifact_write
        )

    def get(self, meta: ArtifactMetadata) -> ReadFacory:
        return self.factory.get(meta)


@dataclass
class JsonArtifactCacheFactory(IArtifactCacheFactory):
    json_artifact_logger: JsonArtifactLogger
    json_artifact_loader: JsonArtifactLoader

    def get(self, name: str, alias: str, type: str) -> ReadFacory:
        def factory(f):
            return JsonArtifactCache(
                self.json_artifact_logger,
                self.json_artifact_loader,
                name,
                type,
                alias,
                f
            )

        return LambdaReadFacory(factory)


@dataclass
class ReadableRemoteArtifact:
    identifier: ArtifactIdentifier
    read: ArtifactRead

    def load(self):
        self.read.read(self.identifier)


def read_pickle_artifact(idt: ArtifactIdentifier):
    return PickleArtifactRead.map(lambda reader: reader.read(idt))


@dataclass
class DiskCacheArtifactRead(ArtifactRead):
    remote_artifact_factory: RemoteArtifactFactory

    def read(self, idt: ArtifactIdentifier) -> StaticDiskCache:
        ra = self.remote_artifact_factory.from_identifier_object(idt)
        downloaded = ra.download()
        return StaticDiskCache(downloaded)


@dataclass
class KeyValueCacheKvsAdapter(KeyValueStore):
    src: IKeyValueCache

    def keys(self) -> List[str]:
        return self.src.keys()

    def __getitem__(self, item) -> T:
        return self.src[item]


@dataclass
class LmdbArtifactRead(ArtifactRead):
    path_to_lmdb_kvs: Callable[[str], KeyValueStore]
    remote_artifact_factory: RemoteArtifactFactory

    def read(self, idt: ArtifactIdentifier) -> T:
        path = self.remote_artifact_factory.from_identifier_object(idt).download()
        return self.path_to_lmdb_kvs(path)
