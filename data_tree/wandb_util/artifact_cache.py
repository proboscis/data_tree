import abc
import hashlib
import pickle
from abc import ABCMeta
from dataclasses import dataclass
from glob import glob
from pprint import pformat
from typing import Callable, Generic, TypeVar, List

import jsonpickle
import torch
import cloudpickle
from cytoolz import valmap
from loguru import logger
from wandb.apis.public import ArtifactType

from data_tree.dt_wandb.json_artifact import JsonArtifactLogger, JsonArtifactLoader
from pinject_design.di.injected import Injected
from data_tree.wandb_util import random_artifact_file_path
from data_tree.wandb_util.artifact_identifier import ArtifactIdentifier, ArtifactMetadata
from data_tree.wandb_util.artifact_logger import ArtifactLogger
from data_tree.wandb_util.artifact_wrapper import RemoteArtifactFactory, PublicRemoteArtifactFactory, \
    RemoteArtifactMetadataFactory, RemoteArtifactMetadata
from data_tree.wandb_util.cache import IRead, ReadFacory, LambdaReadFacory

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


class ArtifactIo(ArtifactRead, ArtifactWrite, metaclass=ABCMeta): pass


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


class FileRead(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self, file: str) -> T:
        pass


class PickleRead(FileRead[T]):
    def read(self, file: str) -> T:
        with open(file, 'rb') as f:
            return pickle.load(f)

class CloudPickleRead(FileRead[T]):
    def read(self, file: str) -> T:
        with open(file, 'rb') as f:
            return cloudpickle.load(f)

class TorchRead(FileRead[T]):
    def read(self, file: str) -> T:
        with open(file, 'rb') as f:
            return torch.load(f)


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
    wandb:"wandb"

    def write(self, data: T, meta: ArtifactMetadata):
        path = random_artifact_file_path("pkl",self.wandb)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        self.artifact_logger.save2(path, meta)

@dataclass
class CloudPickleArtifactWrite(ArtifactWrite):
    artifact_logger: ArtifactLogger
    wandb:"wandb"

    def write(self, data: T, meta: ArtifactMetadata):
        path = random_artifact_file_path("pkl",self.wandb)
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)
        self.artifact_logger.save2(path, meta)

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
                logger.error(f"error while loading cache from artifact")
                logger.error(e)
                logger.error(f"failed to load cache: {self.identifier}")
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
                logger.error(f"error while loading cache from artifact")
                logger.error(e)
                logger.error(f"failed to load cache: {self.meta}")
                self._value = self.f()
                self.write.write(self._value, self.meta)
        return self._value

ARTIFACT_TYPE_MEMO = "memo"

@dataclass
class ArtifactMemo:
    read: ArtifactRead
    write: ArtifactWrite
    name: str
    f: Callable

    def __post_init__(self):
        self.cache = dict()

    def __call__(self, *args, **kwargs):
        dumped = jsonpickle.dumps((args, kwargs))
        input_hash = hashlib.sha256(dumped.encode()).hexdigest()
        meta = ArtifactMetadata(
            identifier=ArtifactIdentifier(
                name=self.name,
                type=ARTIFACT_TYPE_MEMO,
                version=input_hash[:10]
            ),
            metadata=dict(
                input=dumped
            )
        )
        if input_hash not in self.cache:
            try:
                logger.info(f"checking artifact memo for \n{pformat(dumped)}")
                hash, loaded = self.read.read(meta.identifier)
                if input_hash == hash:
                    self.cache[input_hash] = loaded
                else:
                    raise RuntimeError(
                        f"input hash is different from cached value.:before:{hash},now:{input_hash}")
            except Exception as e:
                logger.error(f"error while loading cache from artifact")
                logger.error(e)
                logger.error(f"failed to load cache: {meta}")
                value = self.f(*args, **kwargs)
                obj = input_hash, value
                self.write.write(obj, meta)
                self.cache[input_hash] = value
        return self.cache[input_hash]


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
            kwargs_types=valmap(lambda v: type(v),kwargs)
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
                logger.error(f"error while loading cache from artifact")
                logger.error(e)
                logger.error(f"failed to load cache: {meta}")

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
                logger.error(f"error while loading cache from artifact")
                logger.error(e)
                logger.error(f"failed to load cache: {self.meta}")
                self._value = self.f()
                self.json_artifact_write.write(self._value, self.meta)
        return self._value
class IArtifactCacheFactory(abc.ABC):
    @abc.abstractmethod
    def get(self,meta:ArtifactMetadata)->ReadFacory:
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

    def get(self, name, f: Callable) -> ArtifactMemo:
        return ArtifactMemo(
            name=name,
            read=self.read,
            write=self.write,
            f=f
        )


@dataclass
class PickleArtifactMemoFactory:
    pickle_artifact_read: PickleArtifactRead
    pickle_artifact_write: PickleArtifactWrite

    def __post_init__(self):
        self.factory = ArtifactMemoFactory(read=self.pickle_artifact_read, write=self.pickle_artifact_write)

    def get(self, name: str, f: Callable) -> ArtifactMemo:
        return self.factory.get(name=name, f=f)


@dataclass
class PickleArtifactCacheFactory(IArtifactCacheFactory):
    pickle_artifact_read: PickleArtifactRead
    pickle_artifact_write: PickleArtifactWrite

    def __post_init__(self):
        self.factory = ArtifactCacheFactory(
            read=self.pickle_artifact_read,
            write=self.pickle_artifact_write
        )

    def get(self, meta: ArtifactMetadata)->ReadFacory:
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

def read_pickle_artifact(idt:ArtifactIdentifier):
    return PickleArtifactRead.map(lambda reader:reader.read(idt))
