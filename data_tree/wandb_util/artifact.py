import abc
import os
import uuid
from abc import abstractmethod
from dataclasses import dataclass

import wandb

from data_tree import logger
from data_tree.wandb_util.artifact_finder import WandbArtifactFinder
from data_tree.wandb_util.artifact_logger import ArtifactLogger
from data_tree.util import Pickled
from data_tree.wandb_util.artifact_wrapper import RemoteArtifactFactory, RemoteArtifact


class ArtifactSavable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def save(self, obj, dst: str) -> str:
        pass

    @abc.abstractmethod
    def load(self, src: str) -> object:
        pass


@dataclass
class CachedArtifact:
    wandb: wandb
    validate_artifact: bool
    cache_dir: str
    artifact_dir: str
    name: str
    type: str

    def download(self) -> str:
        art = self.wandb.run.use_artifact(self.name, type=self.type)

        def _download():
            return art.download(root=self.artifact_dir)

        pkl_downloaded = Pickled(f"{self.cache_dir}/{self.name}_path.pkl", _download)
        if self.validate_artifact:
            pkl_downloaded.clear()
        else:
            logger.warning(f"not validating wandb artifact at {self.name}")
        return pkl_downloaded.value


@dataclass
class ArtifactUser:
    wandb: wandb
    validate_artifact: bool
    artifact_dir: str
    cache_dir: str

    def get_artifact(self, name, type):
        return CachedArtifact(self.wandb, self.validate_artifact, self.artifact_dir, self.cache_dir, name, type)


class EpochArtifacts:
    def __init__(self,
                 identifier: str,
                 type: str,
                 wandb_artifact_finder: WandbArtifactFinder,
                 remote_artifact_factory: RemoteArtifactFactory,
                 artifact_logger: ArtifactLogger,
                 name: str
                 ):
        self.wandb_artifact_finder = wandb_artifact_finder
        self.remote_artifact_factory = remote_artifact_factory
        self.artifact_logger = artifact_logger
        self.identifier = identifier
        self.type = type
        self.name = name
        arts = self.wandb_artifact_finder.find_artifacts()
        arts = [a for a in arts if a.type == self.type]
        arts = [a for a in arts if a.art.metadata.get("identifier") == self.identifier]
        self.epoch_to_art = {int(a.art.metadata["epoch"]):
                                 a.to_remote_artifact() for a in arts}

    def save(self, name, files_or_dirs, epoch, metadata: dict = None):
        return self.artifact_logger.save(
            name=name,
            type=self.type,
            file_or_dirs=files_or_dirs, metadata=dict(
                identifier=self.identifier,
                epoch=epoch,
                exp_name=self.name,
                **metadata
            ))

    @property
    def epochs(self):
        return sorted(self.epoch_to_art.keys())

    def load(self, epoch) -> RemoteArtifact:
        return self.epoch_to_art[epoch]


class KeyValueIO(metaclass=abc.ABCMeta):
    @abstractmethod
    def save(self, obj, key_values: dict):
        pass

    @abstractmethod
    def load(self, key):
        pass

    def keys(self) -> list:
        pass


@dataclass
class ArtifactKeyValueIo:
    artifact_savable: ArtifactSavable
    artifact_logger: ArtifactLogger
    wandb_artifact_finder: WandbArtifactFinder

    def save(self, obj, key_values):
        p = os.path.join(wandb.run.dir, str(uuid.uuid4())[:10])
        os.mkdir(p)
        res = self.artifact_savable.save(
            obj=obj,
            dst=p
        )
        self.artifact_logger.save(
            name=key_values.get("name") or "undefined",
            type=key_values.get("type") or "undefined",
            file_or_dirs=res,
            metadata=dict(
                protocol="key_value",
                **key_values
            )
        )

    def load(self, key, value):
        arts = self.wandb_artifact_finder.find_artifacts()
        arts = [a for a in arts if a.metadata.get("protocol") == "key_value"]
        for a in arts:
            if a.metadata.get(key) == value:
                return a


@dataclass
class ArtifactEpochIo:
    artifact_key_value_io: ArtifactKeyValueIo
    name: str
    type: str

    def save(self, obj, epoch):
        return self.artifact_key_value_io.save(obj, dict(
            epoch=epoch,
            name=self.name,
            type=self.type
        ))

    def load(self, epoch):
        return self.artifact_key_value_io.load("epoch", epoch)


@dataclass
class ArtifactKeyValueIoFactory:
    artifact_logger: ArtifactLogger
    wandb_artifact_finder: WandbArtifactFinder

    def get(self, savable: ArtifactSavable):
        return ArtifactKeyValueIo(
            artifact_savable=savable,
            artifact_logger=self.artifact_logger,
            wandb_artifact_finder=self.wandb_artifact_finder,
        )


@dataclass
class ArtifactEpochIoFactory:
    artifact_key_value_io_factory: ArtifactKeyValueIoFactory

    def get(self, savable: ArtifactSavable, name, type):
        kvio = self.artifact_key_value_io_factory.get(savable)
        return ArtifactEpochIo(artifact_key_value_io=kvio, name=name, type=type)


@dataclass
class EpochArtifactsFactory:
    wandb_artifact_finder: WandbArtifactFinder
    remote_artifact_factory: RemoteArtifactFactory
    artifact_logger: ArtifactLogger
    name: str

    def get(self, identifier, type) -> EpochArtifacts:
        return EpochArtifacts(
            wandb_artifact_finder=self.wandb_artifact_finder,
            remote_artifact_factory=self.remote_artifact_factory,
            artifact_logger=self.artifact_logger,
            identifier=identifier,
            type=type,
            name=self.name
        )


