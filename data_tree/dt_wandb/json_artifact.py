import abc
import json
from dataclasses import dataclass
from glob import glob
from typing import Union

import pandas
import wandb

from data_tree import logger
from data_tree.util import ensure_path_exists
from data_tree.wandb_util import random_artifact_file_path
from data_tree.wandb_util.artifact_identifier import ArtifactMetadata, ArtifactIdentifier
from data_tree.wandb_util.artifact_logger import ArtifactLogger
from data_tree.wandb_util.artifact_wrapper import RemoteArtifactFactory
from data_tree.wandb_util.local_artifact import LocalArtifact
from pinjected.di.injected import Injected


@dataclass
class JsonArtifactLogger:
    artifact_logger: ArtifactLogger
    wandb:"wandb"

    def log(self, name: str, type: str, obj: Union[list, dict], description=None, metadata=None,
            aliases=None) -> LocalArtifact:
        return self.log2(obj, ArtifactMetadata(
            identifier=ArtifactIdentifier(
                name=name,
                type=type,
                version=aliases
            ),
            description=description,
            metadata=metadata
        ))

    def log2(self, obj: Union[dict, list], metadata: ArtifactMetadata) -> LocalArtifact:
        from loguru import logger
        logger.info(f"calling log_artifact")
        path = random_artifact_file_path("json",self.wandb)
        ensure_path_exists(path)
        with open(path, "w") as f:
            json.dump(obj, f)

        art = self.artifact_logger.save2(path, metadata)
        return art



@dataclass
class TableArtifactLogger:
    artifact_logger: ArtifactLogger
    """for use of image visualizations"""

    def log(self, metadata: ArtifactMetadata, data: pandas.DataFrame) -> LocalArtifact:
        """
        :param metadata:
        :param data: PIL.Image needs to be converted to wandb.Image
        :return:
        """
        table = wandb.Table(dataframe=data)
        logger.info(f"logging dataframe to wandb with metadata: {metadata}")
        return self.artifact_logger.save_artifact2(
            target=table,
            target_name=metadata.identifier.name,
            artifact_metadata=metadata
        )


@dataclass
class JsonArtifactLoader:
    remote_artifact_factory: RemoteArtifactFactory

    def load(self, identifier, type):
        art = self.remote_artifact_factory.from_identifier(identifier=identifier, type=type)
        downloaded = art.download()
        try:
            jp = glob(f"{downloaded}/*.json")[0]
            with open(jp, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"cannot find a json file downloaded at {downloaded}.")
            raise e
        return data

    def metadata(self, identifier, type):
        self.remote_artifact_factory.from_identifier(identifier=identifier, type=type)

    def load_identifier(self, artifact_identifier: Union[ArtifactIdentifier, str]):
        if isinstance(artifact_identifier, str):
            type, name, version = artifact_identifier.split(":")
        else:
            type, name, version = artifact_identifier.type, artifact_identifier.name, artifact_identifier.version
        identifier = ":".join([name, version])
        return self.load(identifier, type)


class JsonArtifact(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def keys(self):
        pass

    @abc.abstractmethod
    def items(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


def __init__impl(self, json_artifact_loader: JsonArtifactLoader):
    self.data = json_artifact_loader.load_identifier(self.identifier)


def get_json_artifact(
        json_artifact_loader: JsonArtifactLoader,
        identifier: ArtifactIdentifier
):
    return json_artifact_loader.load_identifier(identifier)


def __getitem__impl(self, item):
    return self.data[item]


def __len__impl(self):
    return len(self.data)


def keys_impl(self):
    return self.data.keys()


def items_impl(self):
    yield from self.data.items()


def json_artifact_deprecated(
        artifact_identifier: Union[ArtifactIdentifier, str]
):
    artifact_identifier = ensure_artifact_identifier(artifact_identifier)

    _class = type(f"JsonArtifact_{artifact_identifier.type}_{artifact_identifier.name}_{artifact_identifier.version}",
                  (JsonArtifact,),
                  {
                      "__init__": __init__impl,
                      "__getitem__": __getitem__impl,
                      "__len__": __len__impl,
                      "keys": keys_impl,
                      "items": items_impl,
                      "identifier": artifact_identifier,
                  })
    return _class


def json_artifact(
        artifact_identifier: Union[ArtifactIdentifier, str]
):
    artifact_identifier = ensure_artifact_identifier(artifact_identifier)
    return Injected.bind(get_json_artifact, identifier=lambda:artifact_identifier)


def ensure_artifact_identifier(identifier: Union[ArtifactIdentifier, str]):
    if isinstance(identifier, str):
        _type, name, version = identifier.split(":")
        identifier = ArtifactIdentifier(
            name=name,
            type=_type,
            version=version
        )
    return identifier
