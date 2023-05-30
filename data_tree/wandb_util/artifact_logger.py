import os
from dataclasses import dataclass
from typing import Union, List

import wandb
from loguru import logger

from data_tree.wandb_util.artifact_getter import WandbArtifactGetter
from data_tree.wandb_util.artifact_identifier import ArtifactMetadata
from data_tree.wandb_util.local_artifact import LocalArtifact


@dataclass
class LoggableArtifact:
    metadata: ArtifactMetadata
    file_or_dirs: Union[str, List[str]]


@dataclass
class ArtifactLogger:
    wandb: wandb
    wandb_artifact_getter: WandbArtifactGetter

    def save_artifact(self, art: LoggableArtifact):
        return self.save2(art.file_or_dirs, art.metadata)

    def save2(self, file_or_dirs: Union[str, List[str]], metadata: ArtifactMetadata):
        if metadata.identifier.version is None:
            aliases = []
        elif metadata.identifier.version != "latest":
            aliases = [metadata.identifier.version]
        else:
            aliases = []
        aliases += metadata.aliases
        return self.save(name=metadata.identifier.name, file_or_dirs=file_or_dirs, description=metadata.description,
                         aliases=aliases, type=metadata.identifier.type, metadata=metadata.metadata)

    def save(self, name, type,
             file_or_dirs: Union[str, List[str]],
             description: str = None,
             metadata: dict = None,
             aliases: list = None) -> LocalArtifact:
        logger.info(f"making artifact with name:{name}, type:{type}")
        art = self.wandb_artifact_getter.artifact_for_save(name, type=type, metadata=metadata, description=description)
        if isinstance(file_or_dirs, str):
            file_or_dirs = [file_or_dirs]
        for p in file_or_dirs:
            if os.path.isdir(p):
                logger.info(f"adding artifact from dir:{p}")
                art.add_dir(p)
            else:
                logger.info(f"adding artifact from file:{p}")
                art.add_file(p)
        logger.info(f"logging artifact... name={name}/aliases={aliases}")
        self.wandb.log_artifact(art, aliases=aliases)
        logger.info(f"logging artifact...done.")
        return LocalArtifact(self.wandb, art)

    def save_table(self, name, type, table, description: str = None, metadata=None):
        art = self.wandb_artifact_getter.artifact_for_save(name, type=type, metadata=metadata, description=description)
        art.add(table, type)
        logger.info(f"logging artifact {name}...")
        self.wandb.log_artifact(art)
        logger.info(f"logging artifact...done.")
        return LocalArtifact(self.wandb, art)

    def save_artifact(self,
                      name,
                      type,
                      object,
                      object_name: str,
                      description=None,
                      metadata=None):
        logger.info(f"making artifact with name:{name}, type:{type}")
        art = self.wandb_artifact_getter.artifact_for_save(name, type=type, metadata=metadata, description=description)
        art.add(object, object_name)
        logger.info(f"logging artifact...")
        self.wandb.log_artifact(art)
        logger.info(f"logging artifact...done.")
        return LocalArtifact(self.wandb, art)

    def save_artifact2(self, target, target_name, artifact_metadata: ArtifactMetadata):
        logger.info(f"creating artifact with metadata:{artifact_metadata}")
        art = self.wandb_artifact_getter.artifact_for_save(
            name=artifact_metadata.identifier.name,
            type=artifact_metadata.identifier.type,
            metadata=artifact_metadata.metadata,
            description=artifact_metadata.description,
        )
        art.add(target, target_name)
        self.wandb.log_artifact(art, aliases=artifact_metadata.aliases)
        logger.info(f"logging artifact...done.")
        return LocalArtifact(self.wandb, art)
