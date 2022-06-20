from dataclasses import dataclass
from glob import glob

import torch
from filelock import FileLock

from data_tree.wandb_util import random_artifact_file_path, RandomArtifactPathProvider
from data_tree.wandb_util.artifact_identifier import ArtifactMetadata, ArtifactIdentifier
from data_tree.wandb_util.artifact_logger import ArtifactLogger
from data_tree.util import ensure_path_exists
from data_tree.wandb_util.artifact_wrapper import RemoteArtifactFactory


@dataclass
class TorchArtifactLogger:
    artifact_logger: ArtifactLogger
    random_artifact_path_provider: RandomArtifactPathProvider

    def log(self, name: str, type: str, obj: object, description=None, metadata=None, aliases=None) -> "LocalArtifact":
        path = self.random_artifact_path_provider.random_artifact_file_path("pth")
        ensure_path_exists(path)
        torch.save(obj, path)
        return self.artifact_logger.save(
            name=name,
            type=type,
            file_or_dirs=path,
            description=description,
            metadata=metadata, aliases=aliases
        )

    def log2(self, metadata: ArtifactMetadata, obj: object) -> "LocalArtifact":
        return self.log(
            name=metadata.identifier.name,
            type=metadata.identifier.type,
            obj=obj,
            description=metadata.description,
            metadata=metadata.metadata,
            aliases=metadata.aliases
        )


@dataclass
class TorchArtifactLoader:
    remote_artifact_factory: RemoteArtifactFactory
    device: str
    wandb_download_lock: FileLock

    def load(self, identifier, type: str):
        art = self.remote_artifact_factory.from_identifier(identifier=identifier, type=type)
        with self.wandb_download_lock:
            downloaded = art.download()
        torch_path = glob(f"{downloaded}/*.pth")[0]
        return torch.load(torch_path, map_location=self.device)

    def load2(self, identifier: ArtifactIdentifier):
        return self.load(identifier.identifier_str(), type=identifier.type)
