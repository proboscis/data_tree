from dataclasses import dataclass
from pathlib import Path

from data_tree import logger
from data_tree.wandb_util.artifact_identifier import ArtifactIdentifier, ArtifactMetadata, ArtifactMetadataLike
from data_tree.dt_wandb.json_artifact import JsonArtifactLogger, JsonArtifactLoader
from data_tree.storage_manager import FileStorageManager



@dataclass
class ManagedPathArtifactLogger:
    storage_manager: FileStorageManager
    json_artifact_logger: JsonArtifactLogger

    def log_path(self, conditions: dict, name: str, type: str, description=None, metadata=None, aliases=None) -> str:
        """use a path returned from this to save anything to be tracked which cannot be uploaded to wandb"""
        assert len(conditions)
        path = self.storage_manager.get_store_path_candidates(basename=name, **conditions)[0]
        logger.info(f"logged path:{path}")
        self.json_artifact_logger.log(name=name, type=type, obj=dict(
            conditions=conditions,
            path=path,
            description=description,
            metadata=metadata,
            aliases=aliases
        ))
        return path

    def log_artifact_path(self, metadata: ArtifactMetadata):
        """
        returns a managed path for storing huge data
        :param metadata:
        :return:
        """
        conds = dict(
            name=metadata.identifier.name,
            type=metadata.identifier.type,
            version=metadata.identifier.version,
            metadata=metadata.metadata
        )
        return self.log_path(conditions=conds, name=metadata.identifier.name, type=metadata.identifier.type,
                             description=metadata.description, metadata=metadata.metadata, aliases=metadata.aliases)



@dataclass
class ManagedPathArtifactLoader:
    json_artifact_loader: JsonArtifactLoader
    storage_manager: FileStorageManager

    def get_path(self, identifier: str, type: str):
        data = self.json_artifact_loader.load(identifier, type)
        # "conditions is not in data??
        return self.storage_manager.find(**data["conditions"])

    def get_path_identifier(self,idt:ArtifactIdentifier):
        data = self.json_artifact_loader.load_identifier(idt)
        return self.storage_manager.find(**data['conditions'])
