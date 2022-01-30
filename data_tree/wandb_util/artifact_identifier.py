from dataclasses import dataclass, field
from typing import Union

from loguru import logger


@dataclass
class ArtifactIdentifier:
    # TODO make this hold user,project
    name: str
    type: str
    version:str  = field(default='latest')

    def identifier_str(self) -> str:
        return f"{self.name}:{self.version if self.version is not None else 'latest'}"

    def exists(self, wandb):
        return artifact_exists(self, wandb)


@dataclass
class ArtifactMetadata:
    identifier: ArtifactIdentifier
    description: Union[str, None] = field(default=None)
    metadata: Union[dict, None] = field(default=None)
    aliases: Union[list] = field(default_factory=list)


def artifact_exists(art: ArtifactIdentifier, wandb) -> bool:
    api = wandb.Api()
    try:
        a = api.artifact(art.identifier_str())
        return True
    except Exception as e:
        logger.warning(f"artifct:{art} does not exist?:{e}")
        return False