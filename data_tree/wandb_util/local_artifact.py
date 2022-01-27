from dataclasses import dataclass

import wandb

from data_tree.wandb_util.artifact_interface import IArtifact


@dataclass
class LocalArtifact(IArtifact):
    @property
    def metadata(self):
        return self.artifact.metadata

    wandb: wandb
    artifact: wandb.Artifact

    @property
    def name(self):
        self.artifact.wait()
        return self.artifact.name

    @property
    def version(self):
        self.artifact.wait()
        return self.artifact.version

    @property
    def type(self):
        return self.artifact.type