from dataclasses import dataclass

import wandb

from data_tree.wandb_util.artifact_identifier import ArtifactIdentifier


@dataclass
class WandbArtifactGetter:
    wandb: wandb
    wandb_user: str

    def use_artifact(self, name: str, type: str, user=None, project=None) -> wandb.Artifact:
        from loguru import logger
        user = user or self.wandb_user
        project = project or self.wandb.run.project
        assert project is not None
        tgt = f"{user}/{project}/{name}"
        logger.info(f"using artifact {tgt} with type {type}")

        return self.wandb.run.use_artifact(tgt, type=type)

    def use_artifact2(self,idt:ArtifactIdentifier)->wandb.Artifact:
        return self.use_artifact(
            name=idt.identifier_str(),
            type=idt.type,
        )

    def artifact_for_save(self,
                          name: str,
                          type: str,
                          metadata: dict = None,
                          description: str = None):
        return wandb.Artifact(name, type, metadata=metadata, description=description)

@dataclass
class PublicWandbArtifactGetter:
    wandb_api: wandb.Api
    def get_artifact(self,user,project,name,alias):
        art = self.wandb_api.artifact(f"{user}/{project}/{name}:{alias}")
        return art