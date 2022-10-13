import os
from dataclasses import dataclass
from types import ModuleType
from uuid import uuid4

import wandb

from wandb.apis.public import Run

# from archpainter.wandb_util.artifact_finder import WandbArtifactFinder
from data_tree.util import ensure_path_exists


@dataclass
class RunAndEpoch:
    run: Run
    epoch: int
    metric: float

    def find_artifact(self, typ) -> wandb.Artifact:
        arts = [a for a in self.run.logged_artifacts() if a.type == typ]
        epoch2art = {int(a.metadata["epoch"]): a for a in arts}
        logger.info(epoch2art)
        return epoch2art[self.epoch]


@dataclass
class WandbFileLoader:
    wandb: wandb

    def load_saved_file(self, run_path, file_path):
        return self.wandb.restore(file_path, run_path=run_path).name

@dataclass
class RandomArtifactPathProvider:
    wandb:ModuleType
    def random_artifact_file_path(self,ext):
        return random_artifact_file_path(ext,self.wandb)


def random_artifact_file_path(ext,wandb):
    p = os.path.join(wandb.run.path, str(uuid4())[:8], f"artifact.{ext}")
    ensure_path_exists(p)
    return p


def random_artifact_tmp_dir(wandb):
    p = os.path.join(wandb.run.path, str(uuid4())[:8])
    os.mkdir(p)
    return p
