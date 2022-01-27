from dataclasses import dataclass
from typing import List

from cytoolz import concat

from data_tree.wandb_util.artifact_wrapper import RemoteArtifactMetadataFactory, RemoteArtifactMetadata
from data_tree.wandb_util.run_finder import WandbRunFinder


@dataclass
class WandbArtifactFinder:
    wandb_run_finder: WandbRunFinder
    remote_artifact_metadata_factory:RemoteArtifactMetadataFactory

    def find_artifacts(self, run_query: dict = None) -> List[RemoteArtifactMetadata]:
        runs = self.wandb_run_finder.find_by_query_and_name(run_query)
        to_r = self.remote_artifact_metadata_factory.get
        arts = [to_r(a) for a in list(concat([r.logged_artifacts() for r in runs]))]
        return arts