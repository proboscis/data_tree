from dataclasses import dataclass
from typing import List

from cytoolz import concat

from tqdm import tqdm

from data_tree import logger
from data_tree.wandb_util.artifact_wrapper import RemoteArtifactMetadataFactory, RemoteArtifactMetadata
from data_tree.wandb_util.run_finder import WandbRunFinder
from data_tree.wandb_util.run_ops import WandbRunOps
from data_tree.wandb_util.tabulation import tabulate_dicts


@dataclass
class WandbArtifactFinder:
    wandb_run_finder: WandbRunFinder
    remote_artifact_metadata_factory: RemoteArtifactMetadataFactory

    def find_artifacts(self, run_query: dict = None) -> List[RemoteArtifactMetadata]:
        """
        finding artifacts takes a lot of time. what can I do?
        you can reduce the targetting runs
        :param run_query:
        :return:
        """
        runs = list(self.wandb_run_finder.find_by_query_and_name(run_query))
        logger.info(f"found runs:\n{tabulate_dicts([WandbRunOps(r).as_dict() for r in runs])}")
        to_r = self.remote_artifact_metadata_factory.get

        artifacts = list(concat([list(r.logged_artifacts()) for r in tqdm(runs, desc="fetching artifact info")]))

        metas = [to_r(a) for a in tqdm(artifacts,desc="retrieving metadata of remote artifacts")]
        return metas
