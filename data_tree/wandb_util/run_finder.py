from dataclasses import dataclass
from typing import List

import numpy as np
import wandb
from cytoolz import merge

from wandb.apis.public import Run, Runs

from data_tree import logger
from data_tree.wandb_util import RunAndEpoch
from data_tree.wandb_util.tabulation import tabulate_dicts
from data_tree.wandb_util.run_ops import WandbRunOps


@dataclass
class WandbRunFinder:
    wandb: wandb
    name: str
    wandb_user: str

    def __post_init__(self):
        self.api = wandb.Api(timeout=60)
        self.project = self.wandb.run.project

    def find_by_query_and_name(self, query: dict = None) -> List[Run]:
        logger.info(f"looking for runs..{query}")
        query = merge(
            {"config.name": self.name},
            query or dict()
        )
        res = self.api.runs(f"{self.wandb_user}/{self.project}", query)
        logger.info(f"found runs:\n{tabulate_dicts([WandbRunOps(r).as_dict() for r in list(res)])}")
        return list(res)

    def find_by_query(self, query: dict = None) -> Runs:
        return self.api.runs(f"{self.wandb_user}/{self.project}", query)

    def find_by_job_type(self, job_type: str) -> Runs:
        return self.find_by_query(
            query={"jobType": job_type}
        )

    def find_by_tags(self, tags: List[str]) -> List[Run]:
        if isinstance(tags,str):
            tags = [tags]
        query = {"tags": {"$in": tags}}
        return list(self.api.runs(f"{self.wandb_user}/{self.project}", query))

    def find_by_job_type_and_name(self, job_type: str) -> List[Run]:
        query = {"jobType": job_type}
        return self.find_by_query_and_name(query)

    def find_finished_run_by_job_type(self, job_type) -> List[Run]:
        runs = self.find_by_job_type_and_name(job_type)
        return [r for r in runs if r.state == "finished"]

    def epoch_and_metrics(self, job_type, metric):
        res = []
        for r in self.find_by_job_type_and_name(job_type):
            history = r.history()
            if not history.empty and metric in history.columns:
                epochs = history["epoch"]
                metrics = history[metric]
                mask = ~np.isnan(metrics)
                epochs = epochs[mask]
                metrics = metrics[mask]
                res.append(history[mask])
        return res

    def find_run_and_epoch_by_lowest_metric(self, job_type: str, metric: str) -> RunAndEpoch:

        candidates = []
        for r in self.find_by_job_type_and_name(job_type):
            history = r.history()
            if not history.empty and metric in history.columns:
                epochs = history["epoch"]
                metrics = history[metric]
                mask = ~np.isnan(metrics)
                epochs = epochs[mask]
                metrics = metrics[mask]
                logger.info(history[mask][["epoch", metric]])
                best_idx = metrics.argmin()
                logger.info(f"found idx:{best_idx},epoch:{epochs.iloc[best_idx]}")
                candidates.append(RunAndEpoch(r, epochs.iloc[best_idx], metrics.iloc[best_idx]))
        if len(candidates) == 0:
            raise RuntimeError(f"no evaluate-all result found! thus the candidates is empty.")
        return max(candidates, key=lambda i: i.epoch)
