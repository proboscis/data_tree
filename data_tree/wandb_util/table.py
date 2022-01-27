from dataclasses import dataclass

import wandb
from cytoolz import keymap
from tqdm import tqdm

from data_tree.wandb_util.artifact_logger import ArtifactLogger
from data_tree._series import Series


@dataclass
class PointsVisualizer:
    """
    use this visualize points as artifact
    """
    artifact_logger: ArtifactLogger

    def visualize(self, targets: Series[dict], name: str):
        converter = lambda p: keymap(lambda k: k.replace(".", "_"), p)
        converted = targets.map(converter)
        keys = [k.replace(".", "_") for k in sorted(list(targets[0].keys()))]
        table = wandb.Table(keys)
        for p in tqdm(converted, desc="adding data for table artifact.."):
            data = [p[k] for k in keys]
            table.add_data(*data)
        return self.visualize_table(table, name)

    def visualize_table(self, table: wandb.Table, name: str):
        return self.artifact_logger.save_table(
            name=name,
            type="points_visualization",
            table=table
        )
