from dataclasses import dataclass

import wandb
from cytoolz import valmap
from pandas import DataFrame
from wandb.apis.public import Run

from data_tree.dt_wandb.wandb_util import WandbSavedImage

wandb.Image

@dataclass
class WandbRunOps:
    src: Run

    def as_dict(self):
        return dict(
            id=self.src.id,
            name=self.src.name,
            path=self.src.path,
            state=self.src.state
        )

    def _parse_element(self, elem):
        match elem:
            case {"_type": "image-file"}:
                return WandbSavedImage(self.src, elem)
            case _:
                assert "_type" not in elem if isinstance(elem, dict) else True
                return elem

    def history(self):
        df: DataFrame = self.src.history()
        df = df.applymap(self._parse_element)
        return df

    def summary(self):
        return valmap(self._parse_element,self.src.summary)

    def __getstate__(self):
        return self.src.path
    def __setstate__(self, state):
        self.src = wandb.Api().run("/".join(state))