import os
from dataclasses import dataclass

import PIL
from PIL import Image

from wandb.apis.public import Run

from data_tree import logger


@dataclass
class WandbSavedImage:
    run:Run
    info:dict
    def download(self)->PIL.Image.Image:
        if os.path.exists(self.info["path"]):
            return Image.open(self.info["path"])
        logger.debug(f"downloading image from wandb:{self.info}")
        with self.run.file(self.info["path"]).download(replace=True) as f:
            img = Image.open(f.name).resize((256,256))
        return img
    def __hash__(self):
        return hash(tuple(self.info.items()))

    def __getstate__(self):
        #wandb.Api().run()
        return self.run.path,self.info

    def __setstate__(self, state):
        import wandb
        path,info = state
        run = wandb.Api().run("/".join(path))
        self.run = run
        self.info = info

