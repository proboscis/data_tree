import os
from dataclasses import dataclass

import PIL
from PIL import Image
from loguru import logger
from wandb.apis.public import Run


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