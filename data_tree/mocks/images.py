import numpy as np

from omni_converter.auto_data.auto_v2 import IAutoData
from strd.utilities.rulebook import legacy_auto as auto

def dummy_images(n=9,auto=auto) -> IAutoData:
    """
    :return:AutoImage(["image,RGB,RGB"])
    """
    imgs = auto("numpy,int32,BHWC,RGB,None", np.random.random((n, 256, 256, 3))).convert(
        "[image,RGB,RGB]")
    return imgs


def dummy_filled_images(n=9, color=(0, 0, 0),auto=auto) -> IAutoData:
    """
    :return:AutoImage(["image,RGB,RGB"])
    """
    imgs = auto("numpy,int32,BHWC,RGB,0_1", np.zeros(n * 256 * 256 * 3).reshape((n, 256, 256, 3)))
    imgs.value[:] = color
    return imgs.convert("[image,RGB,RGB]")
