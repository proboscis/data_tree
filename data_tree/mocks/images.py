import numpy as np

from data_tree import auto_image
from data_tree.auto_data.auto_image import AutoImage


def dummy_images(n=9) -> AutoImage:
    """
    :return:AutoImage(["image,RGB,RGB"])
    """
    imgs = auto_image("numpy,int32,BHWC,RGB,None")(np.random.random((n, 256, 256, 3))).convert(
        "[image,RGB,RGB]")
    return imgs


def dummy_filled_images(n=9,color=(0,0,0))->AutoImage:
    """
    :return:AutoImage(["image,RGB,RGB"])
    """
    imgs = auto_image("numpy,int32,BHWC,RGB,0_1")(np.zeros(n * 256 * 256 * 3).reshape((n, 256, 256, 3)))
    imgs.value[:] = color
    return imgs.convert("[image,RGB,RGB]")