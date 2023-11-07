import numpy as np

from omni_converter.auto_data.auto_v2 import IAutoData
def dummy_images(n=9,auto=None) -> IAutoData:
    """
    :return:AutoImage(["image,RGB,RGB"])
    """
    if auto is None:
        from proboscis_image_rules.rulebook import auto
    imgs = auto("numpy,int32,BHWC,RGB,None", np.random.random((n, 256, 256, 3))).convert(
        "[image,RGB,RGB]")
    return imgs


def dummy_filled_images(n=9, color=(0, 0, 0),auto=None) -> IAutoData:
    """
    :return:AutoImage(["image,RGB,RGB"])
    """
    if auto is None:
        from proboscis_image_rules.rulebook import auto
    imgs = auto("numpy,int32,BHWC,RGB,0_1", np.zeros(n * 256 * 256 * 3).reshape((n, 256, 256, 3)))
    imgs.value[:] = color
    return imgs.convert("[image,RGB,RGB]")
