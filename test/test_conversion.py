import pytest
from IPython import embed
from loguru import logger
import numpy as np
def test_rgb_to_yuv():
    from data_tree import auto
    from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz, pix2pix_rgb_batch, TORCH_XYZ_BATCH
    from archpainter.dataset.ask import ask256
    img = ask256.original_pairs[0][0]
    import numpy as np
    ary = auto("numpy,uint8,HWC,RGB,0_255")(img[1]).to("numpy,uint8,HWC,YCbCr,0_255")
    from matplotlib import pyplot as plt
    plt.figure()
    plt.hist(ary[:, :, 0].flatten(), bins=100)
    plt.hist(ary[:, :, 1].flatten(), bins=100)
    plt.hist(ary[:, :, 2].flatten(), bins=100)
    plt.show()
    #auto("numpy,uint8,HWC,YCbCr,0_255")(ary).convert("numpy,uint8,CHW,YCbCr,0_255").cast("numpy,uint8,BHW,L,0_255").show()
    #logger.info(auto("image,RGB,RGB")(None).converter("image,YCbCr,YCbCr"))
formats = [
    "numpy_rgb",
    "numpy_rgba",
    "torch,float32,BCHW,RGB,0_1",
    "image,RGB,RGB",
    "image,L,L",
    "vgg_prep",
    "vgg_prep_batch",
    "pix2pix,nc=3",
    "pix2pix_batch,nc=3"
]
@pytest.fixture(params=[0,1,2])
def conv_pair(request):
    p = request.param # 0 or 1 or 2
    return p


def test_list_vgg_prep_to_vgg_prep_batch():
    from data_tree import auto
    from archpainter.dataset.ask import ask256
    img = ask256.original_pairs[0][0]
    start = auto("numpy,uint8,HWC,RGB,0_255")(img[1])
    vgg_prep = start.convert("vgg_prep")
    inverted = vgg_prep.convert("numpy,uint8,HWC,RGB,0_255")
    diff = start.value - inverted.value
    assert abs(diff).max() < 1.001
    #assert np.abs(inverted.value - start.value).max() < 1
    converted = vgg_prep.convert("vgg_prep_batch").convert("numpy_rgb")
    converted.to("image,RGB,RGB,None:None:3")
    # now, can we embed numbers into a state?
    logger.warning(f"check outputs!")
    #embed()
