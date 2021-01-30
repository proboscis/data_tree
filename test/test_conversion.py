from IPython import embed
from loguru import logger
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

def test_list_vgg_prep_to_vgg_prep_batch():
    from data_tree import auto
    from archpainter.dataset.ask import ask256
    img = ask256.original_pairs[0][0]
    vgg_prep = auto("numpy,uint8,HWC,RGB,0_255")(img[1]).to("vgg_prep")
    converted = auto("vgg_prep")(vgg_prep).convert("vgg_prep_batch").to("numpy,uint8,HWC,RGB,0_255")
    # now, can we embed numbers into a state?
    embed()
