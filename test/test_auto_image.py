
from data_tree.coconut.convert import AutoImage
from data_tree import auto_img
import numpy as np
def test_visdom_show():
    img = np.ones((3, 100, 100)) * 0.5
    img = auto_img("numpy,float32,CHW,RGB,0_1")(img)
    img.visdom()