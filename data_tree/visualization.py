import numpy as np
from ipywidgets import interactive
from numpy import array

from data_tree._series import Trace


def show_trace(trace: Trace):
    trace.metadata


def show_backtrack(tgt):
    def _(i):
        bts = tgt.backtrack(i)
        for bt in bts:
            figure()
            title_str = str(bt.dataset.__class__.__name__) + f": Y=>{bt.y}"
            if len(bt.x.shape) == 3:
                imshow(bt.x / 255)
            elif len(bt.x.shape) == 2:
                imshow(bt.x / 255, cmap="gray")
            elif isinstance(bt.x, np.ndarray):
                hist(bt.x, bins=100)
            else:
                title_str += f" , X=>{bt.x}"
            title(title_str)
        src = bts[-1]
        img = src.dataset.image(src.dataset.ids[src.index])
        figure(figsize=(5, 5))
        imshow(array(img))
        title("SRC")
        return src.y, exp(bts[0].x)

    return interactive(_, i=(0, len(tgt) - 1))
