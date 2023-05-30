import torch
from torch import multiprocessing as mp

from data_tree.mp_util import MTServer, MPServer

mp.set_start_method("spawn", force=True)

import numpy as np


def torch_function(ary):
    with torch.no_grad():
        return torch.from_numpy(ary).cuda().detach().cpu().numpy()


def test_mt_torch():
    s = MTServer(torch_function)
    s.run()

def test_mp_torch():
    s = MPServer(torch_function)
    s.run()
    logger.info(f"query result:{s.query(np.zeros((100, 100)))}")
    s.stop()
