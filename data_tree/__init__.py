from data_tree.cache import ConditionedFilePathProvider
from data_tree._series import Series
from data_tree.table import Table
from data_tree.indexer import Indexer, IdentityIndexer
from data_tree.coconut.convert import AutoImage
from data_tree.coconut.omni_converter import auto_img as omni_auto_img
from data_tree.coconut.omni_converter import unlist
from data_tree.coconut.omni_converter import SOLVER as omni_solver
from data_tree.table import Table,Tables

series = Series.from_iterable
managed_cache = ConditionedFilePathProvider
auto_image = AutoImage
auto = omni_auto_img
"""
example:torch,float32,BCHW,RGB,0_1
"""
unlist_auto = unlist
solver = omni_solver

def auto_img(codec):
    def _l(img):
        return AutoImage(img,codec)
    return _l
