from data_tree.cache import ConditionedFilePathProvider
from data_tree._series import Series
from data_tree.table import SeriesTable
from data_tree.indexer import Indexer, IdentityIndexer
from data_tree.coconut.convert import AutoImage
from data_tree.coconut.omni_converter import auto_img as omni_auto_img

series = Series.from_iterable
managed_cache = ConditionedFilePathProvider
auto_image = AutoImage
auto = omni_auto_img

def auto_img(codec):
    def _l(img):
        return AutoImage(img,codec)
    return _l
