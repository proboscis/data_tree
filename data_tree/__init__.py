


def managed_cache(root_dir):
    from data_tree.cache import ConditionedFilePathProvider
    return ConditionedFilePathProvider(root_dir)

def series(iterable):
    from data_tree._series import Series
    return Series.from_iterable(iterable)

def unlist_auto(items):
    from data_tree.coconut.omni_converter import unlist
    return unlist(items)

def auto(format):
    from data_tree.coconut.omni_converter import auto_img as omni_auto_img
    return omni_auto_img(format)

def auto_image(format):
    from data_tree.auto_data.auto_image import AutoImage
    from data_tree.coconut.omni_converter import SOLVER as omni_solver

    def _gen_ai(value):
        return AutoImage(value,format,solver)


def auto_img(codec):
    def _l(img):
        return AutoImage(img, codec,solver)
    return _l
