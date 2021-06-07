import data_tree.util

scantree = util.scantree
scan_images = util.scan_images
scan_image_paths = util.scan_image_paths

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
    from data_tree.coconut.omni_converter import solver_getter

    def _gen_ai(value):
        return AutoImage(value, format, solver_getter)

    return _gen_ai


def auto_img(codec):
    from data_tree.auto_data.auto_image import AutoImage
    from data_tree.coconut.omni_converter import SOLVER as omni_solver
    def _l(img):
        return AutoImage(img, codec, omni_solver)

    return _l
