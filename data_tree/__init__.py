from typing import Callable, Any

import data_tree.util
from data_tree.auto_data.auto_image import AutoImage
from data_tree.auto_data.auto_v2 import RuledData, AutoData2
from data_tree.auto_data.format_resolution import FORMAT_RESOLUTION_RULES
from data_tree.auto_data.rules import AutoRuleBook

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


def auto(format)->Callable[[Any],AutoImage]:
    return auto_img(format)


def resolve_format(value, format_resolution_rule: AutoRuleBook):
    return AutoData2(value=value, format=value).with_rules(format_resolution_rule).to("format")


from data_tree.auto_data.default_rules import DEFAULT_RULES
def ruled(value,
          format=None,
          rules: AutoRuleBook = DEFAULT_RULES,
          format_resolution_rule: AutoRuleBook = FORMAT_RESOLUTION_RULES) -> RuledData:
    if format is None:
        # try to automatically get format from value given rules
        format = resolve_format(value, format_resolution_rule)
    return AutoData2(value=value, format=format).with_rules(rules)


def auto_image(format):
    from data_tree.auto_data.auto_image import AutoImage

    def _gen_ai(value):
        from data_tree.auto_data.default_solvers import solver_getter
        return AutoImage(value, format, solver_getter)

    return _gen_ai


def auto_img(codec):
    from data_tree.auto_data.auto_image import AutoImage
    from data_tree.auto_data.default_solvers import solver_getter
    def _l(img):
        return AutoImage(img, codec, solver_getter)

    return _l
