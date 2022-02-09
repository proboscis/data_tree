from typing import Callable, Any

from omni_converter.auto_data.auto_v2 import RuledData, IAutoData
from omni_converter.solver.rules import AutoRuleBook


def managed_cache(root_dir):
    from data_tree.cache import ConditionedFilePathProvider
    return ConditionedFilePathProvider(root_dir)


def series(iterable):
    from data_tree._series import Series
    return Series.from_iterable(iterable)


def unlist_auto(items):
    from data_tree.coconut.omni_converter import unlist
    return unlist(items)


def auto(format) -> Callable[[Any], RuledData]:
    return auto_img(format)


def resolve_format(value, format_resolution_rule: AutoRuleBook):
    from omni_converter.auto_data.auto_v2 import AutoData2
    return AutoData2(value=value, format=value).with_rules(format_resolution_rule).to("format")


def ruled(value,
          rules: AutoRuleBook,
          format=None,
          format_resolution_rule: AutoRuleBook = None) -> "RuledData":
    if format_resolution_rule is None:
        from omni_converter.auto_data.format_resolution import FORMAT_RESOLUTION_RULES
        format_resolution_rule = FORMAT_RESOLUTION_RULES
    if format is None:
        # try to automatically get format from value given rules
        format = resolve_format(value, format_resolution_rule)
    from omni_converter.auto_data.auto_v2 import AutoData2
    return AutoData2(value=value, format=format).with_rules(rules)


def auto_image(format):
    return get_cv_ruled(format)


def auto_img(codec):
    return get_cv_ruled(codec)


def get_cv_ruled(format)->IAutoData:
    # every time the solver is recreated.
    from omni_converter import auto as auto_v2
    def _l(value) -> RuledData:
        return auto_v2(value=value, format=format)

    return _l
