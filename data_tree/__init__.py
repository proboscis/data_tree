from typing import Callable, Any

from omni_converter.auto_data.auto_v2 import RuledData
from omni_converter.solver.rules import IAutoRuleBook, AutoRuleBook
from omni_cv_rules.legacy.auto_image import AutoImage


def managed_cache(root_dir):
    from data_tree.cache import ConditionedFilePathProvider
    return ConditionedFilePathProvider(root_dir)


def series(iterable):
    from data_tree._series import Series
    return Series.from_iterable(iterable)


def unlist_auto(items):
    from data_tree.coconut.omni_converter import unlist
    return unlist(items)


def auto(format) -> Callable[[Any], AutoImage]:
    return auto_img(format)


def resolve_format(value, format_resolution_rule: IAutoRuleBook):
    from omni_converter.auto_data.auto_v2 import AutoData2
    return AutoData2(value=value, format=value).with_rules(format_resolution_rule).to("format")


def ruled(value,
          format=None,
          rules: IAutoRuleBook = None,
          format_resolution_rule: IAutoRuleBook = None) -> "RuledData":
    if rules is None:
        from omni_converter.auto_data.default_rules import DEFAULT_RULES
        rules = DEFAULT_RULES
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
    # def _gen_ai(value):
    #     return AutoImage(value, format, solver_getter)
    #
    # return _gen_ai


def auto_img(codec):
    return get_cv_ruled(codec)
    # def _l(img):
    #     return AutoImage(img, codec, solver_getter)
    #
    # return _l

GLOBAL_RULEBOOK = AutoRuleBook()
GLOBAL_RULE = None
def update_global_rule():
    from omni_converter.auto_data.solver import DEFAULT_ASTAR_DESIGN
    from omni_cv_rules.rulebook import non_recursive_rules
    from omni_cv_rules.rulebook import recursive_rules
    d = DEFAULT_ASTAR_DESIGN.bind_instance(
        non_recursive_rules=non_recursive_rules+GLOBAL_RULEBOOK,
        recursive_rules=recursive_rules
    )
    global GLOBAL_RULE
    GLOBAL_RULE = d.provide("rulebook")
update_global_rule()



def get_cv_ruled(format):
    # every time the solver is recreated.

    def _l(value) -> RuledData:
        return ruled(value=value, format=format, rules=GLOBAL_RULE)

    return _l
