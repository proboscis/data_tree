import itertools
from typing import Any, Union, Type

import PIL
import numpy as np
from loguru import logger
from torch import Tensor

from _series import Series
from data_tree.auto_data.omni_solver import Conversion, RuleResult
from pampy import match

from coconut.visualization import infer_widget
from data_tree import AutoImage


def mapper_to_rr(mapper, name=None):
    return [RuleResult([], lambda: mapper, 0.5, name or mapper.__name__)]


def recursive_conversion_rule(src, dst, mapper):
    def _rule_impl(query):
        return match(query,
                     Conversion(src, dst), lambda *args: (
                mapper_to_rr(mapper),
            )[-1],
                     Conversion(Any, dst), lambda orig: (
                [RuleResult(
                    [Conversion(orig, src)],
                    lambda f: mapper_to_rr(lambda x: mapper(f(x))),
                    1,
                    f"any => {orig} => {dst}"
                )],)[-1],
                     Any, lambda x: []
                     )

    return _rule_impl


def provide_conversion_widget(query: Union[Conversion, Any]):
    rrules = [
        recursive_conversion_rule(Type[AutoImage], "widget", infer_widget),
        recursive_conversion_rule(Type[Series], "widget", infer_widget),
        recursive_conversion_rule(Type[PIL.Image.Image], "widget", infer_widget),
        recursive_conversion_rule(Type[Tensor], "widget", infer_widget),
        recursive_conversion_rule(Type[np.ndarray], "widget", infer_widget),
    ]
    res = [r(query) for r in rrules]
    return list(itertools.chain(*res))

# ok now this thingy works.
# but actually, it is harder to state the transition from dst to src
# since the conversion is usually the process of losing information.
# so it is harder to state what exactly can be the source of the query.
# the motivation for what I started implementing this was the need of implementing type class for widget.
# I wanted to say that any dict with auto state can be a widget.