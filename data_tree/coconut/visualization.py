#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x5e23dbf6

# Compiled with Coconut version 1.4.1 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut, _coconut_MatchError, _coconut_tail_call, _coconut_tco, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert
if _coconut_sys.version_info >= (3,):
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import ipywidgets as widgets
from pprint import pformat
from PIL import Image
import PIL
from logzero import logger
import numpy as np
import pandas as pd

nn = lambda none, func: func(none) if none is not None else None
L = _coconut.functools.partial(Image.fromarray, mode="L")
RGB = _coconut.functools.partial(Image.fromarray, mode="RGB")
RGBA = _coconut.functools.partial(Image.fromarray, mode="RGBA")
batch_image_to_image = lambda ary: ary.transpose((1, 0, 2, 3)).reshape((ary.shape[2], -1, ary.shape[3]))
batch_L_to_img = lambda ary: ary
img_to_widget = lambda value: widgets.Box([widgets.Image(value=value._repr_png_(), format="png")])

@_coconut_tco
def ary_to_image(value):
    if not (isinstance)(value, np.ndarray):
        return None

    if value.dtype is not np.uint8:
        logger.warning("automatically converting dtype {value.dtype} to uint8 for visualization")
        value = value.astype("uint8")

    _coconut_match_to = value.shape
    _coconut_case_check_0 = False
    if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4) and (_coconut_match_to[3] == 1):
        bs = _coconut_match_to[0]
        h = _coconut_match_to[1]
        w = _coconut_match_to[2]
        _coconut_case_check_0 = True
    if _coconut_case_check_0:
        return _coconut_tail_call((L), value.transpose(1, 0, 2, 3).reshape(h, bs * w))
    if not _coconut_case_check_0:  # batch of rgbs
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4) and (_coconut_match_to[3] == 3):  # batch of rgbs
            _coconut_case_check_0 = True  # batch of rgbs
        if _coconut_case_check_0:  # batch of rgbs
            return _coconut_tail_call((RGB), (batch_image_to_image)(value))
    if not _coconut_case_check_0:  # batch of rgbas
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4) and (_coconut_match_to[3] == 4):  # batch of rgbas
            _coconut_case_check_0 = True  # batch of rgbas
        if _coconut_case_check_0:  # batch of rgbas
            return _coconut_tail_call((RGBA), (batch_image_to_image)(value))
    if not _coconut_case_check_0:  # a gray image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut_match_to[2] == 1):  # a gray image
            _coconut_case_check_0 = True  # a gray image
        if _coconut_case_check_0:  # a gray image
            return _coconut_tail_call((L), value[:, :, 0])
    if not _coconut_case_check_0:  # an RGB image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut_match_to[2] == 3):  # an RGB image
            _coconut_case_check_0 = True  # an RGB image
        if _coconut_case_check_0:  # an RGB image
            return _coconut_tail_call((RGB), value)
    if not _coconut_case_check_0:  # an RGBA image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut_match_to[2] == 4):  # an RGBA image
            _coconut_case_check_0 = True  # an RGBA image
        if _coconut_case_check_0:  # an RGBA image
            return _coconut_tail_call((RGBA), value)
    if not _coconut_case_check_0:  # batch of gray image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3):  # batch of gray image
            bs = _coconut_match_to[0]  # batch of gray image
            h = _coconut_match_to[1]  # batch of gray image
            w = _coconut_match_to[2]  # batch of gray image
            _coconut_case_check_0 = True  # batch of gray image
        if _coconut_case_check_0 and not (w > 4):  # batch of gray image
            _coconut_case_check_0 = False  # batch of gray image
        if _coconut_case_check_0:  # batch of gray image
            return _coconut_tail_call((L), value.transpose(1, 0, 2).reshape(h, bs * w))
    if not _coconut_case_check_0:  # gray image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2):  # gray image
            _coconut_case_check_0 = True  # gray image
        if _coconut_case_check_0:  # gray image
            return _coconut_tail_call((L), value)
    if not _coconut_case_check_0:
        return None

@_coconut_tco
def ary_stat_widget(ary):
    return _coconut_tail_call(widgets.VBox, [widgets.Label(value=pformat(str(dict(shape=ary.shape, dtype=ary.dtype)))), widgets.HTML(pd.DataFrame(pd.Series(ary.ravel()).describe()).transpose()._repr_html_())])


@_coconut_tco
def ary_to_widget(ary):
    stat_widget = ary_stat_widget(ary)
    viz_widget = (nn)(ary_to_image(ary), img_to_widget)
    viz_widget = widgets.Label(value=str(ary)) if viz_widget is None else viz_widget
    return _coconut_tail_call(widgets.VBox, [viz_widget, stat_widget])

@_coconut_tco
def infer_widget(value):
    _coconut_match_to = value
    _coconut_case_check_1 = False
    if _coconut.isinstance(_coconut_match_to, np.ndarray):
        _coconut_case_check_1 = True
    if _coconut_case_check_1:
        return _coconut_tail_call(ary_to_widget, value)
    if not _coconut_case_check_1:
        return value
