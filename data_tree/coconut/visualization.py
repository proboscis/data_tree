#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xa2c4f1f4

# Compiled with Coconut version 1.4.1 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get("__coconut__")
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules["__coconut__"]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import ipywidgets as widgets  # import ipywidgets as widgets
from pprintpp import pformat  # from pprintpp import pformat,pprint
from pprintpp import pprint  # from pprintpp import pformat,pprint
from PIL import Image  # from PIL import Image
import PIL  # import PIL
from logzero import logger  # from logzero import logger
import numpy as np  # import numpy as np
import pandas as pd  # import pandas as pd
from typing import Iterable  # from typing import Iterable
nn = lambda none, func: func(none) if none is not None else None  # nn = (none,func) -> func(none) if none is not None else None
L = _coconut.functools.partial(Image.fromarray, mode="L")  # L = Image.fromarray$(mode="L")
RGB = _coconut.functools.partial(Image.fromarray, mode="RGB")  # RGB = Image.fromarray$(mode="RGB")
RGBA = _coconut.functools.partial(Image.fromarray, mode="RGBA")  # RGBA = Image.fromarray$(mode="RGBA")
batch_image_to_image = lambda ary: ary.transpose((1, 0, 2, 3)).reshape((ary.shape[2], -1, ary.shape[3]))  # batch_image_to_image = ary->ary.transpose((1, 0, 2, 3)).reshape((ary.shape[2], -1, ary.shape[3]))
batch_L_to_img = lambda ary: ary  # batch_L_to_img = ary->ary 
img_to_widget = lambda value: widgets.Box([widgets.Image(value=value._repr_png_(), format="png")])  # img_to_widget = value ->widgets.Box([widgets.Image(value=value._repr_png_(),format="png")])

from bqplot import pyplot as blt  # from bqplot import pyplot as blt
from logzero import logger  # from logzero import logger
class DummyType(_coconut.collections.namedtuple("DummyType", "")):  # data DummyType
    __slots__ = ()  # data DummyType
    __ne__ = _coconut.object.__ne__  # data DummyType
    def __eq__(self, other):  # data DummyType
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data DummyType
    def __hash__(self):  # data DummyType
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data DummyType

try:  # try:
    import torch  #     import torch
    TensorType = torch.Tensor  #     TensorType = torch.Tensor
except Exception as e:  # except Exception as e:
    logger.warn("failed to load torch. no visualization for torch tensor.".format())  #     logger.warn(f"failed to load torch. no visualization for torch tensor.")
    TensorType = DummyType  #     TensorType = DummyType

def bqplot_hist(ary, title="histogram", bins=50):  # def bqplot_hist(ary, title="histogram", bins=50):
    fig = blt.figure(title=title)  #     fig = blt.figure(title=title)
    blt.hist(ary.flatten(), bins=bins)  #     blt.hist(ary.flatten(), bins=bins)
    fig.layout.width = "400px"  #     fig.layout.width = "400px"
    fig.layout.height = "300px"  #     fig.layout.height = "300px"
    return fig  #     return fig


def ary_to_image(value):  # def ary_to_image(value):
    if not (isinstance)(value, np.ndarray):  #     if not value `isinstance` np.ndarray:
        return None  #         return None        

    if value.dtype is not np.uint8:  #     if value.dtype is not np.uint8:
#logger.warning("automatically converting dtype {value.dtype} to uint8 for visualization")
        value = value.astype("uint8")  #         value = value.astype("uint8")

    _coconut_match_to = value.shape  #     case value.shape:
    _coconut_case_check_0 = False  #     case value.shape:
    if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4) and (_coconut_match_to[3] == 1):  #     case value.shape:
        bs = _coconut_match_to[0]  #     case value.shape:
        h = _coconut_match_to[1]  #     case value.shape:
        w = _coconut_match_to[2]  #     case value.shape:
        _coconut_case_check_0 = True  #     case value.shape:
    if _coconut_case_check_0:  #     case value.shape:
        return (L)(value.transpose(1, 0, 2, 3).reshape(h, bs * w))  #             return value.transpose(1,0,2,3).reshape(h,bs*w) |> L
    if not _coconut_case_check_0:  # batch of rgbs  #         match (_,_,_,3): # batch of rgbs
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4) and (_coconut_match_to[3] == 3):  # batch of rgbs  #         match (_,_,_,3): # batch of rgbs
            _coconut_case_check_0 = True  # batch of rgbs  #         match (_,_,_,3): # batch of rgbs
        if _coconut_case_check_0:  # batch of rgbs  #         match (_,_,_,3): # batch of rgbs
            return (RGB)((batch_image_to_image)(value))  #             return value |> batch_image_to_image |> RGB
    if not _coconut_case_check_0:  # batch of rgbas  #         match (_,_,_,4): # batch of rgbas
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4) and (_coconut_match_to[3] == 4):  # batch of rgbas  #         match (_,_,_,4): # batch of rgbas
            _coconut_case_check_0 = True  # batch of rgbas  #         match (_,_,_,4): # batch of rgbas
        if _coconut_case_check_0:  # batch of rgbas  #         match (_,_,_,4): # batch of rgbas
            return (RGBA)((batch_image_to_image)(value))  #             return value |> batch_image_to_image |> RGBA
    if not _coconut_case_check_0:  # a gray image  #         match (_,_,1): # a gray image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut_match_to[2] == 1):  # a gray image  #         match (_,_,1): # a gray image
            _coconut_case_check_0 = True  # a gray image  #         match (_,_,1): # a gray image
        if _coconut_case_check_0:  # a gray image  #         match (_,_,1): # a gray image
            return (L)(value[:, :, 0])  #             return value[:,:,0] |> L
    if not _coconut_case_check_0:  # an RGB image  #         match (_,_,3): # an RGB image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut_match_to[2] == 3):  # an RGB image  #         match (_,_,3): # an RGB image
            _coconut_case_check_0 = True  # an RGB image  #         match (_,_,3): # an RGB image
        if _coconut_case_check_0:  # an RGB image  #         match (_,_,3): # an RGB image
            return (RGB)(value)  #             return value |> RGB
    if not _coconut_case_check_0:  # an RGBA image  #         match (_,_,4): # an RGBA image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut_match_to[2] == 4):  # an RGBA image  #         match (_,_,4): # an RGBA image
            _coconut_case_check_0 = True  # an RGBA image  #         match (_,_,4): # an RGBA image
        if _coconut_case_check_0:  # an RGBA image  #         match (_,_,4): # an RGBA image
            return (RGBA)(value)  #             return value |> RGBA
    if not _coconut_case_check_0:  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3):  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
            bs = _coconut_match_to[0]  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
            h = _coconut_match_to[1]  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
            w = _coconut_match_to[2]  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
            _coconut_case_check_0 = True  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
        if _coconut_case_check_0 and not (w > 4):  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
            _coconut_case_check_0 = False  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
        if _coconut_case_check_0:  # batch of gray image  #         match (bs,h,w) if w>4: # batch of gray image
            return (L)(value.transpose(1, 0, 2).reshape(h, bs * w))  #             return value.transpose(1,0,2).reshape(h,bs*w) |> L
    if not _coconut_case_check_0:  # gray image  #         match (_,_): # gray image
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2):  # gray image  #         match (_,_): # gray image
            _coconut_case_check_0 = True  # gray image  #         match (_,_): # gray image
        if _coconut_case_check_0:  # gray image  #         match (_,_): # gray image
            return (L)(value)  #             return value |> L
    if not _coconut_case_check_0:  #     else:
        return None  #         return None

def ary_stat_widget(ary):  # def ary_stat_widget(ary):
    return widgets.VBox([widgets.Label(value=pformat(str(dict(shape=ary.shape, dtype=ary.dtype)))), widgets.HTML(pd.DataFrame(pd.Series(ary.ravel()).describe()).transpose()._repr_html_())])  #     return widgets.VBox([


def ary_to_widget(ary):  # def ary_to_widget(ary):
    stat_widget = ary_stat_widget(ary)  #     stat_widget = ary_stat_widget(ary)
    viz_widget = (nn)(ary_to_image(ary), img_to_widget)  #     viz_widget = ary_to_image(ary) `nn` img_to_widget
    viz_widget = widgets.Label(value=str(ary)) if viz_widget is None else viz_widget  #     viz_widget ??= widgets.Label(value=str(ary))

    children = [viz_widget, (bqplot_hist)(ary), stat_widget]  #     children = [
    return widgets.VBox(children)  #     return widgets.VBox(children)

def output_widget(value):  # def output_widget(value):
    out = widgets.Output()  #     out = widgets.Output()
    with out:  #     with out:
        pprint(repr(value))  #         pprint(repr(value))
    return out  #     return out

def infer_widget(value):  # def infer_widget(value):
    while True:  #     from data_tree._series import Series
        from data_tree._series import Series  #     from data_tree._series import Series
        _coconut_match_to = value  #     case value:
        _coconut_case_check_1 = False  #     case value:
        if _coconut.isinstance(_coconut_match_to, Series):  #     case value:
            _coconut_case_check_1 = True  #     case value:
        if _coconut_case_check_1:  #     case value:
            return value.widget()  #             return value.widget()
        if not _coconut_case_check_1:  #         match _ is PIL.Image.Image:
            if _coconut.isinstance(_coconut_match_to, PIL.Image.Image):  #         match _ is PIL.Image.Image:
                _coconut_case_check_1 = True  #         match _ is PIL.Image.Image:
            if _coconut_case_check_1:  #         match _ is PIL.Image.Image:
                return (img_to_widget)(value)  #             return value |> img_to_widget
        if not _coconut_case_check_1:  #         match _ is TensorType:
            if _coconut.isinstance(_coconut_match_to, TensorType):  #         match _ is TensorType:
                _coconut_case_check_1 = True  #         match _ is TensorType:
            if _coconut_case_check_1:  #         match _ is TensorType:
                try:  #         match _ is TensorType:
                    _coconut_is_recursive = infer_widget is _coconut_recursive_func_5  #         match _ is TensorType:
                except _coconut.NameError:  #         match _ is TensorType:
                    _coconut_is_recursive = False  #         match _ is TensorType:
                if _coconut_is_recursive:  #         match _ is TensorType:
                    value = value.detach().numpy()  #         match _ is TensorType:
                    continue  #         match _ is TensorType:
                else:  #         match _ is TensorType:
                    return infer_widget(value.detach().numpy())  #         match _ is TensorType:

        if not _coconut_case_check_1:  #         match _ is np.ndarray:
            if _coconut.isinstance(_coconut_match_to, np.ndarray):  #         match _ is np.ndarray:
                _coconut_case_check_1 = True  #         match _ is np.ndarray:
            if _coconut_case_check_1:  #         match _ is np.ndarray:
                return ary_to_widget(value)  #             return ary_to_widget(value)
        if not _coconut_case_check_1:  #         match _ is tuple if value `hasattr` "_asdict":
            if _coconut.isinstance(_coconut_match_to, tuple):  #         match _ is tuple if value `hasattr` "_asdict":
                _coconut_case_check_1 = True  #         match _ is tuple if value `hasattr` "_asdict":
            if _coconut_case_check_1 and not ((hasattr)(value, "_asdict")):  #         match _ is tuple if value `hasattr` "_asdict":
                _coconut_case_check_1 = False  #         match _ is tuple if value `hasattr` "_asdict":
            if _coconut_case_check_1:  #         match _ is tuple if value `hasattr` "_asdict":
                return (output_widget)(value)  #             return value |> output_widget
        if not _coconut_case_check_1:  #         match _ is (tuple,list):
            if _coconut.isinstance(_coconut_match_to, (tuple, list)):  #         match _ is (tuple,list):
                _coconut_case_check_1 = True  #         match _ is (tuple,list):
            if _coconut_case_check_1:  #         match _ is (tuple,list):
                items = [infer_widget(item) for item in value]  #             items = [infer_widget(item) for item in value]
                return widgets.VBox([widgets.GridBox(items, layout=widgets.Layout(grid_template_columns="auto auto auto", border="solid 2px")), widgets.Label(value="displaying tuple with {_coconut_format_0} elements".format(_coconut_format_0=(len(value))))])  #             return widgets.VBox([
        if not _coconut_case_check_1:  #     else:
            return (output_widget)(value)  #         return value |> output_widget


        return None  # 
_coconut_recursive_func_5 = infer_widget  #
