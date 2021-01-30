#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x6d9f3c11

# Compiled with Coconut version 1.4.3 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get("__coconut__")
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules["__coconut__"]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

from data_tree.coconut.convert import *  # from data_tree.coconut.convert import *
from data_tree.coconut.auto_data import AutoSolver  # from data_tree.coconut.auto_data import AutoSolver,AutoData
from data_tree.coconut.auto_data import AutoData  # from data_tree.coconut.auto_data import AutoSolver,AutoData
from data_tree.coconut.monad import try_monad  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
from data_tree.coconut.monad import Try  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
from data_tree.coconut.monad import Success  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
from data_tree.coconut.monad import Failure  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
from frozendict import frozendict  # from frozendict import frozendict
from typing import Mapping  # from typing import Mapping
from ipywidgets import Text  # from ipywidgets import Text
import ipywidgets as widgets  # import ipywidgets as widgets
from itertools import product  # from itertools import product
from loguru import logger  # from loguru import logger
from collections import namedtuple  # from collections import namedtuple
import numpy as np  # import numpy as np

def imagedef2dict(imdef: 'ImageDef'):  # def imagedef2dict(imdef:ImageDef):
    _coconut_match_to = imdef  #     case imdef:
    _coconut_case_check_1 = False  #     case imdef:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2):  #     case imdef:
        data_type = _coconut_match_to[0]  #     case imdef:
        tags = _coconut_match_to[1]  #     case imdef:
        _coconut_case_check_1 = True  #     case imdef:
    if _coconut_case_check_1:  #     case imdef:
        _coconut_match_to = data_type  #     case imdef:
        _coconut_case_check_0 = False  #     case imdef:
        if (_coconut.isinstance(_coconut_match_to, Numpy)) and (_coconut.len(_coconut_match_to) == 4):  #     case imdef:
            dtype = _coconut_match_to[0]  #     case imdef:
            arrange = _coconut_match_to[1]  #     case imdef:
            ch_rpr = _coconut_match_to[2]  #     case imdef:
            v_range = _coconut_match_to[3]  #     case imdef:
            _coconut_case_check_0 = True  #     case imdef:
        if _coconut_case_check_0:  #     case imdef:
            info = dict(type="numpy", dtype=dtype, arrange=arrange, ch_rpr=ch_rpr, v_range=str(v_range))  #                     info = dict(type="numpy",dtype=dtype,arrange=arrange,ch_rpr=ch_rpr,v_range=str(v_range))
        if not _coconut_case_check_0:  #                 match Torch(dtype,arrange,ch_rpr,v_range):
            if (_coconut.isinstance(_coconut_match_to, Torch)) and (_coconut.len(_coconut_match_to) == 4):  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                dtype = _coconut_match_to[0]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                arrange = _coconut_match_to[1]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                ch_rpr = _coconut_match_to[2]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                v_range = _coconut_match_to[3]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                _coconut_case_check_0 = True  #                 match Torch(dtype,arrange,ch_rpr,v_range):
            if _coconut_case_check_0:  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                info = dict(type="torch", dtype=dtype, arrange=arrange, ch_rpr=ch_rpr, v_range=str(v_range))  #                     info = dict(type="torch",dtype=dtype,arrange=arrange,ch_rpr=ch_rpr,v_range=str(v_range))
        if not _coconut_case_check_0:  #                 match PILImages(mode,ch_rpr):
            if (_coconut.isinstance(_coconut_match_to, PILImages)) and (_coconut.len(_coconut_match_to) == 2):  #                 match PILImages(mode,ch_rpr):
                mode = _coconut_match_to[0]  #                 match PILImages(mode,ch_rpr):
                ch_rpr = _coconut_match_to[1]  #                 match PILImages(mode,ch_rpr):
                _coconut_case_check_0 = True  #                 match PILImages(mode,ch_rpr):
            if _coconut_case_check_0:  #                 match PILImages(mode,ch_rpr):
                info = dict(type="images", ch_rpr=ch_rpr, mode=mode)  #                     info = dict(type="images",ch_rpr=ch_rpr,mode=mode)
        if not _coconut_case_check_0:  #                 match PILImage(mode,ch_rpr):
            if (_coconut.isinstance(_coconut_match_to, PILImage)) and (_coconut.len(_coconut_match_to) == 2):  #                 match PILImage(mode,ch_rpr):
                mode = _coconut_match_to[0]  #                 match PILImage(mode,ch_rpr):
                ch_rpr = _coconut_match_to[1]  #                 match PILImage(mode,ch_rpr):
                _coconut_case_check_0 = True  #                 match PILImage(mode,ch_rpr):
            if _coconut_case_check_0:  #                 match PILImage(mode,ch_rpr):
                info = dict(type="image", ch_rpr=ch_rpr, mode=mode)  #                     info = dict(type="image",ch_rpr=ch_rpr,mode=mode)
        if not _coconut_case_check_0:  #             else:
            raise RuntimeError("cannot convert unknown imagedef:{_coconut_format_0} to dict.".format(_coconut_format_0=(imdef)))  #                 raise RuntimeError(f"cannot convert unknown imagedef:{imdef} to dict.")
        return frozendict(**info, **{t: True for t in tags})  #             return frozendict(
    if not _coconut_case_check_1:  #     else:
        raise RuntimeError("cannot convert unknown imdef:{_coconut_format_0} to dict.".format(_coconut_format_0=(imdef)))  #         raise RuntimeError(f"cannot convert unknown imdef:{imdef} to dict.")

def cast_imdef_to_dict(state):  # def cast_imdef_to_dict(state):
    if isinstance(state, ImageDef):  #     if isinstance(state,ImageDef):
        return [imagedef2dict(state)]  #         return [imagedef2dict(state)]

def cast_imdef_str_to_imdef(state):  # def cast_imdef_str_to_imdef(state):
    if isinstance(state, str):  #     if isinstance(state,str):
        try:  #         try:
            res = str_to_img_def(state)  #             res = str_to_img_def(state)
            return [res]  #             return [res]
        except Exception as e:  #         except Exception as e:
            pass  #             pass
def imdef2imdef_str(imdef):  # def imdef2imdef_str(imdef):
    _coconut_match_to = imdef  #     case imdef:
    _coconut_case_check_3 = False  #     case imdef:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2):  #     case imdef:
        data_type = _coconut_match_to[0]  #     case imdef:
        tags = _coconut_match_to[1]  #     case imdef:
        _coconut_case_check_3 = True  #     case imdef:
    if _coconut_case_check_3:  #     case imdef:
        _coconut_match_to = data_type  #     case imdef:
        _coconut_case_check_2 = False  #     case imdef:
        if (_coconut.isinstance(_coconut_match_to, Numpy)) and (_coconut.len(_coconut_match_to) == 4):  #     case imdef:
            dtype = _coconut_match_to[0]  #     case imdef:
            arrange = _coconut_match_to[1]  #     case imdef:
            ch_rpr = _coconut_match_to[2]  #     case imdef:
            v_range = _coconut_match_to[3]  #     case imdef:
            _coconut_case_check_2 = True  #     case imdef:
        if _coconut_case_check_2:  #     case imdef:
            base = "numpy,{_coconut_format_0},{_coconut_format_1},{_coconut_format_2},{_coconut_format_3}".format(_coconut_format_0=(dtype), _coconut_format_1=(arrange), _coconut_format_2=(ch_rpr), _coconut_format_3=(v_range))  #                     base = f"numpy,{dtype},{arrange},{ch_rpr},{v_range}"
        if not _coconut_case_check_2:  #                 match Torch(dtype,arrange,ch_rpr,v_range):
            if (_coconut.isinstance(_coconut_match_to, Torch)) and (_coconut.len(_coconut_match_to) == 4):  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                dtype = _coconut_match_to[0]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                arrange = _coconut_match_to[1]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                ch_rpr = _coconut_match_to[2]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                v_range = _coconut_match_to[3]  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                _coconut_case_check_2 = True  #                 match Torch(dtype,arrange,ch_rpr,v_range):
            if _coconut_case_check_2:  #                 match Torch(dtype,arrange,ch_rpr,v_range):
                base = "torch,{_coconut_format_0},{_coconut_format_1},{_coconut_format_2},{_coconut_format_3}".format(_coconut_format_0=(dtype), _coconut_format_1=(arrange), _coconut_format_2=(ch_rpr), _coconut_format_3=(v_range))  #                     base = f"torch,{dtype},{arrange},{ch_rpr},{v_range}"
        if not _coconut_case_check_2:  #                 match PILImages(mode,ch_rpr):
            if (_coconut.isinstance(_coconut_match_to, PILImages)) and (_coconut.len(_coconut_match_to) == 2):  #                 match PILImages(mode,ch_rpr):
                mode = _coconut_match_to[0]  #                 match PILImages(mode,ch_rpr):
                ch_rpr = _coconut_match_to[1]  #                 match PILImages(mode,ch_rpr):
                _coconut_case_check_2 = True  #                 match PILImages(mode,ch_rpr):
            if _coconut_case_check_2:  #                 match PILImages(mode,ch_rpr):
                base = "images,{_coconut_format_0},{_coconut_format_1}".format(_coconut_format_0=(mode), _coconut_format_1=(ch_rpr))  #                     base = f"images,{mode},{ch_rpr}"
        if not _coconut_case_check_2:  #                 match PILImage(mode,ch_rpr):
            if (_coconut.isinstance(_coconut_match_to, PILImage)) and (_coconut.len(_coconut_match_to) == 2):  #                 match PILImage(mode,ch_rpr):
                mode = _coconut_match_to[0]  #                 match PILImage(mode,ch_rpr):
                ch_rpr = _coconut_match_to[1]  #                 match PILImage(mode,ch_rpr):
                _coconut_case_check_2 = True  #                 match PILImage(mode,ch_rpr):
            if _coconut_case_check_2:  #                 match PILImage(mode,ch_rpr):
                base = "image,{_coconut_format_0},{_coconut_format_1}".format(_coconut_format_0=(mode), _coconut_format_1=(ch_rpr))  #                     base = f"image,{mode},{ch_rpr}"
        if not _coconut_case_check_2:  #             else:
            raise RuntimeError("cannot convert unknown imagedef:{_coconut_format_0} to str.".format(_coconut_format_0=(imdef)))  #                 raise RuntimeError(f"cannot convert unknown imagedef:{imdef} to str.")
        if tags:  #             if tags:
            return base + "|{_coconut_format_0}".format(_coconut_format_0=(','.join(tags)))  #                 return base+f"|{','.join(tags)}"
        else:  #             else:
            return base  #                 return base
    if not _coconut_case_check_3:  #     else:
        raise RuntimeError("cannot convert unknown imdef:{_coconut_format_0} to str.".format(_coconut_format_0=(imdef)))  #         raise RuntimeError(f"cannot convert unknown imdef:{imdef} to str.")
def cast_imdef_to_imdef_str(imdef):  # def cast_imdef_to_imdef_str(imdef):
    _coconut_match_to = imdef  #     case imdef:
    _coconut_case_check_4 = False  #     case imdef:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2):  #     case imdef:
        _coconut_case_check_4 = True  #     case imdef:
    if _coconut_case_check_4:  #     case imdef:
        res = [imdef2imdef_str(imdef)]  #             res = [imdef2imdef_str(imdef)]
        return res  #             return res
    if not _coconut_case_check_4:  #     else:
        return None  #         return None

def imgs2tile(imgs, w=1024, h=1024, max_image=100, padding=1):  # def imgs2tile(imgs,w=1024,h=1024,max_image=100,padding=1):
    mode = imgs[0].mode  #     mode = imgs[0].mode
    ch = len(mode)  #     ch = len(mode)
#nrow = int(sqrt(len(imgs[:max_image]))+0.5)
    n_imgs = len(imgs[:max_image])  #     n_imgs = len(imgs[:max_image])
    nrow = int(sqrt(n_imgs))  #     nrow = int(sqrt(n_imgs))
    if (nrow * nrow < n_imgs):  #     if (nrow*nrow < n_imgs):
        nrow += 1  #         nrow += 1
    r = int((w - ((nrow + 1) * padding)) / nrow)  #     r = int((w-((nrow+1)*padding))/nrow)

    imgs = np.array([((np.array)(img.resize((r, r)))) for img in imgs[:max_image]])  #     imgs = np.array([(img.resize((r,r)) |> np.array) for img in imgs[:max_image]])
    if ch == 1:  #     if ch == 1:
        imgs = imgs[:, :, :, None]  #         imgs = imgs[:,:,:,None]
    return make_grid(imgs, nrow, padding=padding)  #     return make_grid(imgs,nrow,padding=padding)

def rule_imgs2tile(state):  # def rule_imgs2tile(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_5 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], PILImages)) and (_coconut.len(_coconut_match_to[0]) == 2):  #     case state:
        mode = _coconut_match_to[0][0]  #     case state:
        chrpr = _coconut_match_to[0][1]  #     case state:
        tags = _coconut_match_to[1]  #     case state:
        _coconut_case_check_5 = True  #     case state:
    if _coconut_case_check_5:  #     case state:
        return [(imgs2tile, ImageDef(Numpy("uint8", "HWC", chrpr, VR_0_255), tags), "imgs2tile", 10)]  #             return [(


def rule_img2widget(state):  # def rule_img2widget(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_6 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], PILImage)) and (_coconut.len(_coconut_match_to[0]) == 2):  #     case state:
        tags = _coconut_match_to[1]  #     case state:
        _coconut_case_check_6 = True  #     case state:
    if _coconut_case_check_6:  #     case state:
        return [(infer_widget, "widget", "infer_widget", 1)]  #             return [(

def dict2imdef(state):  # def dict2imdef(state):
    if isinstance(state, Mapping):  #     if isinstance(state,Mapping):
        _coconut_match_to = state  #         case state:
        _coconut_case_check_7 = False  #         case state:
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #         case state:
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #         case state:
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #         case state:
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #         case state:
            _coconut_match_temp_3 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #         case state:
            _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #         case state:
            if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_4 is not _coconut_sentinel):  #         case state:
                _dtype = _coconut_match_temp_1  #         case state:
                _arng = _coconut_match_temp_2  #         case state:
                _ch_rpr = _coconut_match_temp_3  #         case state:
                _v_range = _coconut_match_temp_4  #         case state:
                tags = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "ch_rpr", "v_range")))  #         case state:
                _coconut_case_check_7 = True  #         case state:
        if _coconut_case_check_7:  #         case state:
            return [ImageDef(Numpy(_dtype, _arng, _ch_rpr, _v_range), frozenset(tags.keys()))]  #                 return [ImageDef(Numpy(_dtype,_arng,_ch_rpr,_v_range),frozenset(tags.keys()))]
        if not _coconut_case_check_7:  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
            if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                _coconut_match_temp_3 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_4 is not _coconut_sentinel):  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                    _dtype = _coconut_match_temp_1  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                    _arng = _coconut_match_temp_2  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                    _ch_rpr = _coconut_match_temp_3  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                    _v_range = _coconut_match_temp_4  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                    tags = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "ch_rpr", "v_range")))  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                    _coconut_case_check_7 = True  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
            if _coconut_case_check_7:  #             match {"type":"torch","dtype":_dtype,"arrange":_arng,"ch_rpr":_ch_rpr,"v_range":_v_range,**tags}:
                return [ImageDef(Torch(_dtype, _arng, _ch_rpr, _v_range), frozenset(tags.keys()))]  #                 return [ImageDef(Torch(_dtype,_arng,_ch_rpr,_v_range),frozenset(tags.keys()))]
        if not _coconut_case_check_7:  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
            if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                _coconut_match_temp_1 = _coconut_match_to.get("mode", _coconut_sentinel)  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                _coconut_match_temp_2 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "image") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_2 is not _coconut_sentinel):  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    _mode = _coconut_match_temp_1  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    _ch_rpr = _coconut_match_temp_2  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    tags = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "mode", "ch_rpr")))  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    _coconut_case_check_7 = True  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
            if _coconut_case_check_7:  #             match {"type":"image","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                return [ImageDef(PILImage(_mode, _ch_rpr), frozenset(tags.keys()))]  #                 return [ImageDef(PILImage(_mode,_ch_rpr),frozenset(tags.keys()))]
        if not _coconut_case_check_7:  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
            if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                _coconut_match_temp_1 = _coconut_match_to.get("mode", _coconut_sentinel)  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                _coconut_match_temp_2 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "images") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_2 is not _coconut_sentinel):  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    _mode = _coconut_match_temp_1  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    _ch_rpr = _coconut_match_temp_2  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    tags = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "mode", "ch_rpr")))  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                    _coconut_case_check_7 = True  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
            if _coconut_case_check_7:  #             match {"type":"images","mode":_mode,"ch_rpr":_ch_rpr,**tags}:
                return [ImageDef(PILImages(_mode, _ch_rpr), frozenset(tags.keys()))]  #                 return [ImageDef(PILImages(_mode,_ch_rpr),frozenset(tags.keys()))]

def rule_numpy2img(state):  # def rule_numpy2img(state):
    if isinstance(state, Mapping):  #     if isinstance(state,Mapping):
        _coconut_match_to = state  #         case state:
        _coconut_case_check_8 = False  #         case state:
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #         case state:
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #         case state:
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #         case state:
            _coconut_match_temp_2 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #         case state:
            _coconut_match_temp_3 = _coconut_match_to.get("arrange", _coconut_sentinel)  #         case state:
            _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #         case state:
            if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "uint8") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "RGB") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "HWC") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "0_255"):  #         case state:
                tags = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "ch_rpr", "arrange", "v_range")))  #         case state:
                _coconut_case_check_8 = True  #         case state:
        if _coconut_case_check_8:  #         case state:
            return [(Image.fromarray, ImageDef(PILImage("RGB", "RGB"), frozenset(tags.keys())), "Image.fromarray", 1)]  #                 return [(
        if not _coconut_case_check_8:  #                     Image.fromarray,
            if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                     Image.fromarray,
                _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                     Image.fromarray,
                _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                     Image.fromarray,
                _coconut_match_temp_2 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                     Image.fromarray,
                _coconut_match_temp_3 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                     Image.fromarray,
                _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                     Image.fromarray,
                if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "uint8") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "L") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "HW") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "0_255"):  #                     Image.fromarray,
                    tags = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "ch_rpr", "arrange", "v_range")))  #                     Image.fromarray,
                    _coconut_case_check_8 = True  #                     Image.fromarray,
            if _coconut_case_check_8:  #                     Image.fromarray,
                return [(Image.fromarray, ImageDef(PILImage("L", "L"), frozenset(tags.keys())), "Image.fromarray", 1)]  #                 return [(

def rule_image2gray(state):  # def rule_image2gray(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_9 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], PILImage)) and (_coconut.len(_coconut_match_to[0]) == 2):  #     case state:
        ch_rpr = _coconut_match_to[0][0]  #     case state:
        ch_rpr2 = _coconut_match_to[0][1]  #     case state:
        tags = _coconut_match_to[1]  #     case state:
        _coconut_case_check_9 = True  #     case state:
    if _coconut_case_check_9:  #     case state:
        return [(_coconut.operator.methodcaller("convert", "L"), ImageDef(PILImage("L", "L"), tags), "image2gray", 10), (_coconut.operator.methodcaller("convert", "LA"), ImageDef(PILImage("LA", "LA"), tags), "image2gray-alpha", 10),]  #             return [

def rule_image2lab(state):  # def rule_image2lab(state):
    from skimage import color  #     from skimage import color
    _coconut_match_to = state  #     case state:
    _coconut_case_check_10 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "RGB") and (_coconut_match_to[0][3] == "0_1"):  #     case state:
        tags = _coconut_match_to[1]  #     case state:
        _coconut_case_check_10 = True  #     case state:
    if _coconut_case_check_10:  #     case state:
        return [(color.rgb2lab, ImageDef(Numpy("float64", "HWC", "LAB", "LAB"), tags), "sklearn.color.rgb2lab")]  #             return [
    if not _coconut_case_check_10:  #                 (color.rgb2lab,ImageDef(Numpy("float64","HWC","LAB","LAB"),tags),"sklearn.color.rgb2lab")
        if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "LAB") and (_coconut_match_to[0][3] == "LAB"):  #                 (color.rgb2lab,ImageDef(Numpy("float64","HWC","LAB","LAB"),tags),"sklearn.color.rgb2lab")
            tags = _coconut_match_to[1]  #                 (color.rgb2lab,ImageDef(Numpy("float64","HWC","LAB","LAB"),tags),"sklearn.color.rgb2lab")
            _coconut_case_check_10 = True  #                 (color.rgb2lab,ImageDef(Numpy("float64","HWC","LAB","LAB"),tags),"sklearn.color.rgb2lab")
        if _coconut_case_check_10:  #                 (color.rgb2lab,ImageDef(Numpy("float64","HWC","LAB","LAB"),tags),"sklearn.color.rgb2lab")
            return [(color.lab2rgb, ImageDef(Numpy("float64", "HWC", "RGB", "0_1"), tags), "sklearn.color.lab2rgb")]  #             return [

def convert_ignore_channel(ary, f):  # def convert_ignore_channel(ary,f):
    """
    shape:(H,W,C)
    """  #     """
    ignored = ary[:, :, [-1]]  #     ignored = ary[:,:,[-1]]
    tgt = ary[:, :, :-1]  #     tgt = ary[:,:,:-1]
    converted = f(tgt)  #     converted = f(tgt)
    result = np.concatenate((tgt, ignored), axis=-1)  #     result = np.concatenate((tgt,ignored),axis=-1)
    return result  #     return result

def rule_rgba2laba(state):  # def rule_rgba2laba(state):
    from skimage import color  #     from skimage import color
    _coconut_match_to = state  #     case state:
    _coconut_case_check_11 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "RGBA") and (_coconut_match_to[0][3] == "0_1"):  #     case state:
        tags = _coconut_match_to[1]  #     case state:
        _coconut_case_check_11 = True  #     case state:
    if _coconut_case_check_11:  #     case state:
        return [(lambda a: convert_ignore_channel(a, color.rgb2lab), ImageDef(Numpy("float64", "HWC", "LABA", "LABA"), tags), "rgba2laba (ignores alpha)")]  #             return [
    if not _coconut_case_check_11:  #                 (a->convert_ignore_channel(a,color.rgb2lab),ImageDef(Numpy("float64","HWC","LABA","LABA"),tags),"rgba2laba (ignores alpha)")
        if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "LABA") and (_coconut_match_to[0][3] == "LABA"):  #                 (a->convert_ignore_channel(a,color.rgb2lab),ImageDef(Numpy("float64","HWC","LABA","LABA"),tags),"rgba2laba (ignores alpha)")
            tags = _coconut_match_to[1]  #                 (a->convert_ignore_channel(a,color.rgb2lab),ImageDef(Numpy("float64","HWC","LABA","LABA"),tags),"rgba2laba (ignores alpha)")
            _coconut_case_check_11 = True  #                 (a->convert_ignore_channel(a,color.rgb2lab),ImageDef(Numpy("float64","HWC","LABA","LABA"),tags),"rgba2laba (ignores alpha)")
        if _coconut_case_check_11:  #                 (a->convert_ignore_channel(a,color.rgb2lab),ImageDef(Numpy("float64","HWC","LABA","LABA"),tags),"rgba2laba (ignores alpha)")
            return [(lambda a: convert_ignore_channel(a, color.lab2rgb), ImageDef(Numpy("float64", "HWC", "RGBA", "0_1"), tags), "laba2rgba (ignores alpha)")]  #             return [


def rule_lab_value_conversion(state):  # def rule_lab_value_conversion(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_12 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "LAB") and (_coconut_match_to[0][3] == "LAB"):  #     case state:
        tags = _coconut_match_to[1]  #     case state:
        _coconut_case_check_12 = True  #     case state:
    if _coconut_case_check_12:  #     case state:
        return [((_vr_lab_to_0_1, ImageDef(Numpy("float64", "HWC", "LAB", "0_1"), tags), "vr_lab_to_0_1"))]  #             return [((_vr_lab_to_0_1,ImageDef(Numpy("float64","HWC","LAB","0_1"),tags),"vr_lab_to_0_1"))]
    if not _coconut_case_check_12:  #         match ImageDef(Numpy("float64","HWC","LABA","LABA"),tags):
        if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "LABA") and (_coconut_match_to[0][3] == "LABA"):  #         match ImageDef(Numpy("float64","HWC","LABA","LABA"),tags):
            tags = _coconut_match_to[1]  #         match ImageDef(Numpy("float64","HWC","LABA","LABA"),tags):
            _coconut_case_check_12 = True  #         match ImageDef(Numpy("float64","HWC","LABA","LABA"),tags):
        if _coconut_case_check_12:  #         match ImageDef(Numpy("float64","HWC","LABA","LABA"),tags):
            return [((lambda a: convert_ignore_channel(a, _vr_lab_to_0_1), ImageDef(Numpy("float64", "HWC", "LABA", "0_1"), tags), "vr_laba_to_0_1"))]  #             return [((a->convert_ignore_channel(a,_vr_lab_to_0_1),ImageDef(Numpy("float64","HWC","LABA","0_1"),tags),"vr_laba_to_0_1"))]
    if not _coconut_case_check_12:  #         match ImageDef(Numpy("float64","HWC","LAB","0_1"),tags):
        if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "LAB") and (_coconut_match_to[0][3] == "0_1"):  #         match ImageDef(Numpy("float64","HWC","LAB","0_1"),tags):
            tags = _coconut_match_to[1]  #         match ImageDef(Numpy("float64","HWC","LAB","0_1"),tags):
            _coconut_case_check_12 = True  #         match ImageDef(Numpy("float64","HWC","LAB","0_1"),tags):
        if _coconut_case_check_12:  #         match ImageDef(Numpy("float64","HWC","LAB","0_1"),tags):
            return [((_0_1_to_vr_lab, ImageDef(Numpy("float64", "HWC", "LAB", "LAB"), tags), "0_1_to_vr_lab"))]  #             return [((_0_1_to_vr_lab,ImageDef(Numpy("float64","HWC","LAB","LAB"),tags),"0_1_to_vr_lab"))]
    if not _coconut_case_check_12:  #         match ImageDef(Numpy("float64","HWC","LABA","0_1"),tags):
        if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], Numpy)) and (_coconut.len(_coconut_match_to[0]) == 4) and (_coconut_match_to[0][0] == "float64") and (_coconut_match_to[0][1] == "HWC") and (_coconut_match_to[0][2] == "LABA") and (_coconut_match_to[0][3] == "0_1"):  #         match ImageDef(Numpy("float64","HWC","LABA","0_1"),tags):
            tags = _coconut_match_to[1]  #         match ImageDef(Numpy("float64","HWC","LABA","0_1"),tags):
            _coconut_case_check_12 = True  #         match ImageDef(Numpy("float64","HWC","LABA","0_1"),tags):
        if _coconut_case_check_12:  #         match ImageDef(Numpy("float64","HWC","LABA","0_1"),tags):
            return [((lambda a: convert_ignore_channel(a, _0_1_to_vr_lab), ImageDef(Numpy("float64", "HWC", "LABA", "LABA"), tags), "vr_0_1_to_laba"))]  #             return [((a->convert_ignore_channel(a,_0_1_to_vr_lab),ImageDef(Numpy("float64","HWC","LABA","LABA"),tags),"vr_0_1_to_laba"))]

def _vr_lab_to_0_1(ary):  # def _vr_lab_to_0_1(ary):
    r = ary.copy()  #     r = ary.copy()
    r[:, :, 0] = ary[:, :, 0] * 0.01  #     r[:,:,0] = ary[:,:,0] * 0.01
    r[:, :, 1] = (ary[:, :, 1] + 128.0) / 255.0  #     r[:,:,1] = (ary[:,:,1] + 128.0) / 255.0
    r[:, :, 2] = (ary[:, :, 2] + 128.0) / 255.0  #     r[:,:,2] = (ary[:,:,2] + 128.0) / 255.0
    return r  #     return r

def _0_1_to_vr_lab(ary):  # def _0_1_to_vr_lab(ary):
    r = ary.copy()  #     r = ary.copy()
    r[:, :, 0] = ary[:, :, 0] * 100  #     r[:,:,0] = ary[:,:,0] * 100
    r[:, :, 1] = (ary[:, :, 1] * 255) - 128.0  #     r[:,:,1] = (ary[:,:,1] * 255) - 128.0
    r[:, :, 2] = (ary[:, :, 2] * 255) - 128.0  #     r[:,:,2] = (ary[:,:,2] * 255) - 128.0
    return r  #     return r

"""
def dict2visdomable(state):
    case state:
        match {"type":"numpy","dtype":"float32","arrange":"CHW" or "BCHW","ch_rpr":"RGB" or "L",**others} if "visdomable" not in state:
            return [frozendict(
                **state,
                visdomable=True
            )]
"""  # """
def to_visdom_function(state):  # def to_visdom_function(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_13 = False  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #     case state:
        _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #     case state:
        _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #     case state:
        _coconut_match_temp_3 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #     case state:
        _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #     case state:
        if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "CHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "RGB") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "0_255"):  #     case state:
            others = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "ch_rpr", "v_range")))  #     case state:
            _coconut_case_check_13 = True  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #     case state:
        _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #     case state:
        _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #     case state:
        _coconut_match_temp_3 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #     case state:
        if (not _coconut_case_check_13) and (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "CHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "L") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "0_255"):  #     case state:
            _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #     case state:
            others = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "ch_rpr", "v_range")))  #     case state:
            _coconut_case_check_13 = True  #     case state:
    if _coconut_case_check_13:  #     case state:
        return [(lambda ary: lambda visdom: _coconut.functools.partial(visdom.image, ary), "visdom_function", "to_visdom_function")]  #             return [
    if not _coconut_case_check_13:  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_3 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "BCHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "RGB") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "0_255"):  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
                others = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "ch_rpr", "v_range")))  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
                _coconut_case_check_13 = True  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            _coconut_match_temp_3 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            if (not _coconut_case_check_13) and (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "BCHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "L") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "0_255"):  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
                _coconut_match_temp_4 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
                others = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "ch_rpr", "v_range")))  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
                _coconut_case_check_13 = True  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
        if _coconut_case_check_13:  #                 (ary->visdom->visdom.image$(ary),"visdom_function","to_visdom_function")
            return [(lambda ary: lambda visdom: _coconut.functools.partial(visdom.images, ary), "visdom_function", "to_visdom_function")]  #             return [
def any2widget(state):  # def any2widget(state):
    return [(lambda ary: Text(str(ary)), "widget", "anything_to_text_widget", 1000)]  #     return [(ary->Text(str(ary)),"widget","anything_to_text_widget",1000)]
class AutoTuple(_coconut.collections.namedtuple("AutoTuple", "formats")):  # data AutoTuple(formats is tuple)
    __slots__ = ()  # data AutoTuple(formats is tuple)
    __ne__ = _coconut.object.__ne__  # data AutoTuple(formats is tuple)
    def __eq__(self, other):  # data AutoTuple(formats is tuple)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data AutoTuple(formats is tuple)
    def __hash__(self):  # data AutoTuple(formats is tuple)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data AutoTuple(formats is tuple)
    def __new__(_cls, *_coconut_match_to_args, **_coconut_match_to_kwargs):  # data AutoTuple(formats is tuple)
        _coconut_match_check = False  # data AutoTuple(formats is tuple)
        _coconut_FunctionMatchError = _coconut_get_function_match_error()  # data AutoTuple(formats is tuple)
        if (_coconut.len(_coconut_match_to_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "formats" in _coconut_match_to_kwargs)) == 1):  # data AutoTuple(formats is tuple)
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("formats")  # data AutoTuple(formats is tuple)
            if (_coconut.isinstance(_coconut_match_temp_0, tuple)) and (not _coconut_match_to_kwargs):  # data AutoTuple(formats is tuple)
                formats = _coconut_match_temp_0  # data AutoTuple(formats is tuple)
                _coconut_match_check = True  # data AutoTuple(formats is tuple)

        if not _coconut_match_check:  # data AutoTuple(formats is tuple)
            _coconut_match_val_repr = _coconut.repr(_coconut_match_to_args)  # data AutoTuple(formats is tuple)
            _coconut_match_err = _coconut_FunctionMatchError("pattern-matching failed for " "'data AutoTuple(formats is tuple)'" " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))  # data AutoTuple(formats is tuple)
            _coconut_match_err.pattern = 'data AutoTuple(formats is tuple)'  # data AutoTuple(formats is tuple)
            _coconut_match_err.value = _coconut_match_to_args  # data AutoTuple(formats is tuple)
            raise _coconut_match_err  # data AutoTuple(formats is tuple)

        return _coconut.tuple.__new__(_cls, (formats,))  # data AutoTuple(formats is tuple)


def auto_tuple2widget(state):  # def auto_tuple2widget(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_14 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, AutoTuple)) and (_coconut.len(_coconut_match_to) == 1):  #     case state:
        items = _coconut_match_to[0]  #     case state:
        _coconut_case_check_14 = True  #     case state:
    if _coconut_case_check_14 and not (all((i == "widget" for i in items))):  #     case state:
        _coconut_case_check_14 = False  #     case state:
    if _coconut_case_check_14:  #     case state:
        return [(lambda values: widgets.VBox(values), "widget", "auto_tuple of widgets to a widget", 1)]  #             return [

def isnamedtuple(x):  # def isnamedtuple(x):
    t = type(x)  #     t = type(x)
    b = t.__bases__  #     b = t.__bases__
    if hasattr(x, "__slots__"):  #     if hasattr(x,"__slots__"): return True
        return True  #     if hasattr(x,"__slots__"): return True
    if len(b) != 1 or b[0] != tuple:  #     if len(b) != 1 or b[0] != tuple: return False
        return False  #     if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)  #     f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):  #     if not isinstance(f, tuple): return False
        return False  #     if not isinstance(f, tuple): return False
    return all((type(n) == str for n in f))  #     return all(type(n)==str for n in f)

def cast_tuple2auto_tuple(state):  # def cast_tuple2auto_tuple(state):
    if isinstance(state, str):  #     if isinstance(state,str):
        return None  #         return None
    if isinstance(state, AutoTuple):  #     if isinstance(state,AutoTuple):
        res = [state.formats]  #         res = [state.formats]
        return res  #         return res
    elif type(state) == tuple:  #     elif type(state) == tuple:
        res = [AutoTuple(state)]  #         res = [AutoTuple(state)]
        return res  #         return res

def map_tuple_i(t, i, f):  # def map_tuple_i(t,i,f):
    res = list(t)  #     res = list(t)
    res[i] = f(res[i])  #     res[i] = f(res[i])
    return tuple(res)  #     return tuple(res)

def map_state(states, i, new_state):  # def map_state(states,i,new_state):
    res = list(states)  #     res = list(states)
    res[i] = new_state  #     res[i] = new_state
    return tuple(res)  #     return tuple(res)

def intra_tuple_conversions(state):  # def intra_tuple_conversions(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_15 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, AutoTuple)) and (_coconut.len(_coconut_match_to) == 1):  #     case state:
        items = _coconut_match_to[0]  #     case state:
        _coconut_case_check_15 = True  #     case state:
    if _coconut_case_check_15:  #     case state:
        res = []  #             res = []
        for i in range(len(items)):  #             for i in range(len(items)):
            ith_state = items[i]  #                 ith_state = items[i]
            res += [(lambda f, new_state, cost, name: (lambda values: map_tuple_i(values, i, f), AutoTuple(map_state(items, i, new_state)), "map {_coconut_format_0}th element with {_coconut_format_1}".format(_coconut_format_0=(i), _coconut_format_1=(name)), cost))(f, new_state, cost, name) for f, new_state, cost, name in SOLVER.solver.neighbors(ith_state)]  #                 res += [
        return res  #             return res


def map_each(t, mappers):  # def map_each(t,mappers):
#logger.warning(f"items:{t}")
#logger.warning(f"mappers:{mappers}")
    return tuple([f(item) for f, item in zip(mappers, t)])  #     return tuple([f(item) for f,item in zip(mappers,t)])

def smart_tuple_conversion(state, end):  # def smart_tuple_conversion(state,end):
    _coconut_match_to = (state, end)  #     case (state,end):
    _coconut_case_check_16 = False  #     case (state,end):
    if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], AutoTuple)) and (_coconut.len(_coconut_match_to[0]) == 1):  #     case (state,end):
        formats = _coconut_match_to[0][0]  #     case (state,end):
        t = _coconut_match_to[1]  #     case (state,end):
        _coconut_case_check_16 = True  #     case (state,end):
    if _coconut_case_check_16 and not (type(t) == tuple and len(formats) == len(t)):  #     case (state,end):
        _coconut_case_check_16 = False  #     case (state,end):
    if _coconut_case_check_16:  #     case (state,end):
        cs = []  #             cs = []
        cost = 0  #             cost = 0
        for i in range(len(formats)):  #             for i in range(len(formats)):
            c = SOLVER.solver.search_direct(formats[i], end[i])  #                 c = SOLVER.solver.search_direct(formats[i],end[i])
            cost += sum((e.cost for e in c.edges))  #                 cost += sum(e.cost for e in c.edges)
            cs.append(c)  #                 cs.append(c)
        res = [(lambda t: map_each(t, cs), end, "{_coconut_format_0}->{_coconut_format_1}".format(_coconut_format_0=(state), _coconut_format_1=(end)), cost)]  #             res = [
        logger.debug(res)  #             logger.debug(res)
        return res  #             return res
    if not _coconut_case_check_16:  #         match (AutoTuple(formats),"widget"):
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], AutoTuple)) and (_coconut.len(_coconut_match_to[0]) == 1) and (_coconut_match_to[1] == "widget"):  #         match (AutoTuple(formats),"widget"):
            formats = _coconut_match_to[0][0]  #         match (AutoTuple(formats),"widget"):
            _coconut_case_check_16 = True  #         match (AutoTuple(formats),"widget"):
        if _coconut_case_check_16:  #         match (AutoTuple(formats),"widget"):
            f, new_state, name, cost = smart_tuple_conversion(state, ("widget",) * len(state.formats))[0]  #             f,new_state,name,cost = smart_tuple_conversion(state,("widget",)*len(state.formats))[0]
            logger.debug("cost:{_coconut_format_0}".format(_coconut_format_0=(cost)))  #             logger.debug(f"cost:{cost}")
            return [(lambda t: widgets.VBox(f(t)), end, "{_coconut_format_0}->{_coconut_format_1}".format(_coconut_format_0=(state), _coconut_format_1=(end)), cost + 1)]  #             return [(

class AutoList(_coconut.collections.namedtuple("AutoList", "state")):  # data AutoList(state):
    __slots__ = ()  # data AutoList(state):
    __ne__ = _coconut.object.__ne__  # data AutoList(state):
    def __eq__(self, other):  # data AutoList(state):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data AutoList(state):
    def __hash__(self):  # data AutoList(state):
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data AutoList(state):
    def __str__(self):  #     def __str__(self):
        return "[{_coconut_format_0}]".format(_coconut_format_0=(self.state))  #         return f"[{self.state}]"


def unlist(items):  # def unlist(items):
    return SOLVER.new_auto_data([i.value for i in items], AutoList(items[0].format))  #     return SOLVER.new_auto_data([i.value for i in items],AutoList(items[0].format))

def cast_ary_str_to_ary_type(state):  # def cast_ary_str_to_ary_type(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_17 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, _coconut.str)) and (_coconut_match_to.startswith("[")) and (_coconut_match_to.endswith("]")):  #     case state:
        element_state = _coconut_match_to[_coconut.len("["):-_coconut.len("]")]  #     case state:
        _coconut_case_check_17 = True  #     case state:
    if _coconut_case_check_17:  #     case state:
        return [AutoList(element_state)]  #             return [AutoList(element_state)]
    if not _coconut_case_check_17:  #         match AutoList(es is str):
        if (_coconut.isinstance(_coconut_match_to, AutoList)) and (_coconut.len(_coconut_match_to) == 1) and (_coconut.isinstance(_coconut_match_to[0], str)):  #         match AutoList(es is str):
            es = _coconut_match_to[0]  #         match AutoList(es is str):
            _coconut_case_check_17 = True  #         match AutoList(es is str):
        if _coconut_case_check_17:  #         match AutoList(es is str):
            return ["[{_coconut_format_0}]".format(_coconut_format_0=(es))]  #             return [f"[{es}]"]

def intra_list_conversions(state):  # def intra_list_conversions(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_18 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, AutoList)) and (_coconut.len(_coconut_match_to) == 1):  #     case state:
        es = _coconut_match_to[0]  #     case state:
        _coconut_case_check_18 = True  #     case state:
    if _coconut_case_check_18:  #     case state:
        return [(lambda f, new_state, cost, name: (lambda items: [f(i) for i in items], AutoList(new_state), "[{_coconut_format_0}]".format(_coconut_format_0=(name)), cost + 1))(f, new_state, cost, name) for f, new_state, cost, name in SOLVER.solver.neighbors(es)]  #             return [((f,new_state,cost,name)->(

def img_list_is_imgs(state):  # def img_list_is_imgs(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_19 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, AutoList)) and (_coconut.len(_coconut_match_to) == 1) and (_coconut.isinstance(_coconut_match_to[0], _coconut.str)) and (_coconut_match_to[0].startswith("image,")):  #     case state:
        formats = _coconut_match_to[0][_coconut.len("image,"):]  #     case state:
        _coconut_case_check_19 = True  #     case state:
    if _coconut_case_check_19:  #     case state:
        return ["images,{_coconut_format_0}".format(_coconut_format_0=(formats))]  #             return [f"images,{formats}"]
    if not _coconut_case_check_19:  #         match "images,"+formats:
        if (_coconut.isinstance(_coconut_match_to, _coconut.str)) and (_coconut_match_to.startswith("images,")):  #         match "images,"+formats:
            formats = _coconut_match_to[_coconut.len("images,"):]  #         match "images,"+formats:
            _coconut_case_check_19 = True  #         match "images,"+formats:
        if _coconut_case_check_19:  #         match "images,"+formats:
            return [AutoList("image," + formats)]  #             return [AutoList("image,"+formats)]
def numpys_to_numpy(state):  # def numpys_to_numpy(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_20 = False  #     case state:
    if (_coconut.isinstance(_coconut_match_to, AutoList)) and (_coconut.len(_coconut_match_to) == 1) and (_coconut.isinstance(_coconut_match_to[0], _coconut.abc.Mapping)):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to[0].get("type", _coconut_sentinel)  #     case state:
        _coconut_match_temp_1 = _coconut_match_to[0].get("arrange", _coconut_sentinel)  #     case state:
        if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "numpy") and (_coconut_match_temp_1 is not _coconut_sentinel):  #     case state:
            arng = _coconut_match_temp_1  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to[0].items() if k not in set(("type", "arrange")))  #     case state:
            _coconut_case_check_20 = True  #     case state:
    if _coconut_case_check_20 and not ("B" not in arng):  #     case state:
        _coconut_case_check_20 = False  #     case state:
    if _coconut_case_check_20:  #     case state:
        return [(lambda numpys: np.array(numpys), frozendict({"type": "numpy", "arrange": "B" + arng, **kwargs}), "merge arrays to array".format(), 10)]  #             return [
def tensor_to_list(state):  # def tensor_to_list(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_21 = False  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("arrange", _coconut_sentinel)  #     case state:
        if _coconut_match_temp_0 is not _coconut_sentinel:  #     case state:
            arng = _coconut_match_temp_0  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("arrange",)))  #     case state:
            _coconut_case_check_21 = True  #     case state:
    if _coconut_case_check_21 and not (len(arng) > 1):  #     case state:
        _coconut_case_check_21 = False  #     case state:
    if _coconut_case_check_21:  #     case state:
        return [(lambda tensor: [t for t in tensor], AutoList(frozendict(arrange=arng[1:], **kwargs)), "tensor to list of tensor".format(), 2)]  #             return [

def pil_convert(state):  # def pil_convert(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_22 = False  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #     case state:
        if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "image"):  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type",)))  #     case state:
            _coconut_case_check_22 = True  #     case state:
    if _coconut_case_check_22:  #     case state:
        new_state = dict(**state)  #             new_state = dict(**state)
        return [(lambda img: lambda mode: SOLVER.new_auto_data(img.convert(mode), "image,{_coconut_format_0},{_coconut_format_1}".format(_coconut_format_0=(mode), _coconut_format_1=(mode))), "pil_convert", "image_to_pil_converter", 1)]  #             return [

def rgb_to_rgba(state):  # def rgb_to_rgba(state):
    if state == "numpy,uint8,HWC,RGB,0_255":  #     if state == "numpy,uint8,HWC,RGB,0_255":
        return [(lambda a: np.concatenate((a, np.ones((*a.shape[:2], 1), dtype="uint8") * 255), axis=2), "numpy,uint8,HWC,RGBA,0_255", "add 255 as alpha channel", 10)]  #         return [(
    elif state == "numpy,uint8,BHWC,RGB,0_255":  #     elif state == "numpy,uint8,BHWC,RGB,0_255":
        return [(lambda a: np.concatenate((a, np.ones((*a.shape[:3], 1), dtype="uint8") * 255), axis=3), "numpy,uint8,BHWC,RGBA,0_255", "add 255 as alpha channel to batch", 10)]  #         return [(

@memoize()  # @memoize()
def pix2pix_normalizer(nc):  # def pix2pix_normalizer(nc):
    from torchvision import transforms  #     import torchvision.transforms as transforms
    return transforms.Normalize((0.5,) * nc, (0.5,) * nc)  #     return transforms.Normalize((0.5,)*nc,(0.5,)*nc)


def torch_img_to_pixpix_input(state):  # def torch_img_to_pixpix_input(state):
    import torch  #     import torch
    _coconut_match_to = state  #     case state:
    _coconut_case_check_23 = False  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #     case state:
        _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #     case state:
        _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #     case state:
        _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #     case state:
        _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #     case state:
        if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "CHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "RGB"):  #     case state:
            rpr = _coconut_match_temp_4  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #     case state:
            _coconut_case_check_23 = True  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #     case state:
        _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #     case state:
        _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #     case state:
        _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #     case state:
        _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #     case state:
        if (not _coconut_case_check_23) and (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "CHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "RGBA"):  #     case state:
            rpr = _coconut_match_temp_4  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #     case state:
            _coconut_case_check_23 = True  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #     case state:
        _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #     case state:
        _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #     case state:
        _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #     case state:
        _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #     case state:
        if (not _coconut_case_check_23) and (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "CHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "L"):  #     case state:
            rpr = _coconut_match_temp_4  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #     case state:
            _coconut_case_check_23 = True  #     case state:
    if _coconut_case_check_23:  #     case state:
        return [(pix2pix_normalizer(len(rpr)), "pix2pix,nc={_coconut_format_0}".format(_coconut_format_0=(len(rpr))), "convert to pixpix normalized input", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 pix2pix_normalizer(len(rpr)),
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "BCHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "RGB"):  #                 pix2pix_normalizer(len(rpr)),
                rpr = _coconut_match_temp_4  #                 pix2pix_normalizer(len(rpr)),
                kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #                 pix2pix_normalizer(len(rpr)),
                _coconut_case_check_23 = True  #                 pix2pix_normalizer(len(rpr)),
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            if (not _coconut_case_check_23) and (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "BCHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "RGBA"):  #                 pix2pix_normalizer(len(rpr)),
                rpr = _coconut_match_temp_4  #                 pix2pix_normalizer(len(rpr)),
                kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #                 pix2pix_normalizer(len(rpr)),
                _coconut_case_check_23 = True  #                 pix2pix_normalizer(len(rpr)),
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 pix2pix_normalizer(len(rpr)),
            if (not _coconut_case_check_23) and (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "BCHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "L"):  #                 pix2pix_normalizer(len(rpr)),
                rpr = _coconut_match_temp_4  #                 pix2pix_normalizer(len(rpr)),
                kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #                 pix2pix_normalizer(len(rpr)),
                _coconut_case_check_23 = True  #                 pix2pix_normalizer(len(rpr)),
        if _coconut_case_check_23:  #                 pix2pix_normalizer(len(rpr)),
            return [(lambda t: torch.cat([pix2pix_normalizer(len(rpr))(i)[None] for i in t], dim=0), "pix2pix_batch,nc={_coconut_format_0}".format(_coconut_format_0=(len(rpr))), "convert to pixpix normalized input", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 t->torch.cat([pix2pix_normalizer(len(rpr))(i)[None] for i in t],dim=0),
        if _coconut_match_to == "pix2pix_laba":  #                 t->torch.cat([pix2pix_normalizer(len(rpr))(i)[None] for i in t],dim=0),
            _coconut_case_check_23 = True  #                 t->torch.cat([pix2pix_normalizer(len(rpr))(i)[None] for i in t],dim=0),
        if _coconut_case_check_23:  #                 t->torch.cat([pix2pix_normalizer(len(rpr))(i)[None] for i in t],dim=0),
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,CHW,LABA,0_1".format(), "inverse pix2pix_laba to img ", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix_lab":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,CHW,LAB,0_1".format(), "inverse pix2pix_lab to img ", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix_laba_batch":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,BCHW,LABA,0_1".format(), "inverse pix2pix_laba batch to img", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix_lab_batch":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,BCHW,LAB,0_1".format(), "inverse pix2pix_laba batch to img", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix,nc=4":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,CHW,RGBA,0_1".format(), "inverse pix2pix to img", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix_batch,nc=4":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,BCHW,RGBA,0_1".format(), "inverse pix2pix batch nc=4 to img", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix_batch,nc=3":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,BCHW,RGB,0_1".format(), "inverse pix2pix batch nc=3 to img", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix,nc=3":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,CHW,RGB,0_1".format(), "inverse pix2pix to img", 1)]  #             return [(
    if not _coconut_case_check_23:  #                 a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix_batch,nc=1":  #                 a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                 a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                 a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,BCHW,L,0_1".format(), "inverse pix2pix_batch,nc=1 to img", 1)]  #             return [(
    if not _coconut_case_check_23:  #                a -> a*0.5+0.5,
        if _coconut_match_to == "pix2pix,nc=1":  #                a -> a*0.5+0.5,
            _coconut_case_check_23 = True  #                a -> a*0.5+0.5,
        if _coconut_case_check_23:  #                a -> a*0.5+0.5,
            return [(lambda a: a * 0.5 + 0.5, "torch,float32,CHW,L,0_1".format(), "inverse pix2pix,nc=1 to img", 1)]  #             return [(

@memoize()  # @memoize()
def _VGG_NORMALIZER():  # def _VGG_NORMALIZER():
    from torchvision import transforms  #     import torchvision.transforms as transforms
    nrm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #     nrm = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    return nrm  #     return nrm
def inverse_vgg_prep(tensor):  # def inverse_vgg_prep(tensor):
    return tensor * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]  #     return tensor * torch.tensor([0.229,0.224,0.225])[:,None,None] + torch.tensor([0.485,0.456,0.406])[:,None,None]
def inverse_vgg_prep_batch(tensor):  # def inverse_vgg_prep_batch(tensor):
    return tensor * torch.tensor([0.229, 0.224, 0.225])[None, :, None, None] + torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]  #     return tensor * torch.tensor([0.229,0.224,0.225])[None,:,None,None] + torch.tensor([0.485,0.456,0.406])[None,:,None,None]
def torch_img_to_vgg_prep(state):  # def torch_img_to_vgg_prep(state):
    VGG_NORMALIZER = _VGG_NORMALIZER()  #     VGG_NORMALIZER = _VGG_NORMALIZER()
    _coconut_match_to = state  #     case state:
    _coconut_case_check_24 = False  #     case state:
    if _coconut_match_to == "vgg_prep":  #     case state:
        _coconut_case_check_24 = True  #     case state:
    if _coconut_case_check_24:  #     case state:
        return [(inverse_vgg_prep, "torch,float32,CHW,RGB,0_1", "inverse from vgg_prep", 1)]  #             return [(
    if not _coconut_case_check_24:  #                 inverse_vgg_prep,
        if _coconut_match_to == "vgg_prep_batch":  #                 inverse_vgg_prep,
            _coconut_case_check_24 = True  #                 inverse_vgg_prep,
        if _coconut_case_check_24:  #                 inverse_vgg_prep,
            return [(inverse_vgg_prep_batch, "torch,float32,BCHW,RGB,0_1", "inverse from vgg_prep_batch", 1)]  #             return [(
    if not _coconut_case_check_24:  #                 inverse_vgg_prep_batch,
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 inverse_vgg_prep_batch,
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 inverse_vgg_prep_batch,
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 inverse_vgg_prep_batch,
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 inverse_vgg_prep_batch,
            _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 inverse_vgg_prep_batch,
            _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 inverse_vgg_prep_batch,
            if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "CHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "RGB"):  #                 inverse_vgg_prep_batch,
                kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #                 inverse_vgg_prep_batch,
                _coconut_case_check_24 = True  #                 inverse_vgg_prep_batch,
        if _coconut_case_check_24:  #                 inverse_vgg_prep_batch,
            return [(VGG_NORMALIZER, "vgg_prep".format(), "convert to vgg normalized input", 1)]  #             return [(
    if not _coconut_case_check_24:  #                 VGG_NORMALIZER,
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 VGG_NORMALIZER,
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 VGG_NORMALIZER,
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 VGG_NORMALIZER,
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 VGG_NORMALIZER,
            _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 VGG_NORMALIZER,
            _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 VGG_NORMALIZER,
            if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "BCHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "RGB"):  #                 VGG_NORMALIZER,
                kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #                 VGG_NORMALIZER,
                _coconut_case_check_24 = True  #                 VGG_NORMALIZER,
        if _coconut_case_check_24:  #                 VGG_NORMALIZER,
            return [(lambda t: torch.cat([VGG_NORMALIZER(i)[None] for i in t], dim=0), "vgg_prep_batch".format(), "convert to vgg normalized input batch", 1)]  #             return [(
    if not _coconut_case_check_24:  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
            _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
            _coconut_match_temp_1 = _coconut_match_to.get("dtype", _coconut_sentinel)  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
            _coconut_match_temp_2 = _coconut_match_to.get("arrange", _coconut_sentinel)  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
            _coconut_match_temp_3 = _coconut_match_to.get("v_range", _coconut_sentinel)  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
            _coconut_match_temp_4 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
            if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "torch") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "float32") and (_coconut_match_temp_2 is not _coconut_sentinel) and (_coconut_match_temp_2 == "BCHW") and (_coconut_match_temp_3 is not _coconut_sentinel) and (_coconut_match_temp_3 == "0_1") and (_coconut_match_temp_4 is not _coconut_sentinel) and (_coconut_match_temp_4 == "RGBA"):  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
                kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "dtype", "arrange", "v_range", "ch_rpr")))  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
                _coconut_case_check_24 = True  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
        if _coconut_case_check_24:  #                 t->torch.cat([VGG_NORMALIZER(i)[None] for i in t],dim=0),
            return [(lambda t: torch.cat([torch.cat((VGG_NORMALIZER(i[:3])[None], i[[3]][None]), dim=1) for i in t], dim=0), "vgg_prep_batch_masked".format(), "convert to vgg normalized input batch", 1)]  #             return [(
def repeat_ch(state):  # def repeat_ch(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_25 = False  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("type", _coconut_sentinel)  #     case state:
        _coconut_match_temp_1 = _coconut_match_to.get("mode", _coconut_sentinel)  #     case state:
        _coconut_match_temp_2 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #     case state:
        if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "image") and (_coconut_match_temp_1 is not _coconut_sentinel) and (_coconut_match_temp_1 == "L") and (_coconut_match_temp_2 is not _coconut_sentinel):  #     case state:
            ch = _coconut_match_temp_2  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("type", "mode", "ch_rpr")))  #     case state:
            _coconut_case_check_25 = True  #     case state:
    if _coconut_case_check_25 and not (len(ch) == 1):  #     case state:
        _coconut_case_check_25 = False  #     case state:
    if _coconut_case_check_25:  #     case state:
        return [(lambda a: np.repeat(np.array(a)[:, :, None], 3, axis=2), frozendict(type="numpy", dtype="uint8", arrange="HWC", ch_rpr=ch * 3, v_range="0_255"), "repeat_channel_3", 50)]  #             return [



def lll_is_rgb(state):  # def lll_is_rgb(state):
    _coconut_match_to = state  #     case state:
    _coconut_case_check_26 = False  #     case state:
    if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):  #     case state:
        _coconut_match_temp_0 = _coconut_match_to.get("ch_rpr", _coconut_sentinel)  #     case state:
        if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_0 == "LLL"):  #     case state:
            kwargs = dict((k, v) for k, v in _coconut_match_to.items() if k not in set(("ch_rpr",)))  #     case state:
            _coconut_case_check_26 = True  #     case state:
    if _coconut_case_check_26:  #     case state:
        return [frozendict(ch_rpr="RGB", **kwargs)]  #             return [frozendict(ch_rpr="RGB",**kwargs)]



DEFAULT_RULES = AutoImage.default_rules.copy() + [AutoSolver.create_cast_rule(cast_imdef_to_dict, "cast_imdef_to_dict"), AutoSolver.create_cast_rule(cast_imdef_str_to_imdef, "cast_imdef_str_to_imdef"), AutoSolver.create_cast_rule(cast_imdef_to_imdef_str, "cast_imdef_to_imdef_str"), AutoSolver.create_cast_rule(dict2imdef, "dict2imdef"), AutoSolver.create_cast_rule(cast_ary_str_to_ary_type, "cast_ary_str_to_ary_type"), AutoSolver.create_cast_rule(img_list_is_imgs, "img_list_is_imgs"), AutoSolver.create_cast_rule(lll_is_rgb, "lll_is_rgb", cost=10), AutoSolver.create_cast_rule(cast_tuple2auto_tuple, "tuple <--> auto_tuple"), AutoSolver.create_conversion_rule(any2widget), AutoSolver.create_conversion_rule(to_visdom_function), AutoSolver.create_conversion_rule(rule_imgs2tile), AutoSolver.create_conversion_rule(rule_img2widget), AutoSolver.create_conversion_rule(rule_numpy2img), AutoSolver.create_conversion_rule(rule_image2gray), AutoSolver.create_conversion_rule(rule_image2lab), AutoSolver.create_conversion_rule(rule_rgba2laba), AutoSolver.create_conversion_rule(rule_lab_value_conversion), AutoSolver.create_conversion_rule(intra_list_conversions), AutoSolver.create_conversion_rule(numpys_to_numpy), AutoSolver.create_conversion_rule(tensor_to_list), AutoSolver.create_conversion_rule(pil_convert), AutoSolver.create_conversion_rule(rgb_to_rgba), AutoSolver.create_conversion_rule(repeat_ch), AutoSolver.create_conversion_rule(torch_img_to_pixpix_input), AutoSolver.create_conversion_rule(torch_img_to_vgg_prep), AutoSolver.create_conversion_rule(auto_tuple2widget), AutoSolver.create_alias_rule("numpy_rgb", "numpy,uint8,HWC,RGB,0_255"), AutoSolver.create_alias_rule("numpy_rgba", "numpy,uint8,HWC,RGBA,0_255"),]  # DEFAULT_RULES = AutoImage.default_rules.copy() + [

SMART_RULES = [AutoSolver.create_smart_conversion_rule(smart_tuple_conversion),]  # SMART_RULES =[

try:  # try:
    import wandb  #     import wandb
    def img_to_wandb_img(state):  #     def img_to_wandb_img(state):
        _coconut_match_to = state  #         case state:
        _coconut_case_check_27 = False  #         case state:
        if (_coconut.isinstance(_coconut_match_to, ImageDef)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut.isinstance(_coconut_match_to[0], PILImage)) and (_coconut.len(_coconut_match_to[0]) == 2):  #         case state:
            _coconut_case_check_27 = True  #         case state:
        if _coconut_case_check_27:  #         case state:
            return [(lambda img: wandb.Image(img), "wandb.Image", "image to wandb image", 1)]  #                 return [(
    DEFAULT_RULES.append(AutoSolver.create_conversion_rule(img_to_wandb_img))  #     DEFAULT_RULES.append(AutoSolver.create_conversion_rule(img_to_wandb_img))
    logger.warning("added wandb related conversions".format())  #     logger.warning(f"added wandb related conversions")
except Exception as e:  # except Exception as e:
    logger.warning("could not add wandb related conversions since wandb could not be imported".format())  #     logger.warning(f"could not add wandb related conversions since wandb could not be imported")


def tuple_distance(x, y):  # def tuple_distance(x,y):
    assert len(x) == len(y), "cannot compare two tuples with different length"  #     assert len(x) == len(y),"cannot compare two tuples with different length"
    return len(x) - sum(tuple([i == j for i, j in zip(x, y)]))  #     return len(x) - sum(tuple([i==j for i,j in zip(x,y)]))


@memoize()  # @memoize()
def state_distance(x, y):  # def state_distance(x,y):
    conversion = SOLVER.solver.search_direct(x, y, silent=True)  #     conversion = SOLVER.solver.search_direct(x,y,silent=True)
    d = len(conversion.edges)  #     d = len(conversion.edges)
#logger.info(f"heuristic conversion:{conversion}")
#logger.info(f"{x} to {y}:{d}")
    return d  #     return d

@memoize()  # @memoize()
def tuple_state_distance(x, y):  # def tuple_state_distance(x,y):
    return sum([state_distance(i, j) for i, j in zip(x, y)])  #     return sum([state_distance(i,j) for i,j in  zip(x,y)])

@memoize()  # @memoize()
def tuple_widget_heuristics(x, y):  # def tuple_widget_heuristics(x,y):
    """
    you have to make the solver solve one by one.
    """  #     """
    res = 0  #     res = 0
    return 0  #     return 0
    if type(x) == tuple and type(y) == tuple:  #     if type(x) == tuple and type(y) == tuple:
        if len(x) == len(y):  #         if len(x) == len(y):
            res = tuple_distance(x, y)  #             res = tuple_distance(x,y)
    if isinstance(x, AutoTuple) and type(y) == tuple:  #     if isinstance(x,AutoTuple) and type(y) == tuple:
        if len(x.formats) == len(y):  #         if len(x.formats) == len(y):
            xs = x.formats  #             xs = x.formats
            ys = y  #             ys = y
            res = tuple_distance(xs, ys)  #             res = tuple_distance(xs,ys)
    elif isinstance(x, AutoTuple) and y == "widget":  #     elif isinstance(x,AutoTuple) and y == "widget":
        xs = x.formats  #         xs = x.formats
        ys = ("widget",) * len(x.formats)  #         ys = ("widget",)*len(x.formats)
        res = tuple_distance(xs, ys)  #         res = tuple_distance(xs,ys)
#if res == 0:
#    logger.info(f"{x}->{y}:{res}")
#    pass
    return res  #     return res

def tuple_edge_cutter(x, y, end):  # def tuple_edge_cutter(x,y,end):
    return False  #     return False
    if isinstance(x, AutoTuple) and type(y) == tuple and type(end) == tuple:  #     if isinstance(x,AutoTuple) and type(y) == tuple and type(end) == tuple:
        n = len(x.formats)  #          n = len(x.formats)
        if n == len(y) and n == len(end):  #          if n == len(y) and n == len(end):
            x2end = tuple_distance(x.formats, end)  #              x2end = tuple_distance(x.formats,end)
            y2end = tuple_distance(y, end)  #              y2end = tuple_distance(y,end)
            x_matching = n - x2end  #              x_matching = n - x2end
            y_matching = n - y2end  #              y_matching = n - y2end
            if y_matching < x_matching:  #              if y_matching < x_matching:
                logger.debug("cut {_coconut_format_0} to {_coconut_format_1} for {_coconut_format_2}".format(_coconut_format_0=(x), _coconut_format_1=(y), _coconut_format_2=(end)))  #                  logger.debug(f"cut {x} to {y} for {end}")
                return True  #                  return True
    return False  #     return False




SOLVER = AutoSolver(rules=DEFAULT_RULES.copy(), smart_rules=SMART_RULES.copy(), heuristics=tuple_widget_heuristics, edge_cutter=tuple_edge_cutter)  # SOLVER = AutoSolver(
auto_img = lambda format: lambda value: AutoData(value, format, SOLVER)  # auto_img = format->value->AutoData(value,format,SOLVER)
