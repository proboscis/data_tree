#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x5f0fb1bc

# Compiled with Coconut version 1.6.0 [Vocational Guidance Counsellor]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop
import sys as _coconut_sys, os as _coconut_os
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.dirname(_coconut_os.path.abspath(__file__)))
_coconut_cached_module = _coconut_sys.modules.get("__coconut__")
if _coconut_cached_module is not None and _coconut_os.path.dirname(_coconut_cached_module.__file__) != _coconut_file_dir:
    del _coconut_sys.modules["__coconut__"]
_coconut_sys.path.insert(0, _coconut_file_dir)
_coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):
    _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")
    import __coconut__ as _coconut__coconut__
    _coconut__coconut__.__name__ = _coconut_full_module_name
    for _coconut_v in vars(_coconut__coconut__).values():
        if getattr(_coconut_v, "__module__", None) == "__coconut__":
            try:
                _coconut_v.__module__ = _coconut_full_module_name
            except AttributeError:
                _coconut_v_type = type(_coconut_v)
                if getattr(_coconut_v_type, "__module__", None) == "__coconut__":
                    _coconut_v_type.__module__ = _coconut_full_module_name
    _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import io  # import io

import PIL  # import PIL
from PIL import Image  # from PIL import Image


def to_image_bytes(img, format) -> 'bytes':  # def to_image_bytes(img,format)->bytes:
    buf = io.BytesIO()  #     buf = io.BytesIO()
    opts = dict()  #     opts = dict()
    if "jpg" in format.lower():  #     if "jpg" in format.lower():
        opts["subsampling"] = 0  #         opts["subsampling"] = 0
        opts["quality"] = 100  #         opts["quality"] = 100
    img.save(buf, format=format, **opts)  #     img.save(buf,format=format,**opts)
    return (buf.getvalue())  #     return buf.getvalue()
def from_jpg_bytes(bs: 'bytes') -> 'PIL.Image.Image':  # def from_jpg_bytes(bs:bytes)->PIL.Image.Image:
    return (Image.open(io.BytesIO(bs)))  #     return Image.open(io.BytesIO(bs))

def rule_image_str_to_jpg_bytes(state):  # def rule_image_str_to_jpg_bytes(state):
    _coconut_case_match_to_0 = state  #     case state:
    _coconut_case_match_check_0 = False  #     case state:
    if _coconut_case_match_to_0 == "image,RGB,RGB":  #     case state:
        _coconut_case_match_check_0 = True  #     case state:
    if _coconut_case_match_check_0:  #     case state:
        return ([(lambda i: to_image_bytes(i, "jpeg"), "jpg_bytes", "image to jpg bytes", 1), (lambda i: to_image_bytes(i, "png"), "png_bytes", "image to png bytes", 1)])  #             return [
    if not _coconut_case_match_check_0:  #                 (i -> to_image_bytes(i,"jpeg"),
        if _coconut_case_match_to_0 == "jpg_bytes":  #                 (i -> to_image_bytes(i,"jpeg"),
            _coconut_case_match_check_0 = True  #                 (i -> to_image_bytes(i,"jpeg"),
        if _coconut_case_match_check_0:  #                 (i -> to_image_bytes(i,"jpeg"),
            return ([(from_jpg_bytes, "image,RGB,RGB", "jpg bytes to Image", 1), ])  #             return [
    if not _coconut_case_match_check_0:  #                 (from_jpg_bytes,
        if _coconut_case_match_to_0 == "png_bytes":  #                 (from_jpg_bytes,
            _coconut_case_match_check_0 = True  #                 (from_jpg_bytes,
        if _coconut_case_match_check_0:  #                 (from_jpg_bytes,
            return ([(from_jpg_bytes, "image,RGB,RGB", "png bytes to Image"), ])  #             return [
