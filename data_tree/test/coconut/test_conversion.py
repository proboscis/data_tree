#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x174feecc

# Compiled with Coconut version 2.2.0

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop
import sys as _coconut_sys
_coconut_header_info = ('2.2.0', '35', True, False, False)
import os as _coconut_os
_coconut_cached__coconut__ = _coconut_sys.modules.get('__coconut__')
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
_coconut_pop_path = False
if _coconut_cached__coconut__ is None or getattr(_coconut_cached__coconut__, "_coconut_header_info", None) != _coconut_header_info and _coconut_os.path.dirname(_coconut_cached__coconut__.__file__ or "") != _coconut_file_dir:
    if _coconut_cached__coconut__ is not None:
        _coconut_sys.modules['_coconut_cached__coconut__'] = _coconut_cached__coconut__
        del _coconut_sys.modules['__coconut__']
    _coconut_sys.path.insert(0, _coconut_file_dir)
    _coconut_pop_path = True
    _coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
    if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):
        _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")
        import __coconut__ as _coconut__coconut__
        _coconut__coconut__.__name__ = _coconut_full_module_name
        for _coconut_v in vars(_coconut__coconut__).values():
            if getattr(_coconut_v, "__module__", None) == '__coconut__':
                try:
                    _coconut_v.__module__ = _coconut_full_module_name
                except AttributeError:
                    _coconut_v_type = type(_coconut_v)
                    if getattr(_coconut_v_type, "__module__", None) == '__coconut__':
                        _coconut_v_type.__module__ = _coconut_full_module_name
        _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _namedtuple_of, _coconut, _coconut_super, _coconut_Expected, _coconut_MatchError, _coconut_iter_getitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_raise, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec, _coconut_comma_op, _coconut_multi_dim_arr, _coconut_mk_anon_namedtuple, _coconut_matmul, _coconut_py_str, _coconut_flatten, _coconut_multiset, _coconut_back_none_pipe, _coconut_back_none_star_pipe, _coconut_back_none_dubstar_pipe, _coconut_forward_none_compose, _coconut_back_none_compose, _coconut_forward_none_star_compose, _coconut_back_none_star_compose, _coconut_forward_none_dubstar_compose, _coconut_back_none_dubstar_compose
if _coconut_pop_path:
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

from data_tree.coconut.astar import astar  # from data_tree.coconut.astar import astar
from loguru import logger  # from loguru import logger
import numpy as np  # import numpy as np
import torch  # import torch
from PIL import Image  # from PIL import Image
from data_tree import auto_image  # from data_tree import auto_image
from data_tree.coconut.convert import Torch  # from data_tree.coconut.convert import Torch,Numpy,VR_0_1,ImageDef
from data_tree.coconut.convert import Numpy  # from data_tree.coconut.convert import Torch,Numpy,VR_0_1,ImageDef
from data_tree.coconut.convert import VR_0_1  # from data_tree.coconut.convert import Torch,Numpy,VR_0_1,ImageDef
from data_tree.coconut.convert import ImageDef  # from data_tree.coconut.convert import Torch,Numpy,VR_0_1,ImageDef
from data_tree.coconut.convert import str_to_img_def  # from data_tree.coconut.convert import str_to_img_def,_conversions,_edges
from data_tree.coconut.convert import _conversions  # from data_tree.coconut.convert import str_to_img_def,_conversions,_edges
from data_tree.coconut.convert import _edges  # from data_tree.coconut.convert import str_to_img_def,_conversions,_edges
from data_tree.coconut.astar import AStarSolver  # from data_tree.coconut.astar import AStarSolver
start = str_to_img_def("numpy,float32,HWC,RGB,0_1")  # start = str_to_img_def("numpy,float32,HWC,RGB,0_1")
end = str_to_img_def("torch,uint8,BHWC,RGB,0_255")  # end = str_to_img_def("torch,uint8,BHWC,RGB,0_255")
class END(_coconut.collections.namedtuple("END", ()), ImageDef):  # data END from ImageDef
    __slots__ = ()  # data END from ImageDef
    _coconut_is_data = True  # data END from ImageDef
    __match_args__ = ()  # data END from ImageDef
    def __add__(self, other): return _coconut.NotImplemented  # data END from ImageDef
    def __mul__(self, other): return _coconut.NotImplemented  # data END from ImageDef
    def __rmul__(self, other): return _coconut.NotImplemented  # data END from ImageDef
    __ne__ = _coconut.object.__ne__  # data END from ImageDef
    def __eq__(self, other):  # data END from ImageDef
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data END from ImageDef
    def __hash__(self):  # data END from ImageDef
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data END from ImageDef

_coconut_call_set_names(END)  # data DUMMY from ImageDef
class DUMMY(_coconut.collections.namedtuple("DUMMY", ()), ImageDef):  # data DUMMY from ImageDef
    __slots__ = ()  # data DUMMY from ImageDef
    _coconut_is_data = True  # data DUMMY from ImageDef
    __match_args__ = ()  # data DUMMY from ImageDef
    def __add__(self, other): return _coconut.NotImplemented  # data DUMMY from ImageDef
    def __mul__(self, other): return _coconut.NotImplemented  # data DUMMY from ImageDef
    def __rmul__(self, other): return _coconut.NotImplemented  # data DUMMY from ImageDef
    __ne__ = _coconut.object.__ne__  # data DUMMY from ImageDef
    def __eq__(self, other):  # data DUMMY from ImageDef
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data DUMMY from ImageDef
    def __hash__(self):  # data DUMMY from ImageDef
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data DUMMY from ImageDef

_coconut_call_set_names(DUMMY)  # from proboscis_image_rules.rulebook import legacy_auto as auto
from proboscis_image_rules.rulebook import legacy_auto as auto  # from proboscis_image_rules.rulebook import legacy_auto as auto
def dummy_rule(imdef):  # def dummy_rule(imdef):
    return ([(lambda a: a, end, 1, "dummy"),])  #     return [(a->a,end,1,"dummy")]


def dummy_rule2(node):  # def dummy_rule2(node):
    paths = []  #     paths = []
    _coconut_case_match_to_0 = node  #     case node:
    _coconut_case_match_check_0 = False  #     case node:
    _coconut_match_temp_0 = _coconut.getattr(DUMMY, "_coconut_is_data", False) or _coconut.isinstance(DUMMY, _coconut.tuple) and _coconut.all(_coconut.getattr(_coconut_x, "_coconut_is_data", False) for _coconut_x in DUMMY)  # type: ignore  #     case node:
    _coconut_case_match_check_0 = True  #     case node:
    if _coconut_case_match_check_0:  #     case node:
        _coconut_case_match_check_0 = False  #     case node:
        if not _coconut_case_match_check_0:  #     case node:
            if (_coconut_match_temp_0) and (_coconut.isinstance(_coconut_case_match_to_0, DUMMY)):  #     case node:
                _coconut_match_temp_1 = _coconut.len(_coconut_case_match_to_0) <= _coconut.max(0, _coconut.len(_coconut_case_match_to_0.__match_args__)) and _coconut.all(i in _coconut.getattr(_coconut_case_match_to_0, "_coconut_data_defaults", {}) and _coconut_case_match_to_0[i] == _coconut.getattr(_coconut_case_match_to_0, "_coconut_data_defaults", {})[i] for i in _coconut.range(0, _coconut.len(_coconut_case_match_to_0.__match_args__))) if _coconut.hasattr(_coconut_case_match_to_0, "__match_args__") else _coconut.len(_coconut_case_match_to_0) == 0  # type: ignore  #     case node:
                if _coconut_match_temp_1:  #     case node:
                    _coconut_case_match_check_0 = True  #     case node:

        if not _coconut_case_match_check_0:  #     case node:
            if (not _coconut_match_temp_0) and (_coconut.isinstance(_coconut_case_match_to_0, DUMMY)):  #     case node:
                _coconut_case_match_check_0 = True  #     case node:
            if _coconut_case_match_check_0:  #     case node:
                _coconut_case_match_check_0 = False  #     case node:
                if not _coconut_case_match_check_0:  #     case node:
                    if _coconut.type(_coconut_case_match_to_0) in _coconut_self_match_types:  #     case node:
                        _coconut_case_match_check_0 = True  #     case node:

                if not _coconut_case_match_check_0:  #     case node:
                    if not _coconut.type(_coconut_case_match_to_0) in _coconut_self_match_types:  #     case node:
                        _coconut_match_temp_2 = _coconut.getattr(DUMMY, '__match_args__', ())  #     case node:
                        if not _coconut.isinstance(_coconut_match_temp_2, _coconut.tuple):  #     case node:
                            raise _coconut.TypeError("DUMMY.__match_args__ must be a tuple")  #     case node:
                        if _coconut.len(_coconut_match_temp_2) < 0:  #     case node:
                            raise _coconut.TypeError("too many positional args in class match (pattern requires 0; 'DUMMY' only supports %s)" % (_coconut.len(_coconut_match_temp_2),))  #     case node:
                        _coconut_case_match_check_0 = True  #     case node:




    if _coconut_case_match_check_0:  #     case node:
        paths += [(lambda a: END, END(), 1, "to_end"),]  #             paths += [(a->END,END(),1,"to_end")]
    if not _coconut_case_match_check_0:  #         match _ is ImageDef:
        if _coconut.isinstance(_coconut_case_match_to_0, ImageDef):  #         match _ is ImageDef:
            _coconut_case_match_check_0 = True  #         match _ is ImageDef:
        if _coconut_case_match_check_0:  #         match _ is ImageDef:
            paths += [(lambda a: DUMMY, DUMMY(), 1, "to_dummy"),]  #             paths += [(a->DUMMY,DUMMY(),1,"to_dummy")]

    return (paths)  #     return paths


def test_something():  # def test_something():
    def neighbors(node):  #     def neighbors(node):
        return ([(lambda a: a + "a", node + "a", 1, "add_a"),])  #         return [(a->a+"a",node + "a",1,"add_a")]

    def matcher(node):  #     def matcher(node):
        return (node == "aaa")  #         return node == "aaa"

    def heuristics(node):  #     def heuristics(node):
        return (0)  #         return 0

    (log_conversion)(astar(start="a", matcher=matcher, neighbors=neighbors, heuristics=heuristics).result)  #     astar(


def imdef_neighbors(imdef):  # def imdef_neighbors(imdef):
    return ([(e.f, e.b, e.cost, e.name) for e in _edges(imdef)])  #     return [(e.f,e.b,e.cost,e.name) for e in _edges(imdef)]


def test_new_astar():  # def test_new_astar():
    (log_conversion)(astar(start=start, matcher=lambda d: d == end, neighbors=imdef_neighbors, heuristics=lambda a: 0).result)  #     astar(


def log_conversion(converter):  # def log_conversion(converter):
    path = [e.name for e in converter.edges]  #     path = [e.name for e in converter.edges]
    logger.info(path)  #     logger.info(path)




def test_astar_solver():  # def test_astar_solver():


    solver = AStarSolver(rules=[imdef_neighbors,])  #     solver=AStarSolver(
    (log_conversion)(solver.search_direct(start, end))  #     solver.search_direct(start,end) |> log_conversion


    solver.add_rule(dummy_rule)  #     solver.add_rule(dummy_rule)
    (log_conversion)(solver.search_direct(start, end))  #     solver.search_direct(start,end) |> log_conversion



def test_auto_image():  # def test_auto_image():
    x = np.zeros((100, 100, 3), dtype="float32")  #     x = np.zeros((100,100,3),dtype="float32")
    x = auto_image(x, start)  #     x = auto_image(x,start)
    (log_conversion)(x.converter(end))  #     x.converter(end) |> log_conversion
    x.solver.add_rule(dummy_rule)  #     x.solver.add_rule(dummy_rule)
    x.solver.add_rule(dummy_rule2)  #     x.solver.add_rule(dummy_rule2)
    (log_conversion)(x.converter(end))  #     x.converter(end) |> log_conversion
    (log_conversion)(x.converter(END()))  #     x.converter(END()) |> log_conversion
    x.reset_solver()  #     x.reset_solver()


def test_non_batch_img_op():  # def test_non_batch_img_op():
    from data_tree.coconut.convert import AutoImage  #     from data_tree.coconut.convert import AutoImage
    x = np.zeros((100, 100), dtype="float32")  #     x = np.zeros((100,100),dtype="float32")

    start = (str_to_img_def)("images,L,L")  #     start = "images,L,L" |> str_to_img_def
    end = (str_to_img_def)("numpy,float32,HW,L,0_1")  #     end = "numpy,float32,HW,L,0_1" |> str_to_img_def
    auto_x = auto_image(x, "numpy,float32,HW,L,0_1")  #     auto_x = auto_image(x,"numpy,float32,HW,L,0_1")
    assert auto_x.image_op(_coconut.operator.methodcaller("resize", (256, 256))).to(end).shape == (256, 256), "image_op must work on non batched image"  #     assert auto_x.image_op(.resize((256,256))).to(end).shape == (256,256),"image_op must work on non batched image"
#AutoImage.solver.search_direct(start,end) |> log_conversion


def test_casting():  # def test_casting():
    from data_tree.coconut.omni_converter import SOLVER  #     from data_tree.coconut.omni_converter import SOLVER,cast_imdef_to_dict,cast_imdef_str_to_imdef
    from data_tree.coconut.omni_converter import cast_imdef_to_dict  #     from data_tree.coconut.omni_converter import SOLVER,cast_imdef_to_dict,cast_imdef_str_to_imdef
    from data_tree.coconut.omni_converter import cast_imdef_str_to_imdef  #     from data_tree.coconut.omni_converter import SOLVER,cast_imdef_to_dict,cast_imdef_str_to_imdef
    logger.info("{_coconut_format_0}".format(_coconut_format_0=(cast_imdef_str_to_imdef('numpy,float32,HW,L,0_1'))))  #     logger.info(f"{cast_imdef_str_to_imdef('numpy,float32,HW,L,0_1')}")



def test_omni_converter():  # def test_omni_converter():
    from data_tree.coconut.omni_converter import auto_img  #     from data_tree.coconut.omni_converter import auto_img,cast_imdef_str_to_imdef,cast_imdef_to_imdef_str
    from data_tree.coconut.omni_converter import cast_imdef_str_to_imdef  #     from data_tree.coconut.omni_converter import auto_img,cast_imdef_str_to_imdef,cast_imdef_to_imdef_str
    from data_tree.coconut.omni_converter import cast_imdef_to_imdef_str  #     from data_tree.coconut.omni_converter import auto_img,cast_imdef_str_to_imdef,cast_imdef_to_imdef_str
    from omni_converter.coconut.auto_data import AutoData  #     from omni_converter.coconut.auto_data import AutoData
    x = np.ones((100, 100, 3), dtype="float32")  #     x = np.ones((100,100,3),dtype="float32")
    auto_x = auto_img("numpy,float32,HW,L,0_1")(x)  # type: AutoData  #     auto_x:AutoData = auto_img("numpy,float32,HW,L,0_1")(x)
    if "__annotations__" not in _coconut.locals():  #     auto_x:AutoData = auto_img("numpy,float32,HW,L,0_1")(x)
        __annotations__ = {}  #     auto_x:AutoData = auto_img("numpy,float32,HW,L,0_1")(x)
    __annotations__["auto_x"] = 'AutoData'  #     auto_x:AutoData = auto_img("numpy,float32,HW,L,0_1")(x)
    assert (auto_x.to("numpy,float32,HW,L,0_255") == 255).all()  #     assert (auto_x.to("numpy,float32,HW,L,0_255") == 255).all()
    assert (auto_x.to(v_range="0_255") == 255).all()  #     assert (auto_x.to(v_range="0_255") == 255).all()
    _x = auto_x.to(dtype="uint8", v_range="0_255")  #     _x = auto_x.to(dtype="uint8",v_range="0_255")
    assert (_x == 255).all(), "original:{_coconut_format_0},converted:{_coconut_format_1}".format(_coconut_format_0=(x), _coconut_format_1=(_x))  #     assert (_x == 255).all(), f"original:{x},converted:{_x}"
    _x = auto_x.to(type="torch", dtype="uint8", v_range="0_255")  #     _x = auto_x.to(type="torch",dtype="uint8",v_range="0_255")
    assert (_x == 255).all(), "original:{_coconut_format_0},converted:{_coconut_format_1}".format(_coconut_format_0=(x), _coconut_format_1=(_x))  #     assert (_x == 255).all(), f"original:{x},converted:{_x}"
#logger.info(auto_x.convert(type="torch",dtype="uint8",v_range="0_255").format)
#logger.info(auto_x.converter(type="torch",dtype="uint8",v_range="0_255"))
#format = "numpy,float32,HW,L,0_1"
#n_format = cast_imdef_str_to_imdef(format)[0]
#assert format == n_format,f"{format} != {n_format}"


def test_tuple_conversion():  # def test_tuple_conversion():
    from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz  #     from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz, pix2pix_rgb_batch, TORCH_XYZ_BATCH
    from archpainter.experiments.rgba2xyz_mod.instances import pix2pix_rgb_batch  #     from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz, pix2pix_rgb_batch, TORCH_XYZ_BATCH
    from archpainter.experiments.rgba2xyz_mod.instances import TORCH_XYZ_BATCH  #     from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz, pix2pix_rgb_batch, TORCH_XYZ_BATCH
    logger.info(auto(("image,RGB,RGB", "image,RGB,RGB"))((None, None)).converter(("pix2pix_batch,nc=1", TORCH_XYZ_BATCH)))  #     logger.info(auto(("image,RGB,RGB","image,RGB,RGB"))((None,None)).converter(("pix2pix_batch,nc=1", TORCH_XYZ_BATCH)))


def test_rgb_to_yuv():  # def test_rgb_to_yuv():
    from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz  #     from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz, pix2pix_rgb_batch, TORCH_XYZ_BATCH
    from archpainter.experiments.rgba2xyz_mod.instances import pix2pix_rgb_batch  #     from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz, pix2pix_rgb_batch, TORCH_XYZ_BATCH
    from archpainter.experiments.rgba2xyz_mod.instances import TORCH_XYZ_BATCH  #     from archpainter.experiments.rgba2xyz_mod.instances import torch_xyz, pix2pix_rgb_batch, TORCH_XYZ_BATCH
    logger.info(auto("image,RGB,RGB")(None).converter("image,YCbCr,YCbCr"))  #     logger.info(auto("image,RGB,RGB")(None).converter("image,YCbCr,YCbCr"))
