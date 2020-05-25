#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xbf3e8f03

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
class END(_coconut.collections.namedtuple("END", ""), ImageDef):  # data END from ImageDef
    __slots__ = ()  # data END from ImageDef
    __ne__ = _coconut.object.__ne__  # data END from ImageDef
    def __eq__(self, other):  # data END from ImageDef
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data END from ImageDef
    def __hash__(self):  # data END from ImageDef
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data END from ImageDef

class DUMMY(_coconut.collections.namedtuple("DUMMY", ""), ImageDef):  # data DUMMY from ImageDef
    __slots__ = ()  # data DUMMY from ImageDef
    __ne__ = _coconut.object.__ne__  # data DUMMY from ImageDef
    def __eq__(self, other):  # data DUMMY from ImageDef
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data DUMMY from ImageDef
    def __hash__(self):  # data DUMMY from ImageDef
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data DUMMY from ImageDef

def dummy_rule(imdef):  # def dummy_rule(imdef):
    return [(lambda a: a, end, 1, "dummy")]  #     return [(a->a,end,1,"dummy")]

def dummy_rule2(node):  # def dummy_rule2(node):
    paths = []  #     paths = []
    _coconut_match_to = node  #     case node:
    _coconut_case_check_0 = False  #     case node:
    if (_coconut.isinstance(_coconut_match_to, DUMMY)) and (_coconut.len(_coconut_match_to) == 0):  #     case node:
        _coconut_case_check_0 = True  #     case node:
    if _coconut_case_check_0:  #     case node:
        paths += [(lambda a: END, END(), 1, "to_end")]  #             paths += [(a->END,END(),1,"to_end")]
    if not _coconut_case_check_0:  #         match _ is ImageDef:
        if _coconut.isinstance(_coconut_match_to, ImageDef):  #         match _ is ImageDef:
            _coconut_case_check_0 = True  #         match _ is ImageDef:
        if _coconut_case_check_0:  #         match _ is ImageDef:
            paths += [(lambda a: DUMMY, DUMMY(), 1, "to_dummy")]  #             paths += [(a->DUMMY,DUMMY(),1,"to_dummy")]

    return paths  #     return paths

def test_something():  # def test_something():
    def neighbors(node):  #     def neighbors(node):
        return [(lambda a: a + "a", node + "a", 1, "add_a")]  #         return [(a->a+"a",node + "a",1,"add_a")]
    def matcher(node):  #     def matcher(node):
        return node == "aaa"  #         return node == "aaa"
    def heuristics(node):  #     def heuristics(node):
        return 0  #         return 0
    (log_conversion)(astar(start="a", matcher=matcher, neighbors=neighbors, heuristics=heuristics).result)  #     astar(

def imdef_neighbors(imdef):  # def imdef_neighbors(imdef):
    return [(e.f, e.b, e.cost, e.name) for e in _edges(imdef)]  #     return [(e.f,e.b,e.cost,e.name) for e in _edges(imdef)]

def test_new_astar():  # def test_new_astar():
    (log_conversion)(astar(start=start, matcher=lambda d: d == end, neighbors=imdef_neighbors, heuristics=lambda a: 0).result)  #     astar(

def log_conversion(converter):  # def log_conversion(converter):
    path = [e.name for e in converter.edges]  #     path = [e.name for e in converter.edges]
    logger.info(path)  #     logger.info(path)



def test_astar_solver():  # def test_astar_solver():


    solver = AStarSolver(rules=[imdef_neighbors])  #     solver=AStarSolver(
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
    from data_tree.coconut.auto_data import AutoData  #     from data_tree.coconut.auto_data import AutoData
    x = np.ones((100, 100, 3), dtype="float32")  #     x = np.ones((100,100,3),dtype="float32")
    auto_x = auto_img("numpy,float32,HW,L,0_1")(x)  # type: AutoData  #     auto_x:AutoData = auto_img("numpy,float32,HW,L,0_1")(x)
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
