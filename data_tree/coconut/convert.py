#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x1cd6c178

# Compiled with Coconut version 1.6.0 [Vocational Guidance Counsellor]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop

import os as _coconut_os
import sys as _coconut_sys

_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
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
from __coconut__ import _coconut_call_set_names, _coconut, _coconut_base_compose, _coconut_get_function_match_error, \
    _coconut_sentinel, _coconut_mark_as_match

_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import numpy as np  # import numpy as np
from data_tree.coconut.visualization import infer_widget  # from data_tree.coconut.visualization import infer_widget
  # from loguru import logger
from data_tree.coconut.astar import AStarSolver  # from data_tree.coconut.astar import new_conversion,AStarSolver,NoRouteException
from data_tree.coconut.astar import NoRouteException  # from data_tree.coconut.astar import new_conversion,AStarSolver,NoRouteException
from math import sqrt  # from math import sqrt
from PIL import Image  # from PIL import Image
import torch  # import torch
import re  # import re
from frozendict import frozendict as fdict  # from frozendict import frozendict as fdict
from frozendict import frozendict  # from frozendict import frozendict

VR_0_1 = "0_1"  # VR_0_1 = "0_1"
VR_0_255 = "0_255"  # VR_0_255 = "0_255"
VR_None = "None"  # VR_None = "None"
VR_XYZ_Normalized = "XYZ_Normalized"  # VR_XYZ_Normalized = "XYZ_Normalized"
ch_splitter = re.compile("[A-Z][a-z]*").findall  # ch_splitter = re.compile("[A-Z][a-z]*").findall

class DataType(_coconut.collections.namedtuple("DataType", ())):  # data DataType
    __slots__ = ()  # data DataType
    __ne__ = _coconut.object.__ne__  # data DataType
    def __eq__(self, other):  # data DataType
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data DataType
    def __hash__(self):  # data DataType
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data DataType
    __match_args__ = ()  # data DataType
#TODO add shape information to tensorlike
#TODO add shape information to PILImage
#TODO make rules be able to handle other metadata.
#so let's assume the input state based on dict.
_coconut_call_set_names(DataType)  # data TensorLike(
class TensorLike(_coconut.collections.namedtuple("TensorLike", ('dtype', 'arrange', 'channel_repr', 'value_range')), DataType):  # data TensorLike(
    __slots__ = ()  # data TensorLike(
    __ne__ = _coconut.object.__ne__  # data TensorLike(
    def __eq__(self, other):  # data TensorLike(
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data TensorLike(
    def __hash__(self):  # data TensorLike(
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data TensorLike(
    __match_args__ = ('dtype', 'arrange', 'channel_repr', 'value_range')  # data TensorLike(
    def __new__(_coconut_cls, *_coconut_match_args, **_coconut_match_kwargs):  # data TensorLike(
        _coconut_match_check_0 = False  # data TensorLike(
        _coconut_match_set_name_dtype = _coconut_sentinel  # data TensorLike(
        _coconut_match_set_name_arrange = _coconut_sentinel  # data TensorLike(
        _coconut_match_set_name_channel_repr = _coconut_sentinel  # data TensorLike(
        _coconut_match_set_name_value_range = _coconut_sentinel  # data TensorLike(
        _coconut_FunctionMatchError = _coconut_get_function_match_error()  # data TensorLike(
        if (_coconut.len(_coconut_match_args) <= 4) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "dtype" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "arrange" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "channel_repr" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "value_range" in _coconut_match_kwargs)) == 1):  # data TensorLike(
            _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("dtype")  # data TensorLike(
            _coconut_match_temp_1 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("arrange")  # data TensorLike(
            _coconut_match_temp_2 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("channel_repr")  # data TensorLike(
            _coconut_match_temp_3 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("value_range")  # data TensorLike(
            if (_coconut.isinstance(_coconut_match_temp_0, str)) and (_coconut.isinstance(_coconut_match_temp_1, str)) and (_coconut.isinstance(_coconut_match_temp_2, str)) and (_coconut.isinstance(_coconut_match_temp_3, str)) and (not _coconut_match_kwargs):  # data TensorLike(
                _coconut_match_set_name_dtype = _coconut_match_temp_0  # data TensorLike(
                _coconut_match_set_name_arrange = _coconut_match_temp_1  # data TensorLike(
                _coconut_match_set_name_channel_repr = _coconut_match_temp_2  # data TensorLike(
                _coconut_match_set_name_value_range = _coconut_match_temp_3  # data TensorLike(
                _coconut_match_check_0 = True  # data TensorLike(
        if _coconut_match_check_0:  # data TensorLike(
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  # data TensorLike(
                dtype = _coconut_match_temp_0  # data TensorLike(
            if _coconut_match_set_name_arrange is not _coconut_sentinel:  # data TensorLike(
                arrange = _coconut_match_temp_1  # data TensorLike(
            if _coconut_match_set_name_channel_repr is not _coconut_sentinel:  # data TensorLike(
                channel_repr = _coconut_match_temp_2  # data TensorLike(
            if _coconut_match_set_name_value_range is not _coconut_sentinel:  # data TensorLike(
                value_range = _coconut_match_temp_3  # data TensorLike(

        if not _coconut_match_check_0:  # data TensorLike(
            raise _coconut_FunctionMatchError('data TensorLike(     dtype is str,     arrange is str,     channel_repr is str,     value_range is str) from DataType:', _coconut_match_args)  # data TensorLike(

        return _coconut.tuple.__new__(_coconut_cls, (dtype, arrange, channel_repr, value_range))  # data TensorLike(
    def __repr__(self):  #     def __repr__(self):
        return ("Tensor({_coconut_format_0}|{_coconut_format_1}|{_coconut_format_2}|{_coconut_format_3}|{_coconut_format_4})".format(_coconut_format_0=(self.data_type), _coconut_format_1=(self.dtype), _coconut_format_2=(self.arrange), _coconut_format_3=(self.channel_repr), _coconut_format_4=(self.value_range)))  #         return f"Tensor({self.data_type}|{self.dtype}|{self.arrange}|{self.channel_repr}|{self.value_range})"

_coconut_call_set_names(TensorLike)  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
class Numpy(_coconut.collections.namedtuple("Numpy", ('dtype', 'arrange', 'channel_repr', 'value_range')), TensorLike):  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
    __slots__ = ()  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
    __ne__ = _coconut.object.__ne__  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __eq__(self, other):  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __hash__(self):  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
    __match_args__ = ('dtype', 'arrange', 'channel_repr', 'value_range')  # data Numpy(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __new__(cls, *args):  #     def __new__(cls,*args):
        return (makedata(cls, *args))  #         return makedata(cls,*args)
    def __repr__(self):  #     def __repr__(self):
        return ("Numpy({_coconut_format_0},{_coconut_format_1},{_coconut_format_2},{_coconut_format_3})".format(_coconut_format_0=(self.dtype), _coconut_format_1=(self.arrange), _coconut_format_2=(self.channel_repr), _coconut_format_3=(self.value_range)))  #         return f"Numpy({self.dtype},{self.arrange},{self.channel_repr},{self.value_range})"

_coconut_call_set_names(Numpy)  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
class Torch(_coconut.collections.namedtuple("Torch", ('dtype', 'arrange', 'channel_repr', 'value_range')), TensorLike):  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
    __slots__ = ()  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
    __ne__ = _coconut.object.__ne__  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __eq__(self, other):  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __hash__(self):  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
    __match_args__ = ('dtype', 'arrange', 'channel_repr', 'value_range')  # data Torch(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __new__(cls, *args):  #     def __new__(cls,*args):
        return (makedata(cls, *args))  #         return makedata(cls,*args)
    def __repr__(self):  #     def __repr__(self):
        return ("Torch({_coconut_format_0},{_coconut_format_1},{_coconut_format_2},{_coconut_format_3})".format(_coconut_format_0=(self.dtype), _coconut_format_1=(self.arrange), _coconut_format_2=(self.channel_repr), _coconut_format_3=(self.value_range)))  #         return f"Torch({self.dtype},{self.arrange},{self.channel_repr},{self.value_range})"

_coconut_call_set_names(Torch)  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
class Hdf5(_coconut.collections.namedtuple("Hdf5", ('dtype', 'arrange', 'channel_repr', 'value_range')), TensorLike):  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
    __slots__ = ()  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
    __ne__ = _coconut.object.__ne__  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __eq__(self, other):  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __hash__(self):  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
    __match_args__ = ('dtype', 'arrange', 'channel_repr', 'value_range')  # data Hdf5(dtype,arrange,channel_repr,value_range) from TensorLike:
    def __new__(cls, *args):  #     def __new__(cls,*args):
        return (makedata(cls, *args))  #         return makedata(cls,*args)
    def __repr__(self):  #     def __repr__(self):
        return ("Hdf5({_coconut_format_0},{_coconut_format_1},{_coconut_format_2},{_coconut_format_3})".format(_coconut_format_0=(self.dtype), _coconut_format_1=(self.arrange), _coconut_format_2=(self.channel_repr), _coconut_format_3=(self.value_range)))  #         return f"Hdf5({self.dtype},{self.arrange},{self.channel_repr},{self.value_range})"

_coconut_call_set_names(Hdf5)  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
class PILImages(_coconut.collections.namedtuple("PILImages", ('mode', 'channel_repr')), DataType):  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
    __slots__ = ()  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
    __ne__ = _coconut.object.__ne__  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
    def __eq__(self, other):  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
    def __hash__(self):  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
    __match_args__ = ('mode', 'channel_repr')  # represents iterable of PIL.Images  # data PILImages(mode,channel_repr) from DataType: # represents iterable of PIL.Images
    def __repr__(self):  #     def __repr__(self):
        return ("PILImages({_coconut_format_0},{_coconut_format_1})".format(_coconut_format_0=(self.mode), _coconut_format_1=(self.channel_repr)))  #         return f"PILImages({self.mode},{self.channel_repr})"
_coconut_call_set_names(PILImages)  # data PILImage(mode,channel_repr) from DataType:
class PILImage(_coconut.collections.namedtuple("PILImage", ('mode', 'channel_repr')), DataType):  # data PILImage(mode,channel_repr) from DataType:
    __slots__ = ()  # data PILImage(mode,channel_repr) from DataType:
    __ne__ = _coconut.object.__ne__  # data PILImage(mode,channel_repr) from DataType:
    def __eq__(self, other):  # data PILImage(mode,channel_repr) from DataType:
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data PILImage(mode,channel_repr) from DataType:
    def __hash__(self):  # data PILImage(mode,channel_repr) from DataType:
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data PILImage(mode,channel_repr) from DataType:
    __match_args__ = ('mode', 'channel_repr')  # data PILImage(mode,channel_repr) from DataType:
    def __repr__(self):  #     def __repr__(self):
        return ("PILImage({_coconut_format_0},{_coconut_format_1})".format(_coconut_format_0=(self.mode), _coconut_format_1=(self.channel_repr)))  #         return f"PILImage({self.mode},{self.channel_repr})"

_coconut_call_set_names(PILImage)  # data ImageDef(data_type is DataType,meta is frozendict):
class ImageDef(_coconut.collections.namedtuple("ImageDef", ('data_type', 'meta'))):  # data ImageDef(data_type is DataType,meta is frozendict):
    __slots__ = ()  # data ImageDef(data_type is DataType,meta is frozendict):
    __ne__ = _coconut.object.__ne__  # data ImageDef(data_type is DataType,meta is frozendict):
    def __eq__(self, other):  # data ImageDef(data_type is DataType,meta is frozendict):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data ImageDef(data_type is DataType,meta is frozendict):
    def __hash__(self):  # data ImageDef(data_type is DataType,meta is frozendict):
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data ImageDef(data_type is DataType,meta is frozendict):
    __match_args__ = ('data_type', 'meta')  # data ImageDef(data_type is DataType,meta is frozendict):
    def __new__(_coconut_cls, *_coconut_match_args, **_coconut_match_kwargs):  # data ImageDef(data_type is DataType,meta is frozendict):
        _coconut_match_check_1 = False  # data ImageDef(data_type is DataType,meta is frozendict):
        _coconut_match_set_name_data_type = _coconut_sentinel  # data ImageDef(data_type is DataType,meta is frozendict):
        _coconut_match_set_name_meta = _coconut_sentinel  # data ImageDef(data_type is DataType,meta is frozendict):
        _coconut_FunctionMatchError = _coconut_get_function_match_error()  # data ImageDef(data_type is DataType,meta is frozendict):
        if (_coconut.len(_coconut_match_args) <= 2) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "data_type" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "meta" in _coconut_match_kwargs)) == 1):  # data ImageDef(data_type is DataType,meta is frozendict):
            _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("data_type")  # data ImageDef(data_type is DataType,meta is frozendict):
            _coconut_match_temp_1 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("meta")  # data ImageDef(data_type is DataType,meta is frozendict):
            if (_coconut.isinstance(_coconut_match_temp_0, DataType)) and (_coconut.isinstance(_coconut_match_temp_1, frozendict)) and (not _coconut_match_kwargs):  # data ImageDef(data_type is DataType,meta is frozendict):
                _coconut_match_set_name_data_type = _coconut_match_temp_0  # data ImageDef(data_type is DataType,meta is frozendict):
                _coconut_match_set_name_meta = _coconut_match_temp_1  # data ImageDef(data_type is DataType,meta is frozendict):
                _coconut_match_check_1 = True  # data ImageDef(data_type is DataType,meta is frozendict):
        if _coconut_match_check_1:  # data ImageDef(data_type is DataType,meta is frozendict):
            if _coconut_match_set_name_data_type is not _coconut_sentinel:  # data ImageDef(data_type is DataType,meta is frozendict):
                data_type = _coconut_match_temp_0  # data ImageDef(data_type is DataType,meta is frozendict):
            if _coconut_match_set_name_meta is not _coconut_sentinel:  # data ImageDef(data_type is DataType,meta is frozendict):
                meta = _coconut_match_temp_1  # data ImageDef(data_type is DataType,meta is frozendict):

        if not _coconut_match_check_1:  # data ImageDef(data_type is DataType,meta is frozendict):
            raise _coconut_FunctionMatchError('data ImageDef(data_type is DataType,meta is frozendict):', _coconut_match_args)  # data ImageDef(data_type is DataType,meta is frozendict):

        return _coconut.tuple.__new__(_coconut_cls, (data_type, meta))  # data ImageDef(data_type is DataType,meta is frozendict):
    def __repr__(self):  #     def __repr__(self):
        if self.meta:  #         if self.meta:
            return ("ImageDef({_coconut_format_0}|{_coconut_format_1})".format(_coconut_format_0=(self.data_type), _coconut_format_1=(self.meta)))  #             return f"ImageDef({self.data_type}|{self.meta})"
        else:  #         else:
            return ("ImageDef({_coconut_format_0})".format(_coconut_format_0=(self.data_type)))  #             return f"ImageDef({self.data_type})"

_coconut_call_set_names(ImageDef)  # DTYPES={"float32","float64","int32","int64","uint8","bool"}
DTYPES = {"float32", "float64", "int32", "int64", "uint8", "bool"}  # DTYPES={"float32","float64","int32","int64","uint8","bool"}
class DataEdge(_coconut.collections.namedtuple("DataEdge", ('a', 'b', 'f', 'cost', 'name', 'meta_shifter'))):  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
    __slots__ = ()  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
    __ne__ = _coconut.object.__ne__  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
    def __eq__(self, other):  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
    def __hash__(self):  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
    __match_args__ = ('a', 'b', 'f', 'cost', 'name', 'meta_shifter')  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
    def __new__(_coconut_cls, *_coconut_match_args, **_coconut_match_kwargs):  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_match_check_2 = False  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_match_set_name_a = _coconut_sentinel  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_match_set_name_b = _coconut_sentinel  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_match_set_name_f = _coconut_sentinel  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_match_set_name_cost = _coconut_sentinel  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_match_set_name_name = _coconut_sentinel  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_match_set_name_meta_shifter = _coconut_sentinel  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        _coconut_FunctionMatchError = _coconut_get_function_match_error()  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        if (_coconut.len(_coconut_match_args) <= 6) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "a" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "b" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "f" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "cost" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "name" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 5, "meta_shifter" in _coconut_match_kwargs)) <= 1):  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("a")  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            _coconut_match_temp_1 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("b")  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            _coconut_match_temp_2 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("f")  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            _coconut_match_temp_3 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("cost")  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            _coconut_match_temp_4 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("name")  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            _coconut_match_temp_5 = _coconut_match_args[5] if _coconut.len(_coconut_match_args) > 5 else _coconut_match_kwargs.pop("meta_shifter") if "meta_shifter" in _coconut_match_kwargs else lambda x: x  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            if (_coconut.isinstance(_coconut_match_temp_0, DataType)) and (_coconut.isinstance(_coconut_match_temp_1, DataType)) and (_coconut.isinstance(_coconut_match_temp_3, int)) and (_coconut.isinstance(_coconut_match_temp_4, str)) and (not _coconut_match_kwargs):  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                _coconut_match_set_name_a = _coconut_match_temp_0  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                _coconut_match_set_name_b = _coconut_match_temp_1  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                _coconut_match_set_name_f = _coconut_match_temp_2  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                _coconut_match_set_name_cost = _coconut_match_temp_3  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                _coconut_match_set_name_name = _coconut_match_temp_4  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                _coconut_match_set_name_meta_shifter = _coconut_match_temp_5  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                _coconut_match_check_2 = True  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
        if _coconut_match_check_2:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            if _coconut_match_set_name_a is not _coconut_sentinel:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                a = _coconut_match_temp_0  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            if _coconut_match_set_name_b is not _coconut_sentinel:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                b = _coconut_match_temp_1  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            if _coconut_match_set_name_f is not _coconut_sentinel:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                f = _coconut_match_temp_2  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            if _coconut_match_set_name_cost is not _coconut_sentinel:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                cost = _coconut_match_temp_3  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            if _coconut_match_set_name_name is not _coconut_sentinel:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                name = _coconut_match_temp_4  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            if _coconut_match_set_name_meta_shifter is not _coconut_sentinel:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
                meta_shifter = _coconut_match_temp_5  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):

        if not _coconut_match_check_2:  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
            raise _coconut_FunctionMatchError('data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):', _coconut_match_args)  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):

        return _coconut.tuple.__new__(_coconut_cls, (a, b, f, cost, name, meta_shifter))  # data DataEdge(a is DataType,b is DataType,f,cost is int,name is str,meta_shifter=x->x):
    def to_edge(self, src: 'ImageDef'):  #     def to_edge(self,src:ImageDef):
        try:  #         try:
            new_meta = self.meta_shifter(src.meta)  #             new_meta = self.meta_shifter(src.meta)
            return (Edge(src, ImageDef(self.b, new_meta), self.f, self.cost, self.name))  #             return Edge(src,ImageDef(self.b,new_meta),self.f,self.cost,self.name)
            _coconut_case_match_to_0 = (self.a, src.meta, self.b, new_meta)  #             case (self.a,src.meta,self.b,new_meta):
            _coconut_case_match_check_0 = False  #             case (self.a,src.meta,self.b,new_meta):
            _coconut_match_set_name_arng1 = _coconut_sentinel  #             case (self.a,src.meta,self.b,new_meta):
            _coconut_match_set_name_s1 = _coconut_sentinel  #             case (self.a,src.meta,self.b,new_meta):
            _coconut_match_set_name_arng2 = _coconut_sentinel  #             case (self.a,src.meta,self.b,new_meta):
            _coconut_match_set_name_s2 = _coconut_sentinel  #             case (self.a,src.meta,self.b,new_meta):
            if (_coconut.isinstance(_coconut_case_match_to_0, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_0) == 4) and (_coconut.isinstance(_coconut_case_match_to_0[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_0[0]) >= 2) and (_coconut.isinstance(_coconut_case_match_to_0[1], _coconut.abc.Mapping)) and (_coconut.len(_coconut_case_match_to_0[1]) == 1) and (_coconut.isinstance(_coconut_case_match_to_0[2], TensorLike)) and (_coconut.len(_coconut_case_match_to_0[2]) >= 2) and (_coconut.isinstance(_coconut_case_match_to_0[3], _coconut.abc.Mapping)) and (_coconut.len(_coconut_case_match_to_0[3]) == 1):  #             case (self.a,src.meta,self.b,new_meta):
                _coconut_match_set_name_arng1 = _coconut_case_match_to_0[0][1]  #             case (self.a,src.meta,self.b,new_meta):
                _coconut_match_temp_0 = _coconut_case_match_to_0[1].get("shape", _coconut_sentinel)  #             case (self.a,src.meta,self.b,new_meta):
                _coconut_match_set_name_arng2 = _coconut_case_match_to_0[2][1]  #             case (self.a,src.meta,self.b,new_meta):
                _coconut_match_temp_1 = _coconut_case_match_to_0[3].get("shape", _coconut_sentinel)  #             case (self.a,src.meta,self.b,new_meta):
                if (_coconut_match_temp_0 is not _coconut_sentinel) and (_coconut_match_temp_1 is not _coconut_sentinel):  #             case (self.a,src.meta,self.b,new_meta):
                    _coconut_match_set_name_s1 = _coconut_match_temp_0  #             case (self.a,src.meta,self.b,new_meta):
                    _coconut_match_set_name_s2 = _coconut_match_temp_1  #             case (self.a,src.meta,self.b,new_meta):
                    _coconut_case_match_check_0 = True  #             case (self.a,src.meta,self.b,new_meta):
            if _coconut_case_match_check_0:  #             case (self.a,src.meta,self.b,new_meta):
                if _coconut_match_set_name_arng1 is not _coconut_sentinel:  #             case (self.a,src.meta,self.b,new_meta):
                    arng1 = _coconut_case_match_to_0[0][1]  #             case (self.a,src.meta,self.b,new_meta):
                if _coconut_match_set_name_s1 is not _coconut_sentinel:  #             case (self.a,src.meta,self.b,new_meta):
                    s1 = _coconut_match_temp_0  #             case (self.a,src.meta,self.b,new_meta):
                if _coconut_match_set_name_arng2 is not _coconut_sentinel:  #             case (self.a,src.meta,self.b,new_meta):
                    arng2 = _coconut_case_match_to_0[2][1]  #             case (self.a,src.meta,self.b,new_meta):
                if _coconut_match_set_name_s2 is not _coconut_sentinel:  #             case (self.a,src.meta,self.b,new_meta):
                    s2 = _coconut_match_temp_1  #             case (self.a,src.meta,self.b,new_meta):
            if _coconut_case_match_check_0:  #             case (self.a,src.meta,self.b,new_meta):
                assert len(arng2) == len(s2)  #                     assert len(arng2) == len(s2)
                assert len(arng1) == len(s1)  #                     assert len(arng1) == len(s1)
            if src.meta != new_meta:  #             if src.meta != new_meta:
                if "shape" in src.meta and "shape" in new_meta:  #                 if "shape" in src.meta and "shape" in new_meta:
                    if len(src.meta["shape"]) > len(new_meta["shape"]):  #                     if len(src.meta["shape"]) > len(new_meta["shape"]):
                        logger.info("{_coconut_format_0}:\n{_coconut_format_1}->{_coconut_format_2}".format(_coconut_format_0=(self.name), _coconut_format_1=(src.meta['shape']), _coconut_format_2=(new_meta['shape'])))  #                         logger.info(f"{self.name}:\n{src.meta['shape']}->{new_meta['shape']}")
        except Exception as e:  #         except Exception as e:
            from IPython import embed  #             from IPython import embed
            embed()  #             embed()
            raise e  #             raise e
_coconut_call_set_names(DataEdge)  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
class Edge(_coconut.collections.namedtuple("Edge", ('a', 'b', 'f', 'cost', 'name'))):  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
    __slots__ = ()  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
    __ne__ = _coconut.object.__ne__  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
    def __eq__(self, other):  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
    def __hash__(self):  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
    __match_args__ = ('a', 'b', 'f', 'cost', 'name')  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
    def __new__(_coconut_cls, *_coconut_match_args, **_coconut_match_kwargs):  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        _coconut_match_check_3 = False  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        _coconut_match_set_name_a = _coconut_sentinel  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        _coconut_match_set_name_b = _coconut_sentinel  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        _coconut_match_set_name_f = _coconut_sentinel  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        _coconut_match_set_name_cost = _coconut_sentinel  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        _coconut_match_set_name_name = _coconut_sentinel  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        _coconut_FunctionMatchError = _coconut_get_function_match_error()  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        if (_coconut.len(_coconut_match_args) <= 5) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "a" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "b" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "f" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 3, "cost" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 4, "name" in _coconut_match_kwargs)) <= 1):  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("a")  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            _coconut_match_temp_1 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("b")  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            _coconut_match_temp_2 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("f")  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            _coconut_match_temp_3 = _coconut_match_args[3] if _coconut.len(_coconut_match_args) > 3 else _coconut_match_kwargs.pop("cost")  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            _coconut_match_temp_4 = _coconut_match_args[4] if _coconut.len(_coconut_match_args) > 4 else _coconut_match_kwargs.pop("name") if "name" in _coconut_match_kwargs else "undefined"  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            if (_coconut.isinstance(_coconut_match_temp_0, ImageDef)) and (_coconut.isinstance(_coconut_match_temp_1, ImageDef)) and (_coconut.isinstance(_coconut_match_temp_3, int)) and (not _coconut_match_kwargs):  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                _coconut_match_set_name_a = _coconut_match_temp_0  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                _coconut_match_set_name_b = _coconut_match_temp_1  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                _coconut_match_set_name_f = _coconut_match_temp_2  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                _coconut_match_set_name_cost = _coconut_match_temp_3  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                _coconut_match_set_name_name = _coconut_match_temp_4  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                _coconut_match_check_3 = True  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
        if _coconut_match_check_3:  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            if _coconut_match_set_name_a is not _coconut_sentinel:  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                a = _coconut_match_temp_0  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            if _coconut_match_set_name_b is not _coconut_sentinel:  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                b = _coconut_match_temp_1  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            if _coconut_match_set_name_f is not _coconut_sentinel:  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                f = _coconut_match_temp_2  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            if _coconut_match_set_name_cost is not _coconut_sentinel:  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                cost = _coconut_match_temp_3  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            if _coconut_match_set_name_name is not _coconut_sentinel:  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
                name = _coconut_match_temp_4  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):

        if not _coconut_match_check_3:  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
            raise _coconut_FunctionMatchError('data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):', _coconut_match_args)  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):

        return _coconut.tuple.__new__(_coconut_cls, (a, b, f, cost, name))  # data Edge(a is ImageDef,b is ImageDef,f,cost is int,name="undefined"):
    def __repr__(self):  #     def __repr__(self):
        return ("{_coconut_format_0} \t-> {_coconut_format_1}\t-> {_coconut_format_2}".format(_coconut_format_0=(self.a), _coconut_format_1=(self.name), _coconut_format_2=(self.b)))  #         return f"{self.a} \t-> {self.name}\t-> {self.b}"
_coconut_call_set_names(Edge)  # from typing import List
from typing import List  # from typing import List

#純粋にエッジだけを考えると、どうしても組み合わせ爆発になる。目的地を知っていれば削減できる。

# 各imdefのNodeベースでエッジを定義するのか。それとも。エッジのルールを網羅するのか。
# オペレーターのほうが数が少ないので、オペレーターだけ定義したい。
# List up operators.
# operator definition: ImageDef->List[DataEdge]
# 1. change data_type
# 2. change dtype
# 3. change arrange
# 4. change ch_repr
# 5. select channel
def to_imagedef(f):  # def to_imagedef(f):
    def _inner(imdef: 'ImageDef'):  #     def _inner(imdef:ImageDef):
        try:  #         try:
#logger.debug(type(imdef))
            if (isinstance)(imdef, ImageDef) and len(imdef) >= 1 and (hasattr)(imdef, "data_type"):  #             if imdef `isinstance` ImageDef and len(imdef) >= 1 and imdef `hasattr` "data_type":
#if imdef `isinstance` ImageDef:
                edges = f(imdef.data_type)  #                 edges = f(imdef.data_type)
                if edges is not None:  #                 if edges is not None:
                    return ([e.to_edge(imdef) for e in edges])  #                     return [e.to_edge(imdef) for e in edges]
                else:  #                 else:
                    return ([])  #                     return []
            else:  #             else:
                return ([])  #                 return []
        except Exception as e:  #         except Exception as e:
            logger.warning("unknown error...imdef:{_coconut_format_0}".format(_coconut_format_0=(imdef)))  #             logger.warning(f"unknown error...imdef:{imdef}")
            logger.warning("{_coconut_format_0} has attr causes exception?".format(_coconut_format_0=(imdef)))  #             logger.warning(f"{imdef} has attr causes exception?")
            logger.warning("{_coconut_format_0}".format(_coconut_format_0=(hasattr(imdef, 'data_type'))))  #             logger.warning(f"{hasattr(imdef,'data_type')}")
            raise e  #             raise e
    return (_inner)  #     return _inner

# there are kinds of rules
# 1. doesnt care about shape, and no shape change
# 2. doesnt care shape, but shape changes
# 3. depends on input shape but shape doesn't change
# 4. depends on input and output shape changes
# well ,just give function a shape and return nop for shape?
# what if some other metadata is added?
# where should I store shape?
# If I change the signature, I have modify them all... for match syntax
# I just


@to_imagedef  # doesn't change shape, so no touch  # @to_imagedef # doesn't change shape, so no touch
def to_PILImages(imdef: 'ImageDef') -> '_coconut.typing.Sequence[Edge]':  # def to_PILImages(imdef:ImageDef)->Edge[]:
    _coconut_case_match_to_1 = imdef  #     case imdef:
    _coconut_case_match_check_1 = False  #     case imdef:
    _coconut_match_set_name_c_repr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_1, Numpy)) and (_coconut.len(_coconut_case_match_to_1) == 4) and (_coconut_case_match_to_1[0] == "uint8") and (_coconut_case_match_to_1[1] == "BHWC") and (_coconut_case_match_to_1[3] == VR_0_255):  #     case imdef:
        _coconut_match_set_name_c_repr = _coconut_case_match_to_1[2]  #     case imdef:
        _coconut_case_match_check_1 = True  #     case imdef:
    if _coconut_case_match_check_1:  #     case imdef:
        if _coconut_match_set_name_c_repr is not _coconut_sentinel:  #     case imdef:
            c_repr = _coconut_case_match_to_1[2]  #     case imdef:
    if _coconut_case_match_check_1 and not (len(ch_splitter(c_repr)) in (3, 4)):  #     case imdef:
        _coconut_case_match_check_1 = False  #     case imdef:
    if _coconut_case_match_check_1:  #     case imdef:
        return ([DataEdge(imdef, PILImages(c_repr, c_repr), lambda ary: [(Image.fromarray)(img) for img in ary], 2, name="numpy batch {_coconut_format_0} to Images".format(_coconut_format_0=(c_repr))), ])  #             return [DataEdge(imdef,PILImages(c_repr,c_repr),ary -> [(Image.fromarray)(img) for img in ary],2,name=f"numpy batch {c_repr} to Images")]
    if not _coconut_case_match_check_1:  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
        _coconut_match_set_name_c_repr = _coconut_sentinel  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
        if (_coconut.isinstance(_coconut_case_match_to_1, Numpy)) and (_coconut.len(_coconut_case_match_to_1) == 4) and (_coconut_case_match_to_1[0] == "uint8") and (_coconut_case_match_to_1[1] == "BHW") and (_coconut_case_match_to_1[3] == VR_0_255):  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
            _coconut_match_set_name_c_repr = _coconut_case_match_to_1[2]  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
            _coconut_case_match_check_1 = True  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
        if _coconut_case_match_check_1:  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
            if _coconut_match_set_name_c_repr is not _coconut_sentinel:  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
                c_repr = _coconut_case_match_to_1[2]  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
        if _coconut_case_match_check_1:  #         match Numpy("uint8","BHW",c_repr,=VR_0_255):
            return ([DataEdge(imdef, PILImages("L", c_repr), lambda ary: [(_coconut_base_compose(Image.fromarray, (_coconut.operator.methodcaller("convert", "L"), 0)))(img) for img in ary], 2, name="numpy batch to images"), ])  #             return [DataEdge(imdef,PILImages("L",c_repr),ary -> [(Image.fromarray ..> .convert("L"))(img) for img in ary],2,name="numpy batch to images")]
    if not _coconut_case_match_check_1:  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
        _coconut_match_set_name_c_repr = _coconut_sentinel  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
        if (_coconut.isinstance(_coconut_case_match_to_1, Numpy)) and (_coconut.len(_coconut_case_match_to_1) == 4) and (_coconut_case_match_to_1[0] == "uint8") and (_coconut_case_match_to_1[1] == "HW") and (_coconut_case_match_to_1[3] == VR_0_255):  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
            _coconut_match_set_name_c_repr = _coconut_case_match_to_1[2]  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
            _coconut_case_match_check_1 = True  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
        if _coconut_case_match_check_1:  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
            if _coconut_match_set_name_c_repr is not _coconut_sentinel:  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
                c_repr = _coconut_case_match_to_1[2]  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
        if _coconut_case_match_check_1:  #         match Numpy("uint8","HW",c_repr,=VR_0_255):
            return ([DataEdge(imdef, PILImage("L", c_repr), _coconut_base_compose(Image.fromarray, (_coconut.operator.methodcaller("convert", "L"), 0)), 2, name="numpy HW to PIL Image"), ])  #             return [DataEdge(imdef,PILImage("L",c_repr), Image.fromarray ..> .convert("L"),2,name="numpy HW to PIL Image")]
    return ([])  #     return []
@to_imagedef  # @to_imagedef
def to_numpy(imdef: 'ImageDef') -> 'List[DataEdge]':  # def to_numpy(imdef:ImageDef)->List[DataEdge]:
    _coconut_case_match_to_2 = imdef  #     case imdef:
    _coconut_case_match_check_2 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_2, Torch)) and (_coconut.len(_coconut_case_match_to_2) == 4):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_2[0]  #     case imdef:
        _coconut_match_set_name_arng = _coconut_case_match_to_2[1]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_2[2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_2[3]  #     case imdef:
        _coconut_case_match_check_2 = True  #     case imdef:
    if _coconut_case_match_check_2:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_2[0]  #     case imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     case imdef:
            arng = _coconut_case_match_to_2[1]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_2[2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_2[3]  #     case imdef:
    if _coconut_case_match_check_2:  #     case imdef:
        return ([DataEdge(imdef, Numpy(dtype, arng, ch_repr, vr), (_coconut_base_compose(_coconut.operator.methodcaller("detach"), (_coconut.operator.methodcaller("cpu"), 0), (_coconut.operator.methodcaller("numpy"), 0))), 1, name="torch_to_numpy"), ])  #             return [DataEdge(imdef,
    if not _coconut_case_match_check_2:  #                          Numpy(dtype  ,arng ,ch_repr,vr),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                          Numpy(dtype  ,arng ,ch_repr,vr),
        if (_coconut.isinstance(_coconut_case_match_to_2, PILImage)) and (_coconut.len(_coconut_case_match_to_2) == 2) and (_coconut_case_match_to_2[0] == "RGB"):  #                          Numpy(dtype  ,arng ,ch_repr,vr),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_2[1]  #                          Numpy(dtype  ,arng ,ch_repr,vr),
            _coconut_case_match_check_2 = True  #                          Numpy(dtype  ,arng ,ch_repr,vr),
        if _coconut_case_match_check_2:  #                          Numpy(dtype  ,arng ,ch_repr,vr),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                          Numpy(dtype  ,arng ,ch_repr,vr),
                ch_repr = _coconut_case_match_to_2[1]  #                          Numpy(dtype  ,arng ,ch_repr,vr),
        if _coconut_case_match_check_2:  #                          Numpy(dtype  ,arng ,ch_repr,vr),
            return ([DataEdge(imdef, Numpy("uint8", "HWC", ch_repr, VR_0_255), np.array, 1, name="image_to_numpy"), ])  #                     return [DataEdge(imdef,
    if not _coconut_case_match_check_2:  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
        if (_coconut.isinstance(_coconut_case_match_to_2, PILImage)) and (_coconut.len(_coconut_case_match_to_2) == 2) and (_coconut_case_match_to_2[0] == "L"):  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_2[1]  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
            _coconut_case_match_check_2 = True  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
        if _coconut_case_match_check_2:  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
                ch_repr = _coconut_case_match_to_2[1]  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
        if _coconut_case_match_check_2:  #                                 Numpy("uint8","HWC",ch_repr,VR_0_255),
            return ([DataEdge(imdef, Numpy("uint8", "HW", ch_repr, VR_0_255), np.array, 1, name="image_to_numpy"), ])  #                     return [DataEdge(imdef,
    if not _coconut_case_match_check_2:  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
        if (_coconut.isinstance(_coconut_case_match_to_2, PILImage)) and (_coconut.len(_coconut_case_match_to_2) == 2) and (_coconut_case_match_to_2[0] == "YCbCr"):  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_2[1]  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
            _coconut_case_match_check_2 = True  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
        if _coconut_case_match_check_2:  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
                ch_repr = _coconut_case_match_to_2[1]  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
        if _coconut_case_match_check_2:  #                                 Numpy("uint8","HW",ch_repr,VR_0_255),
            return ([DataEdge(imdef, Numpy("uint8", "HWC", ch_repr, VR_0_255), np.array, 1, name="YCbCr image to numpy"), ])  #                     return [DataEdge(imdef,

    if not _coconut_case_match_check_2:  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
        _coconut_match_set_name_ch_repr = _coconut_sentinel  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
        if (_coconut.isinstance(_coconut_case_match_to_2, PILImages)) and (_coconut.len(_coconut_case_match_to_2) == 2) and (_coconut_case_match_to_2[0] == "L"):  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_2[1]  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
            _coconut_case_match_check_2 = True  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
        if _coconut_case_match_check_2:  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
                ch_repr = _coconut_case_match_to_2[1]  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
        if _coconut_case_match_check_2:  # A grayscale Image becomes a numpy array  #         match PILImages("L",ch_repr): # A grayscale Image becomes a numpy array
            return ([DataEdge(imdef, Numpy("uint8", "BHW", ch_repr, VR_0_255), (_coconut_base_compose(_coconut.functools.partial(fmap, np.array), (np.array, 0))), 1, name="image_to_numpy"), ])  #             return [DataEdge(imdef,
    if not _coconut_case_match_check_2:  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
        _coconut_match_set_name_mode = _coconut_sentinel  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
        if (_coconut.isinstance(_coconut_case_match_to_2, PILImages)) and (_coconut.len(_coconut_case_match_to_2) == 2):  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
            _coconut_match_set_name_mode = _coconut_case_match_to_2[0]  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_2[1]  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
            _coconut_case_match_check_2 = True  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
        if _coconut_case_match_check_2:  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
            if _coconut_match_set_name_mode is not _coconut_sentinel:  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
                mode = _coconut_case_match_to_2[0]  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
                ch_repr = _coconut_case_match_to_2[1]  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
        if _coconut_case_match_check_2:  #                          Numpy("uint8","BHW",ch_repr,VR_0_255),
            return ([DataEdge(imdef, Numpy("uint8", "BHWC", ch_repr, VR_0_255), (_coconut_base_compose(_coconut.functools.partial(fmap, np.array), (np.array, 0))), 1, name="image_to_numpy"), ])  #             return [DataEdge(imdef,
    return ([])  #     return []
@to_imagedef  # @to_imagedef
def to_torch(imdef):  # def to_torch(imdef):
    import torch  #     import torch
    _coconut_case_match_to_3 = imdef  #     case imdef:
    _coconut_case_match_check_3 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_3, Numpy)) and (_coconut.len(_coconut_case_match_to_3) == 4):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_3[0]  #     case imdef:
        _coconut_match_set_name_arng = _coconut_case_match_to_3[1]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_3[2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_3[3]  #     case imdef:
        _coconut_case_match_check_3 = True  #     case imdef:
    if _coconut_case_match_check_3:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_3[0]  #     case imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     case imdef:
            arng = _coconut_case_match_to_3[1]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_3[2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_3[3]  #     case imdef:
    if _coconut_case_match_check_3:  #     case imdef:
        return ([DataEdge(imdef, Torch(dtype, arng, ch_repr, vr), torch.from_numpy, 2, name="to_torch"), ])  #             return [DataEdge(imdef,Torch(dtype,arng,ch_repr,vr),torch.from_numpy,2,name="to_torch")]
    return ([])  #     return []
@to_imagedef  # @to_imagedef
def change_dtype(imdef: 'ImageDef'):  # TODO match value range to dtype with bool type  # def change_dtype(imdef:ImageDef):# TODO match value range to dtype with bool type
    _coconut_case_match_to_4 = imdef  #     case imdef:
    _coconut_case_match_check_4 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_4, Numpy)) and (_coconut.len(_coconut_case_match_to_4) == 4):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_4[0]  #     case imdef:
        _coconut_match_set_name_arng = _coconut_case_match_to_4[1]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_4[2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_4[3]  #     case imdef:
        _coconut_case_match_check_4 = True  #     case imdef:
    if _coconut_case_match_check_4:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_4[0]  #     case imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     case imdef:
            arng = _coconut_case_match_to_4[1]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_4[2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_4[3]  #     case imdef:
    if _coconut_case_match_check_4:  #     case imdef:
        return ([DataEdge(imdef, imdef.__class__(_dtype, arng, ch_repr, vr), _coconut.operator.methodcaller("astype", _dtype), 1, name="{_coconut_format_0} to {_coconut_format_1}".format(_coconut_format_0=(dtype), _coconut_format_1=(_dtype))) for _dtype in DTYPES if _dtype != dtype])  #             return [DataEdge(
    return ([])  #     return []

# SHAPE SHIFTING RULES


def ss_to_ms(ss, meta):  # def ss_to_ms(ss,meta):
    if "shape" in meta:  #     if "shape" in meta:
        shape = meta["shape"]  #         shape=meta["shape"]
        new_shape = ss(shape)  #         new_shape = ss(shape)
        return (fdict({**meta, "shape": new_shape}))  #         return fdict({**meta,"shape":new_shape})
    return (meta)  #     return meta

ms_0231 = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], s[2], s[3], s[1])))  # ms_0231 = (s->(s[0],s[2],s[3],s[1])) |> ss_to_ms$
ms_0312 = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], s[3], s[1], s[2])))  # ms_0312 = (s->(s[0],s[3],s[1],s[2])) |> ss_to_ms$

def change_arng(imdef):  # def change_arng(imdef):
    _coconut_case_match_to_5 = imdef  #     case imdef:
    _coconut_case_match_check_5 = False  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_5, Numpy)) and (_coconut.len(_coconut_case_match_to_5) == 4) and (_coconut_case_match_to_5[1] == "BCHW"):  #     case imdef:
        _coconut_case_match_check_5 = True  #     case imdef:
    if _coconut_case_match_check_5:  #     case imdef:
        return ([(_coconut.operator.methodcaller("transpose", 0, 2, 3, 1), "BHWC", ms_0231), ])  #             return [(.transpose(0,2,3,1),"BHWC",ms_0231)]
    if not _coconut_case_match_check_5:  #         match Numpy(_,"BHWC",_,_):
        if (_coconut.isinstance(_coconut_case_match_to_5, Numpy)) and (_coconut.len(_coconut_case_match_to_5) == 4) and (_coconut_case_match_to_5[1] == "BHWC"):  #         match Numpy(_,"BHWC",_,_):
            _coconut_case_match_check_5 = True  #         match Numpy(_,"BHWC",_,_):
        if _coconut_case_match_check_5:  #         match Numpy(_,"BHWC",_,_):
            return ([(_coconut.operator.methodcaller("transpose", 0, 3, 1, 2), "BCHW", ms_0312), ])  #             return [(.transpose(0,3,1,2),"BCHW",ms_0312)]
    if not _coconut_case_match_check_5:  #         match Torch(_,"BCHW",_,_):
        if (_coconut.isinstance(_coconut_case_match_to_5, Torch)) and (_coconut.len(_coconut_case_match_to_5) == 4) and (_coconut_case_match_to_5[1] == "BCHW"):  #         match Torch(_,"BCHW",_,_):
            _coconut_case_match_check_5 = True  #         match Torch(_,"BCHW",_,_):
        if _coconut_case_match_check_5:  #         match Torch(_,"BCHW",_,_):
            return ([(_coconut_base_compose(_coconut.operator.methodcaller("transpose", 1, 2), (_coconut.operator.methodcaller("transpose", 2, 3), 0)), "BHWC", ms_0231), ])  #             return [(.transpose(1,2) ..> .transpose(2,3),"BHWC",ms_0231)]
    if not _coconut_case_match_check_5:  #         match Torch(_,"BHWC",_,_):
        if (_coconut.isinstance(_coconut_case_match_to_5, Torch)) and (_coconut.len(_coconut_case_match_to_5) == 4) and (_coconut_case_match_to_5[1] == "BHWC"):  #         match Torch(_,"BHWC",_,_):
            _coconut_case_match_check_5 = True  #         match Torch(_,"BHWC",_,_):
        if _coconut_case_match_check_5:  #         match Torch(_,"BHWC",_,_):
            return ([(_coconut_base_compose(_coconut.operator.methodcaller("transpose", 2, 3), (_coconut.operator.methodcaller("transpose", 1, 2), 0)), "BCHW", ms_0312), ])  #             return [(.transpose(2,3) ..> .transpose(1,2),"BCHW",ms_0312)]
    return ([])  #     return []
@to_imagedef  # @to_imagedef
def change_arrange(imdef: 'ImageDef'):  # def change_arrange(imdef:ImageDef):
    _coconut_match_to_0 = imdef  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    _coconut_match_check_4 = False  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    if (_coconut.isinstance(_coconut_match_to_0, TensorLike)) and (_coconut.len(_coconut_match_to_0) == 4):  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        _coconut_match_set_name_dtype = _coconut_match_to_0[0]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        _coconut_match_set_name_arng = _coconut_match_to_0[1]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        _coconut_match_set_name_ch_repr = _coconut_match_to_0[2]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        _coconut_match_set_name_vr = _coconut_match_to_0[3]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        _coconut_match_check_4 = True  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    if _coconut_match_check_4:  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
            dtype = _coconut_match_to_0[0]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
            arng = _coconut_match_to_0[1]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
            ch_repr = _coconut_match_to_0[2]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
            vr = _coconut_match_to_0[3]  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
    if _coconut_match_check_4:  #     match TensorLike(dtype,arng,ch_repr,vr) in imdef:
        return ([DataEdge(imdef, imdef.__class__(dtype, _arng, ch_repr, vr), f, 1, name="{_coconut_format_0} to {_coconut_format_1}".format(_coconut_format_0=(arng), _coconut_format_1=(_arng)), meta_shifter=meta_shifter) for f, _arng, meta_shifter in change_arng(imdef)])  #         return [DataEdge(imdef,
    return ([])  #     return []
ms_drop_bhwc_alpha = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], s[1], s[2], 3)))  # ms_drop_bhwc_alpha = (s->(s[0],s[1],s[2],3)) |> ss_to_ms$
ms_drop_bchw_alpha = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], 3, s[2], s[3])))  # ms_drop_bchw_alpha = (s->(s[0],3,s[2],s[3])) |> ss_to_ms$
@to_imagedef  # @to_imagedef
def drop_alpha(imdef):  # def drop_alpha(imdef):
    _coconut_case_match_to_6 = imdef  #     case imdef:
    _coconut_case_match_check_6 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_6, TensorLike)) and (_coconut.len(_coconut_case_match_to_6) == 4) and (_coconut_case_match_to_6[1] == "BHWC") and (_coconut_case_match_to_6[2] == "RGBA"):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_6[0]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_6[3]  #     case imdef:
        _coconut_case_match_check_6 = True  #     case imdef:
    if _coconut_case_match_check_6:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_6[0]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_6[3]  #     case imdef:
    if _coconut_case_match_check_6:  #     case imdef:
        return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "BHWC", "RGB", vr), f=lambda a: a[:, :, :, :3], cost=1, name="select rgb channel".format(), meta_shifter=ms_drop_bhwc_alpha), ])  #             return [DataEdge(a=imdef,
    if not _coconut_case_match_check_6:  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
        _coconut_match_set_name_dtype = _coconut_sentinel  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
        _coconut_match_set_name_vr = _coconut_sentinel  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
        if (_coconut.isinstance(_coconut_case_match_to_6, TensorLike)) and (_coconut.len(_coconut_case_match_to_6) == 4) and (_coconut_case_match_to_6[1] == "BCHW") and (_coconut_case_match_to_6[2] == "RGBA"):  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
            _coconut_match_set_name_dtype = _coconut_case_match_to_6[0]  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
            _coconut_match_set_name_vr = _coconut_case_match_to_6[3]  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
            _coconut_case_match_check_6 = True  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
        if _coconut_case_match_check_6:  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
                dtype = _coconut_case_match_to_6[0]  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
            if _coconut_match_set_name_vr is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
                vr = _coconut_case_match_to_6[3]  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
        if _coconut_case_match_check_6:  #                          b=imdef.__class__(dtype,"BHWC","RGB",vr),
            return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "BCHW", "RGB", vr), f=lambda a: a[:, :3], cost=1, name="select rgb channel".format(), meta_shifter=ms_drop_bchw_alpha), ])  #             return [DataEdge(a=imdef,

ms_select_bhwc_channel = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], s[1], s[2], 1)))  # ms_select_bhwc_channel = (s->(s[0],s[1],s[2],1)) |> ss_to_ms$
ms_select_bchw_channel = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], 1, s[2], s[3])))  # ms_select_bchw_channel = (s->(s[0],1,s[2],s[3])) |> ss_to_ms$

@to_imagedef  # @to_imagedef
def select_channel(imdef: 'ImageDef'):  # def select_channel(imdef:ImageDef):
    _coconut_case_match_to_7 = imdef  #     case imdef:
    _coconut_case_match_check_7 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_7, TensorLike)) and (_coconut.len(_coconut_case_match_to_7) == 4) and (_coconut_case_match_to_7[1] == "BHWC"):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_7[0]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_7[2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_7[3]  #     case imdef:
        _coconut_case_match_check_7 = True  #     case imdef:
    if _coconut_case_match_check_7:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_7[0]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_7[2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_7[3]  #     case imdef:
    if _coconut_case_match_check_7 and not (len(ch_repr) >= 1):  #     case imdef:
        _coconut_case_match_check_7 = False  #     case imdef:
    if _coconut_case_match_check_7:  #     case imdef:
        selector = lambda i: lambda a: a[:, :, :, [i, ]]  #             selector = i->a->a[:,:,:,[i]]
        return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "BHWC", c, vr), f=selector(i), cost=10, name="select {_coconut_format_0} channel".format(_coconut_format_0=(c)), meta_shifter=ms_select_bhwc_channel) for i, c in enumerate(ch_splitter(ch_repr))])  #             return [DataEdge(a=imdef,
    if not _coconut_case_match_check_7:  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
        _coconut_match_set_name_dtype = _coconut_sentinel  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
        _coconut_match_set_name_vr = _coconut_sentinel  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
        if (_coconut.isinstance(_coconut_case_match_to_7, TensorLike)) and (_coconut.len(_coconut_case_match_to_7) == 4) and (_coconut_case_match_to_7[1] == "BCHW"):  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            _coconut_match_set_name_dtype = _coconut_case_match_to_7[0]  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_7[2]  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            _coconut_match_set_name_vr = _coconut_case_match_to_7[3]  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            _coconut_case_match_check_7 = True  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
        if _coconut_case_match_check_7:  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
                dtype = _coconut_case_match_to_7[0]  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
                ch_repr = _coconut_case_match_to_7[2]  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            if _coconut_match_set_name_vr is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
                vr = _coconut_case_match_to_7[3]  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
        if _coconut_case_match_check_7 and not (len(ch_repr) >= 1):  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            _coconut_case_match_check_7 = False  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
        if _coconut_case_match_check_7:  #                          b=imdef.__class__(dtype,"BHWC",c,vr),
            selector = lambda i: lambda a: a[:, [i, ]]  #             selector = i->a->a[:,[i]]
            return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "BCHW", c, vr), f=selector(i), cost=10, name="select {_coconut_format_0} channel".format(_coconut_format_0=(c)), meta_shifter=ms_select_bchw_channel) for i, c in enumerate(ch_splitter(ch_repr))])  #             return [DataEdge(a=imdef,
    return ([])  #     return []
ms_drop_bhwc_channel = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], s[1], s[2])))  # ms_drop_bhwc_channel = (s->(s[0],s[1],s[2])) |> ss_to_ms$
ms_drop_bchw_channel = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], s[2], s[3])))  # ms_drop_bchw_channel = (s->(s[0],s[2],s[3])) |> ss_to_ms$
ms_drop_chw_channel = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[1], s[2])))  # ms_drop_chw_channel = (s->(s[1],s[2])) |> ss_to_ms$
ms_drop_hwc_channel = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], s[1])))  # ms_drop_hwc_channel = (s->(s[0],s[1])) |> ss_to_ms$


@to_imagedef  # @to_imagedef
def drop_channel(imdef: 'ImageDef'):  # def drop_channel(imdef:ImageDef):
    _coconut_case_match_to_8 = imdef  #     case imdef:
    _coconut_case_match_check_8 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_8, TensorLike)) and (_coconut.len(_coconut_case_match_to_8) == 4) and (_coconut_case_match_to_8[1] == "BHWC"):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_8[0]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_8[2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_8[3]  #     case imdef:
        _coconut_case_match_check_8 = True  #     case imdef:
    if _coconut_case_match_check_8:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_8[0]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_8[2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_8[3]  #     case imdef:
    if _coconut_case_match_check_8 and not (len(ch_splitter(ch_repr)) == 1):  #     case imdef:
        _coconut_case_match_check_8 = False  #     case imdef:
    if _coconut_case_match_check_8:  #     case imdef:
        return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "BHW", ch_repr, vr), f=lambda a: a[:, :, :, 0], cost=1, name="BHWC to BHW".format(), meta_shifter=ms_drop_bhwc_channel), ])  #             return [DataEdge(a=imdef,
    if not _coconut_case_match_check_8:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        _coconut_match_set_name_dtype = _coconut_sentinel  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        _coconut_match_set_name_vr = _coconut_sentinel  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if (_coconut.isinstance(_coconut_case_match_to_8, TensorLike)) and (_coconut.len(_coconut_case_match_to_8) == 4) and (_coconut_case_match_to_8[1] == "BCHW"):  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_match_set_name_dtype = _coconut_case_match_to_8[0]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_8[2]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_match_set_name_vr = _coconut_case_match_to_8[3]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_case_match_check_8 = True  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if _coconut_case_match_check_8:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
                dtype = _coconut_case_match_to_8[0]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
                ch_repr = _coconut_case_match_to_8[2]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            if _coconut_match_set_name_vr is not _coconut_sentinel:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
                vr = _coconut_case_match_to_8[3]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if _coconut_case_match_check_8 and not (len(ch_splitter(ch_repr)) == 1):  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_case_match_check_8 = False  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if _coconut_case_match_check_8:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "BHW", ch_repr, vr), f=lambda a: a[:, 0], cost=1, name="BCHW to BHW".format(), meta_shifter=ms_drop_bchw_channel), ])  #             return [DataEdge(a=imdef,
    if not _coconut_case_match_check_8:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        _coconut_match_set_name_dtype = _coconut_sentinel  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        _coconut_match_set_name_vr = _coconut_sentinel  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if (_coconut.isinstance(_coconut_case_match_to_8, TensorLike)) and (_coconut.len(_coconut_case_match_to_8) == 4) and (_coconut_case_match_to_8[1] == "CHW"):  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_match_set_name_dtype = _coconut_case_match_to_8[0]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_8[2]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_match_set_name_vr = _coconut_case_match_to_8[3]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_case_match_check_8 = True  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if _coconut_case_match_check_8:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
                dtype = _coconut_case_match_to_8[0]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
                ch_repr = _coconut_case_match_to_8[2]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            if _coconut_match_set_name_vr is not _coconut_sentinel:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
                vr = _coconut_case_match_to_8[3]  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if _coconut_case_match_check_8 and not (len(ch_splitter(ch_repr)) == 1):  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            _coconut_case_match_check_8 = False  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
        if _coconut_case_match_check_8:  #                         b=imdef.__class__(dtype,"BHW",ch_repr,vr),
            return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "HW", ch_repr, vr), f=lambda a: a[0], cost=1, name="CHW to HW", meta_shifter=ms_drop_chw_channel), ])  #             return [DataEdge(a = imdef,b=imdef.__class__(dtype,"HW",ch_repr,vr),
    if not _coconut_case_match_check_8:  #                          f = a->a[0],cost=1,name="CHW to HW",
        _coconut_match_set_name_dtype = _coconut_sentinel  #                          f = a->a[0],cost=1,name="CHW to HW",
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                          f = a->a[0],cost=1,name="CHW to HW",
        _coconut_match_set_name_vr = _coconut_sentinel  #                          f = a->a[0],cost=1,name="CHW to HW",
        if (_coconut.isinstance(_coconut_case_match_to_8, TensorLike)) and (_coconut.len(_coconut_case_match_to_8) == 4) and (_coconut_case_match_to_8[1] == "HWC"):  #                          f = a->a[0],cost=1,name="CHW to HW",
            _coconut_match_set_name_dtype = _coconut_case_match_to_8[0]  #                          f = a->a[0],cost=1,name="CHW to HW",
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_8[2]  #                          f = a->a[0],cost=1,name="CHW to HW",
            _coconut_match_set_name_vr = _coconut_case_match_to_8[3]  #                          f = a->a[0],cost=1,name="CHW to HW",
            _coconut_case_match_check_8 = True  #                          f = a->a[0],cost=1,name="CHW to HW",
        if _coconut_case_match_check_8:  #                          f = a->a[0],cost=1,name="CHW to HW",
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                          f = a->a[0],cost=1,name="CHW to HW",
                dtype = _coconut_case_match_to_8[0]  #                          f = a->a[0],cost=1,name="CHW to HW",
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                          f = a->a[0],cost=1,name="CHW to HW",
                ch_repr = _coconut_case_match_to_8[2]  #                          f = a->a[0],cost=1,name="CHW to HW",
            if _coconut_match_set_name_vr is not _coconut_sentinel:  #                          f = a->a[0],cost=1,name="CHW to HW",
                vr = _coconut_case_match_to_8[3]  #                          f = a->a[0],cost=1,name="CHW to HW",
        if _coconut_case_match_check_8 and not (len(ch_splitter(ch_repr)) == 1):  #                          f = a->a[0],cost=1,name="CHW to HW",
            _coconut_case_match_check_8 = False  #                          f = a->a[0],cost=1,name="CHW to HW",
        if _coconut_case_match_check_8:  #                          f = a->a[0],cost=1,name="CHW to HW",
            return ([DataEdge(a=imdef, b=imdef.__class__(dtype, "HW", ch_repr, vr), f=lambda a: a[:, :, 0], cost=1, name="HWC to HW", meta_shifter=ms_drop_hwc_channel), ])  #             return [DataEdge(a = imdef,b=imdef.__class__(dtype,"HW",ch_repr,vr),

    return ([])  #     return []
def enforce_mode(img, mode):  # def enforce_mode(img,mode):
    return (Image.fromarray(np.array(img), mode))  #     return Image.fromarray(np.array(img),mode)
"""
@to_imagedef
def RGB_to_YCbCr(state):
    case state:
        match PILImage("RGB","RGB"):
            return [DataEdge(
            a=state,
            b=PILImage("YCbCr","YCbCr"),
            f= enforce_mode$(mode="RGB") ..> .convert("YCbCr"),
            cost=1,
            name="RGB to YCbCr"
            )]
        match PILImage("YCbCr","YCbCr"):
            return [DataEdge(
            a=state,
            b=PILImage("RGB","RGB"),
            f= enforce_mode$(mode="YCbCr") ..> .convert("RGB"),
            cost=1,
            name="YCbCr to RGB"
            )]
"""  # """
def rgb_to_ycbcr(image: 'torch.Tensor') -> 'torch.Tensor':  # def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YCbCr version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_ycbcr(input)  # 2x3x4x5
    """  #     """
    if not isinstance(image, torch.Tensor):  #     if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))  #         raise TypeError("Input type is not a torch.Tensor. Got {}".format(

    if len(image.shape) < 3 or image.shape[-3] != 3:  #     if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))  #         raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"

    r = image[..., 0, :, :]  # type: torch.Tensor  #     r: torch.Tensor = image[..., 0, :, :]
    if "__annotations__" not in _coconut.locals():  #     r: torch.Tensor = image[..., 0, :, :]
        __annotations__ = {}  #     r: torch.Tensor = image[..., 0, :, :]
    __annotations__["r"] = 'torch.Tensor'  #     r: torch.Tensor = image[..., 0, :, :]
    g = image[..., 1, :, :]  # type: torch.Tensor  #     g: torch.Tensor = image[..., 1, :, :]
    if "__annotations__" not in _coconut.locals():  #     g: torch.Tensor = image[..., 1, :, :]
        __annotations__ = {}  #     g: torch.Tensor = image[..., 1, :, :]
    __annotations__["g"] = 'torch.Tensor'  #     g: torch.Tensor = image[..., 1, :, :]
    b = image[..., 2, :, :]  # type: torch.Tensor  #     b: torch.Tensor = image[..., 2, :, :]
    if "__annotations__" not in _coconut.locals():  #     b: torch.Tensor = image[..., 2, :, :]
        __annotations__ = {}  #     b: torch.Tensor = image[..., 2, :, :]
    __annotations__["b"] = 'torch.Tensor'  #     b: torch.Tensor = image[..., 2, :, :]

    delta = 0.5  # type: float  #     delta: float = 0.5
    if "__annotations__" not in _coconut.locals():  #     delta: float = 0.5
        __annotations__ = {}  #     delta: float = 0.5
    __annotations__["delta"] = 'float'  #     delta: float = 0.5
    y = 0.299 * r + 0.587 * g + 0.114 * b  # type: torch.Tensor  #     y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    if "__annotations__" not in _coconut.locals():  #     y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
        __annotations__ = {}  #     y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    __annotations__["y"] = 'torch.Tensor'  #     y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + delta  # type: torch.Tensor  #     cb: torch.Tensor = (b - y) * 0.564 + delta
    if "__annotations__" not in _coconut.locals():  #     cb: torch.Tensor = (b - y) * 0.564 + delta
        __annotations__ = {}  #     cb: torch.Tensor = (b - y) * 0.564 + delta
    __annotations__["cb"] = 'torch.Tensor'  #     cb: torch.Tensor = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta  # type: torch.Tensor  #     cr: torch.Tensor = (r - y) * 0.713 + delta
    if "__annotations__" not in _coconut.locals():  #     cr: torch.Tensor = (r - y) * 0.713 + delta
        __annotations__ = {}  #     cr: torch.Tensor = (r - y) * 0.713 + delta
    __annotations__["cr"] = 'torch.Tensor'  #     cr: torch.Tensor = (r - y) * 0.713 + delta
    return (torch.stack([y, cb, cr], -3))  #     return torch.stack([y, cb, cr], -3)


def ycbcr_to_rgb(image: 'torch.Tensor') -> 'torch.Tensor':  # def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """  #     """
    if not isinstance(image, torch.Tensor):  #     if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))  #         raise TypeError("Input type is not a torch.Tensor. Got {}".format(

    if len(image.shape) < 3 or image.shape[-3] != 3:  #     if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))  #         raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"

    y = image[..., 0, :, :]  # type: torch.Tensor  #     y: torch.Tensor = image[..., 0, :, :]
    if "__annotations__" not in _coconut.locals():  #     y: torch.Tensor = image[..., 0, :, :]
        __annotations__ = {}  #     y: torch.Tensor = image[..., 0, :, :]
    __annotations__["y"] = 'torch.Tensor'  #     y: torch.Tensor = image[..., 0, :, :]
    cb = image[..., 1, :, :]  # type: torch.Tensor  #     cb: torch.Tensor = image[..., 1, :, :]
    if "__annotations__" not in _coconut.locals():  #     cb: torch.Tensor = image[..., 1, :, :]
        __annotations__ = {}  #     cb: torch.Tensor = image[..., 1, :, :]
    __annotations__["cb"] = 'torch.Tensor'  #     cb: torch.Tensor = image[..., 1, :, :]
    cr = image[..., 2, :, :]  # type: torch.Tensor  #     cr: torch.Tensor = image[..., 2, :, :]
    if "__annotations__" not in _coconut.locals():  #     cr: torch.Tensor = image[..., 2, :, :]
        __annotations__ = {}  #     cr: torch.Tensor = image[..., 2, :, :]
    __annotations__["cr"] = 'torch.Tensor'  #     cr: torch.Tensor = image[..., 2, :, :]

    delta = 0.5  # type: float  #     delta: float = 0.5
    if "__annotations__" not in _coconut.locals():  #     delta: float = 0.5
        __annotations__ = {}  #     delta: float = 0.5
    __annotations__["delta"] = 'float'  #     delta: float = 0.5
    cb_shifted = cb - delta  # type: torch.Tensor  #     cb_shifted: torch.Tensor = cb - delta
    if "__annotations__" not in _coconut.locals():  #     cb_shifted: torch.Tensor = cb - delta
        __annotations__ = {}  #     cb_shifted: torch.Tensor = cb - delta
    __annotations__["cb_shifted"] = 'torch.Tensor'  #     cb_shifted: torch.Tensor = cb - delta
    cr_shifted = cr - delta  # type: torch.Tensor  #     cr_shifted: torch.Tensor = cr - delta
    if "__annotations__" not in _coconut.locals():  #     cr_shifted: torch.Tensor = cr - delta
        __annotations__ = {}  #     cr_shifted: torch.Tensor = cr - delta
    __annotations__["cr_shifted"] = 'torch.Tensor'  #     cr_shifted: torch.Tensor = cr - delta

    r = y + 1.403 * cr_shifted  # type: torch.Tensor  #     r: torch.Tensor = y + 1.403 * cr_shifted
    if "__annotations__" not in _coconut.locals():  #     r: torch.Tensor = y + 1.403 * cr_shifted
        __annotations__ = {}  #     r: torch.Tensor = y + 1.403 * cr_shifted
    __annotations__["r"] = 'torch.Tensor'  #     r: torch.Tensor = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted  # type: torch.Tensor  #     g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    if "__annotations__" not in _coconut.locals():  #     g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
        __annotations__ = {}  #     g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    __annotations__["g"] = 'torch.Tensor'  #     g: torch.Tensor = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted  # type: torch.Tensor  #     b: torch.Tensor = y + 1.773 * cb_shifted
    if "__annotations__" not in _coconut.locals():  #     b: torch.Tensor = y + 1.773 * cb_shifted
        __annotations__ = {}  #     b: torch.Tensor = y + 1.773 * cb_shifted
    __annotations__["b"] = 'torch.Tensor'  #     b: torch.Tensor = y + 1.773 * cb_shifted
    return (torch.stack([r, g, b], -3))  #     return torch.stack([r, g, b], -3)


@to_imagedef  # @to_imagedef
def RGB_to_YCbCr(state):  # def RGB_to_YCbCr(state):
    _coconut_case_match_to_9 = state  #     case state:
    _coconut_case_match_check_9 = False  #     case state:
    if (_coconut.isinstance(_coconut_case_match_to_9, Torch)) and (_coconut.len(_coconut_case_match_to_9) == 4) and (_coconut_case_match_to_9[0] == "float32") and (_coconut_case_match_to_9[1] == "BCHW") and (_coconut_case_match_to_9[2] == "RGB") and (_coconut_case_match_to_9[3] == VR_0_1):  #     case state:
        _coconut_case_match_check_9 = True  #     case state:
    if _coconut_case_match_check_9:  #     case state:
        return ([DataEdge(a=state, b=Torch("float32", "BCHW", "RGB", VR_0_1), f=rgb_to_ycbcr, cost=1, name="RGB_to_YCbCr(torch)"), ])  #             return [DataEdge(a=state,
    if not _coconut_case_match_check_9:  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
        if (_coconut.isinstance(_coconut_case_match_to_9, Torch)) and (_coconut.len(_coconut_case_match_to_9) == 4) and (_coconut_case_match_to_9[0] == "float32") and (_coconut_case_match_to_9[1] == "BCHW") and (_coconut_case_match_to_9[2] == "YCbCr") and (_coconut_case_match_to_9[3] == VR_0_1):  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
            _coconut_case_match_check_9 = True  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
        if _coconut_case_match_check_9:  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
            return ([DataEdge(a=state, b=Torch("float32", "BCHW", "RGB", VR_0_1), f=ycbcr_to_rgb, cost=1, name="YCbCr_to_RGB(torch)"), ])  #             return [DataEdge(a=state,
    if not _coconut_case_match_check_9:  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
        if (_coconut.isinstance(_coconut_case_match_to_9, Torch)) and (_coconut.len(_coconut_case_match_to_9) == 4) and (_coconut_case_match_to_9[0] == "float32") and (_coconut_case_match_to_9[1] == "CHW") and (_coconut_case_match_to_9[2] == "RGB") and (_coconut_case_match_to_9[3] == VR_0_1):  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
            _coconut_case_match_check_9 = True  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
        if _coconut_case_match_check_9:  #                          b=Torch("float32","BCHW","RGB",VR_0_1),
            return ([DataEdge(a=state, b=Torch("float32", "CHW", "YCbCr", VR_0_1), f=rgb_to_ycbcr, cost=1, name="RGB_to_YCbCr(torch)"), ])  #             return [DataEdge(a=state,
    if not _coconut_case_match_check_9:  #                          b=Torch("float32","CHW","YCbCr",VR_0_1),
        if (_coconut.isinstance(_coconut_case_match_to_9, Torch)) and (_coconut.len(_coconut_case_match_to_9) == 4) and (_coconut_case_match_to_9[0] == "float32") and (_coconut_case_match_to_9[1] == "CHW") and (_coconut_case_match_to_9[2] == "YCbCr") and (_coconut_case_match_to_9[3] == VR_0_1):  #                          b=Torch("float32","CHW","YCbCr",VR_0_1),
            _coconut_case_match_check_9 = True  #                          b=Torch("float32","CHW","YCbCr",VR_0_1),
        if _coconut_case_match_check_9:  #                          b=Torch("float32","CHW","YCbCr",VR_0_1),
            return ([DataEdge(a=state, b=Torch("float32", "CHW", "RGB", VR_0_1), f=ycbcr_to_rgb, cost=1, name="YCbCr_to_RGB(torch)"), ])  #             return [DataEdge(a=state,



#shape change!
ms_add_b_ch = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (1, *s)))  # ms_add_b_ch = (s->(1,*s)) |> ss_to_ms$
"adds 1 as batch shape"  # "adds 1 as batch shape"
ms_del_b_ch = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: s[1:]))  # ms_del_b_ch = (s->s[1:])  |> ss_to_ms$
# TODO what to do with non-batched states?

def en_batch(imdef: 'ImageDef'):  # def en_batch(imdef:ImageDef):
    _coconut_case_match_to_10 = imdef  #     case imdef:
    _coconut_case_match_check_10 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_10, ImageDef)) and (_coconut.len(_coconut_case_match_to_10) == 2) and (_coconut.isinstance(_coconut_case_match_to_10[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_10[0]) == 4):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_10[0][0]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_10[0][2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_10[0][3]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_10[1]  #     case imdef:
        _coconut_case_match_check_10 = True  #     case imdef:
    if _coconut_case_match_check_10:  #     case imdef:
        _coconut_case_match_check_10 = False  #     case imdef:
        if (not _coconut_case_match_check_10) and (_coconut.isinstance(_coconut_case_match_to_10, ImageDef)) and (_coconut.len(_coconut_case_match_to_10) == 2) and (_coconut.isinstance(_coconut_case_match_to_10[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_10[0]) == 4) and (_coconut_case_match_to_10[0][1] == "HWC"):  #     case imdef:
            _coconut_match_set_name_dtype = _coconut_case_match_to_10[0][0]  #     case imdef:
            _coconut_case_match_check_10 = True  #     case imdef:
        if (not _coconut_case_match_check_10) and (_coconut.isinstance(_coconut_case_match_to_10, ImageDef)) and (_coconut.len(_coconut_case_match_to_10) == 2) and (_coconut.isinstance(_coconut_case_match_to_10[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_10[0]) == 4) and (_coconut_case_match_to_10[0][1] == "CHW"):  #     case imdef:
            _coconut_match_set_name_dtype = _coconut_case_match_to_10[0][0]  #     case imdef:
            _coconut_case_match_check_10 = True  #     case imdef:
        if (not _coconut_case_match_check_10) and (_coconut.isinstance(_coconut_case_match_to_10, ImageDef)) and (_coconut.len(_coconut_case_match_to_10) == 2) and (_coconut.isinstance(_coconut_case_match_to_10[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_10[0]) == 4) and (_coconut_case_match_to_10[0][1] == "HW"):  #     case imdef:
            _coconut_match_set_name_dtype = _coconut_case_match_to_10[0][0]  #     case imdef:
            _coconut_case_match_check_10 = True  #     case imdef:

    if _coconut_case_match_check_10:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_10[0][0]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_10[0][2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_10[0][3]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_10[1]  #     case imdef:
    if _coconut_case_match_check_10:  #     case imdef:
        new_arng = "B" + imdef.data_type.arrange  #             new_arng = "B"+imdef.data_type.arrange
        return ([Edge(a=imdef, b=ImageDef(imdef.data_type.__class__(dtype, new_arng, ch_repr, vr), ms_add_b_ch(meta)), f=lambda a: a[None], cost=10, name="tensor_like en_batch".format()), ])  #             return [Edge(a=imdef,
    if not _coconut_case_match_check_10:  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
        _coconut_match_set_name_mode = _coconut_sentinel  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
        _coconut_match_set_name_channel_repr = _coconut_sentinel  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
        _coconut_match_set_name_meta = _coconut_sentinel  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
        if (_coconut.isinstance(_coconut_case_match_to_10, ImageDef)) and (_coconut.len(_coconut_case_match_to_10) == 2) and (_coconut.isinstance(_coconut_case_match_to_10[0], PILImage)) and (_coconut.len(_coconut_case_match_to_10[0]) == 2):  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            _coconut_match_set_name_mode = _coconut_case_match_to_10[0][0]  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            _coconut_match_set_name_channel_repr = _coconut_case_match_to_10[0][1]  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            _coconut_match_set_name_meta = _coconut_case_match_to_10[1]  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            _coconut_case_match_check_10 = True  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
        if _coconut_case_match_check_10:  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            if _coconut_match_set_name_mode is not _coconut_sentinel:  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
                mode = _coconut_case_match_to_10[0][0]  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            if _coconut_match_set_name_channel_repr is not _coconut_sentinel:  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
                channel_repr = _coconut_case_match_to_10[0][1]  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
                meta = _coconut_case_match_to_10[1]  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
        if _coconut_case_match_check_10:  #                          b=ImageDef(imdef.data_type.__class__(dtype,new_arng,ch_repr,vr),ms_add_b_ch(meta)),
            return ([Edge(a=imdef, b=ImageDef(PILImages(mode, channel_repr), ms_add_b_ch(meta)), f=lambda a: [a, ], cost=10, name="pil image en_batch".format()), ])  #             return [Edge(a=imdef,
    return ([])  #     return []

def de_batch(imdef: 'ImageDef'):  # def de_batch(imdef:ImageDef):
    _coconut_case_match_to_11 = imdef  #     case imdef:
    _coconut_case_match_check_11 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_shape = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_11, ImageDef)) and (_coconut.len(_coconut_case_match_to_11) == 2) and (_coconut.isinstance(_coconut_case_match_to_11[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_11[0]) == 4) and (_coconut.isinstance(_coconut_case_match_to_11[1], _coconut.abc.Mapping)) and (_coconut.len(_coconut_case_match_to_11[1]) == 1):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_11[0][0]  #     case imdef:
        _coconut_match_set_name_arng = _coconut_case_match_to_11[0][1]  #     case imdef:
        _coconut_match_set_name_ch = _coconut_case_match_to_11[0][2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_11[0][3]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_11[1]  #     case imdef:
        _coconut_match_temp_0 = _coconut_case_match_to_11[1].get("shape", _coconut_sentinel)  #     case imdef:
        if _coconut_match_temp_0 is not _coconut_sentinel:  #     case imdef:
            _coconut_match_set_name_shape = _coconut_match_temp_0  #     case imdef:
            _coconut_case_match_check_11 = True  #     case imdef:
    if _coconut_case_match_check_11:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_11[0][0]  #     case imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     case imdef:
            arng = _coconut_case_match_to_11[0][1]  #     case imdef:
        if _coconut_match_set_name_ch is not _coconut_sentinel:  #     case imdef:
            ch = _coconut_case_match_to_11[0][2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_11[0][3]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_11[1]  #     case imdef:
        if _coconut_match_set_name_shape is not _coconut_sentinel:  #     case imdef:
            shape = _coconut_match_temp_0  #     case imdef:
    if _coconut_case_match_check_11 and not ("B" in arng and shape[0] == 1):  #     case imdef:
        _coconut_case_match_check_11 = False  #     case imdef:
    if _coconut_case_match_check_11:  #     case imdef:
        return ([Edge(a=imdef, b=ImageDef(imdef.data_type.__class__(dtype, arng[1:], ch, vr), ms_del_b_ch(meta)), f=lambda a: a[0], cost=1, name="de_batch en_batched image".format()), ])  #             return [Edge(
    if not _coconut_case_match_check_11:  #                 a=imdef,
        _coconut_match_set_name_mode = _coconut_sentinel  #                 a=imdef,
        _coconut_match_set_name_ch = _coconut_sentinel  #                 a=imdef,
        _coconut_match_set_name_meta = _coconut_sentinel  #                 a=imdef,
        _coconut_match_set_name_shape = _coconut_sentinel  #                 a=imdef,
        if (_coconut.isinstance(_coconut_case_match_to_11, ImageDef)) and (_coconut.len(_coconut_case_match_to_11) == 2) and (_coconut.isinstance(_coconut_case_match_to_11[0], PILImages)) and (_coconut.len(_coconut_case_match_to_11[0]) == 2) and (_coconut.isinstance(_coconut_case_match_to_11[1], _coconut.abc.Mapping)) and (_coconut.len(_coconut_case_match_to_11[1]) == 1):  #                 a=imdef,
            _coconut_match_set_name_mode = _coconut_case_match_to_11[0][0]  #                 a=imdef,
            _coconut_match_set_name_ch = _coconut_case_match_to_11[0][1]  #                 a=imdef,
            _coconut_match_set_name_meta = _coconut_case_match_to_11[1]  #                 a=imdef,
            _coconut_match_temp_0 = _coconut_case_match_to_11[1].get("shape", _coconut_sentinel)  #                 a=imdef,
            if _coconut_match_temp_0 is not _coconut_sentinel:  #                 a=imdef,
                _coconut_match_set_name_shape = _coconut_match_temp_0  #                 a=imdef,
                _coconut_case_match_check_11 = True  #                 a=imdef,
        if _coconut_case_match_check_11:  #                 a=imdef,
            if _coconut_match_set_name_mode is not _coconut_sentinel:  #                 a=imdef,
                mode = _coconut_case_match_to_11[0][0]  #                 a=imdef,
            if _coconut_match_set_name_ch is not _coconut_sentinel:  #                 a=imdef,
                ch = _coconut_case_match_to_11[0][1]  #                 a=imdef,
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 a=imdef,
                meta = _coconut_case_match_to_11[1]  #                 a=imdef,
            if _coconut_match_set_name_shape is not _coconut_sentinel:  #                 a=imdef,
                shape = _coconut_match_temp_0  #                 a=imdef,
        if _coconut_case_match_check_11 and not ("en_batched" in meta):  #                 a=imdef,
            _coconut_case_match_check_11 = False  #                 a=imdef,
        if _coconut_case_match_check_11:  #                 a=imdef,
            return ([Edge(a=imdef, b=ImageDef(PILImage(mode, ch), ms_del_b_ch(meta)), f=lambda a: a[0], cost=1, name="de_batch en_batched image".format()), ])  #             return [Edge(

def drop_meta(imdef: 'ImageDef'):  # def drop_meta(imdef:ImageDef):
    _coconut_case_match_to_12 = imdef  #     case imdef:
    _coconut_case_match_check_12 = False  #     case imdef:
    _coconut_match_set_name_data_type = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_12, ImageDef)) and (_coconut.len(_coconut_case_match_to_12) == 2):  #     case imdef:
        _coconut_match_set_name_data_type = _coconut_case_match_to_12[0]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_12[1]  #     case imdef:
        _coconut_case_match_check_12 = True  #     case imdef:
    if _coconut_case_match_check_12:  #     case imdef:
        if _coconut_match_set_name_data_type is not _coconut_sentinel:  #     case imdef:
            data_type = _coconut_case_match_to_12[0]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_12[1]  #     case imdef:
    if _coconut_case_match_check_12:  #     case imdef:
        return ([Edge(a=imdef, b=ImageDef(data_type, fdict()), f=lambda a: a, cost=1, name="drop all metadata"), ])  #             return [Edge(


@to_imagedef  # @to_imagedef
def to_rgba(imdef: 'ImageDef'):  # def to_rgba(imdef:ImageDef):
    _coconut_case_match_to_13 = imdef  #     case imdef:
    _coconut_case_match_check_13 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_13, TensorLike)) and (_coconut.len(_coconut_case_match_to_13) == 4) and (_coconut_case_match_to_13[3] == "0_1"):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_13[0]  #     case imdef:
        _coconut_match_set_name_arng = _coconut_case_match_to_13[1]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_13[2]  #     case imdef:
        _coconut_case_match_check_13 = True  #     case imdef:
    if _coconut_case_match_check_13:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_13[0]  #     case imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     case imdef:
            arng = _coconut_case_match_to_13[1]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_13[2]  #     case imdef:
    if _coconut_case_match_check_13 and not (len(ch_splitter(ch_repr)) == 4):  #     case imdef:
        _coconut_case_match_check_13 = False  #     case imdef:
    if _coconut_case_match_check_13:  #     case imdef:
        return ([DataEdge(a=imdef, b=imdef.__class__(dtype, arng, "RGBA", "0_1"), f=lambda a: a, cost=10, name="view {_coconut_format_0} as RGBA ".format(_coconut_format_0=(ch_repr))), ])  #             return [DataEdge(a=imdef,
    if not _coconut_case_match_check_13:  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
        _coconut_match_set_name_dtype = _coconut_sentinel  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
        _coconut_match_set_name_arng = _coconut_sentinel  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
        if (_coconut.isinstance(_coconut_case_match_to_13, TensorLike)) and (_coconut.len(_coconut_case_match_to_13) == 4) and (_coconut_case_match_to_13[3] == "0_1"):  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            _coconut_match_set_name_dtype = _coconut_case_match_to_13[0]  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            _coconut_match_set_name_arng = _coconut_case_match_to_13[1]  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_13[2]  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            _coconut_case_match_check_13 = True  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
        if _coconut_case_match_check_13:  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
                dtype = _coconut_case_match_to_13[0]  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            if _coconut_match_set_name_arng is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
                arng = _coconut_case_match_to_13[1]  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
                ch_repr = _coconut_case_match_to_13[2]  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
        if _coconut_case_match_check_13 and not (len(ch_splitter(ch_repr)) == 3):  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            _coconut_case_match_check_13 = False  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
        if _coconut_case_match_check_13:  #                          b=imdef.__class__(dtype,arng,"RGBA","0_1"),
            return ([DataEdge(a=imdef, b=imdef.__class__(dtype, arng, "RGB", "0_1"), f=lambda a: a, cost=10, name="view {_coconut_format_0} as RGB ".format(_coconut_format_0=(ch_repr))), ])  #             return [DataEdge(a=imdef,
@to_imagedef  #                          b=imdef.__class__(dtype,arng,"RGB","0_1"),
def change_value_range(imdef: 'ImageDef'):  # def change_value_range(imdef:ImageDef):
    _coconut_case_match_to_14 = imdef  #     case imdef:
    _coconut_case_match_check_14 = False  #     case imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch_repr = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_14, TensorLike)) and (_coconut.len(_coconut_case_match_to_14) == 4) and (_coconut_case_match_to_14[3] == VR_0_255):  #     case imdef:
        _coconut_match_set_name_arng = _coconut_case_match_to_14[1]  #     case imdef:
        _coconut_match_set_name_ch_repr = _coconut_case_match_to_14[2]  #     case imdef:
        _coconut_case_match_check_14 = True  #     case imdef:
    if _coconut_case_match_check_14:  #     case imdef:
        _coconut_case_match_check_14 = False  #     case imdef:
        if (not _coconut_case_match_check_14) and (_coconut.isinstance(_coconut_case_match_to_14, TensorLike)) and (_coconut.len(_coconut_case_match_to_14) == 4) and (_coconut_case_match_to_14[0] == "float32"):  #     case imdef:
            _coconut_case_match_check_14 = True  #     case imdef:
        if (not _coconut_case_match_check_14) and (_coconut.isinstance(_coconut_case_match_to_14, TensorLike)) and (_coconut.len(_coconut_case_match_to_14) == 4) and (_coconut_case_match_to_14[0] == "float64"):  #     case imdef:
            _coconut_case_match_check_14 = True  #     case imdef:

    if _coconut_case_match_check_14:  #     case imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     case imdef:
            arng = _coconut_case_match_to_14[1]  #     case imdef:
        if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #     case imdef:
            ch_repr = _coconut_case_match_to_14[2]  #     case imdef:
    if _coconut_case_match_check_14:  #     case imdef:
        return ([DataEdge(a=imdef, b=imdef.__class__(imdef.dtype, arng, ch_repr, VR_0_1), f=lambda a: a / 255.0, cost=len(ch_repr), name="0-255 to 0-1"), ])  #             return [DataEdge(a=imdef,
    if not _coconut_case_match_check_14:  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
        _coconut_match_set_name_arng = _coconut_sentinel  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
        _coconut_match_set_name_ch_repr = _coconut_sentinel  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
        if (_coconut.isinstance(_coconut_case_match_to_14, TensorLike)) and (_coconut.len(_coconut_case_match_to_14) == 4) and (_coconut_case_match_to_14[3] == VR_0_1):  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            _coconut_match_set_name_arng = _coconut_case_match_to_14[1]  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            _coconut_match_set_name_ch_repr = _coconut_case_match_to_14[2]  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            _coconut_case_match_check_14 = True  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
        if _coconut_case_match_check_14:  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            _coconut_case_match_check_14 = False  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            if (not _coconut_case_match_check_14) and (_coconut.isinstance(_coconut_case_match_to_14, TensorLike)) and (_coconut.len(_coconut_case_match_to_14) == 4) and (_coconut_case_match_to_14[0] == "float32"):  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
                _coconut_case_match_check_14 = True  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            if (not _coconut_case_match_check_14) and (_coconut.isinstance(_coconut_case_match_to_14, TensorLike)) and (_coconut.len(_coconut_case_match_to_14) == 4) and (_coconut_case_match_to_14[0] == "float64"):  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
                _coconut_case_match_check_14 = True  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),

        if _coconut_case_match_check_14:  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            if _coconut_match_set_name_arng is not _coconut_sentinel:  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
                arng = _coconut_case_match_to_14[1]  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            if _coconut_match_set_name_ch_repr is not _coconut_sentinel:  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
                ch_repr = _coconut_case_match_to_14[2]  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
        if _coconut_case_match_check_14:  #                          b=imdef.__class__(imdef.dtype,arng,ch_repr,VR_0_1),
            return ([DataEdge(a=imdef, b=imdef.__class__(imdef.dtype, arng, ch_repr, VR_0_255), f=lambda a: a * 255.0, cost=len(ch_repr), name="0-1 to 0-255"), ])  #             return [DataEdge(a=imdef,
    return ([])  #     return []

def xyza_to_rgba(xyza):  # def xyza_to_rgba(xyza):
    xyz = xyza[:3]  #     xyz = xyza[:3]
    a = xyza[[3, ]]  #     a = xyza[[3]]
    rgb = (xyz + 1) / 2  #     rgb = (xyz+1)/2
    return (np.concatenate((rgb, a), axis=0))  #     return np.concatenate((rgb,a),axis=0)
def xyz_to_rgb(xyz):  # def xyz_to_rgb(xyz):
    return ((xyz + 1) / 2)  #     return (xyz+1)/2
def rgb_to_xyz(rgb):  # def rgb_to_xyz(rgb):
    return ((rgb * 2) - 1)  #     return (rgb*2)-1
def rgba_to_xyza(rgba):  # def rgba_to_xyza(rgba):
    rgb = rgba[:3]  #     rgb = rgba[:3]
    a = rgba[[3, ]]  #     a = rgba[[3]]
    xyz = (rgb * 2) - 1  #     xyz = (rgb*2)-1
    return (np.concatenate((xyz, a), axis=0))  #     return np.concatenate((xyz,a),axis=0)

def rule_xyz_to_rgb(imdef):  # def rule_xyz_to_rgb(imdef):
    _coconut_case_match_to_15 = imdef  #     case imdef:
    _coconut_case_match_check_15 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_15, ImageDef)) and (_coconut.len(_coconut_case_match_to_15) == 2) and (_coconut.isinstance(_coconut_case_match_to_15[0], Numpy)) and (_coconut.len(_coconut_case_match_to_15[0]) == 4) and (_coconut_case_match_to_15[0][1] == "CHW") and (_coconut_case_match_to_15[0][2] == "XYZA") and (_coconut_case_match_to_15[0][3] == "-1_1"):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_15[0][0]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_15[1]  #     case imdef:
        _coconut_case_match_check_15 = True  #     case imdef:
    if _coconut_case_match_check_15:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_15[0][0]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_15[1]  #     case imdef:
    if _coconut_case_match_check_15:  #     case imdef:
        return ([(xyza_to_rgba, ImageDef(Numpy(dtype, "CHW", "RGBA", VR_0_1), meta), 2, "xyza_to_rgba"), ])  #             return [
    if not _coconut_case_match_check_15:  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
        _coconut_match_set_name_meta = _coconut_sentinel  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
        if (_coconut.isinstance(_coconut_case_match_to_15, ImageDef)) and (_coconut.len(_coconut_case_match_to_15) == 2) and (_coconut.isinstance(_coconut_case_match_to_15[0], Numpy)) and (_coconut.len(_coconut_case_match_to_15[0]) == 4) and (_coconut_case_match_to_15[0][1] == "CHW") and (_coconut_case_match_to_15[0][2] == "XYZ") and (_coconut_case_match_to_15[0][3] == "-1_1"):  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
            _coconut_match_set_name_dtype = _coconut_case_match_to_15[0][0]  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
            _coconut_match_set_name_meta = _coconut_case_match_to_15[1]  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
            _coconut_case_match_check_15 = True  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
        if _coconut_case_match_check_15:  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
                dtype = _coconut_case_match_to_15[0][0]  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
                meta = _coconut_case_match_to_15[1]  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
        if _coconut_case_match_check_15:  #                 (xyza_to_rgba,ImageDef(Numpy(dtype,"CHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba")
            return ([(xyz_to_rgb, ImageDef(Numpy(dtype, "CHW", "RGB", VR_0_1), meta), 2, "xyz_to_rgb"), ])  #             return [
    if not _coconut_case_match_check_15:  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
        _coconut_match_set_name_meta = _coconut_sentinel  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
        if (_coconut.isinstance(_coconut_case_match_to_15, ImageDef)) and (_coconut.len(_coconut_case_match_to_15) == 2) and (_coconut.isinstance(_coconut_case_match_to_15[0], Numpy)) and (_coconut.len(_coconut_case_match_to_15[0]) == 4) and (_coconut_case_match_to_15[0][1] == "CHW") and (_coconut_case_match_to_15[0][2] == "RGBA") and (_coconut_case_match_to_15[0][3] == VR_0_1):  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
            _coconut_match_set_name_dtype = _coconut_case_match_to_15[0][0]  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
            _coconut_match_set_name_meta = _coconut_case_match_to_15[1]  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
            _coconut_case_match_check_15 = True  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
        if _coconut_case_match_check_15:  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
                dtype = _coconut_case_match_to_15[0][0]  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
                meta = _coconut_case_match_to_15[1]  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
        if _coconut_case_match_check_15:  #                 (xyz_to_rgb,ImageDef(Numpy(dtype,"CHW","RGB",VR_0_1),meta),2,"xyz_to_rgb")
            return ([(rgba_to_xyza, ImageDef(Numpy(dtype, "CHW", "XYZA", "-1_1"), meta), 2, "rgba_to_xyza"), ])  #             return [
    if not _coconut_case_match_check_15:  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
        _coconut_match_set_name_meta = _coconut_sentinel  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
        if (_coconut.isinstance(_coconut_case_match_to_15, ImageDef)) and (_coconut.len(_coconut_case_match_to_15) == 2) and (_coconut.isinstance(_coconut_case_match_to_15[0], Numpy)) and (_coconut.len(_coconut_case_match_to_15[0]) == 4) and (_coconut_case_match_to_15[0][1] == "CHW") and (_coconut_case_match_to_15[0][2] == "RGB") and (_coconut_case_match_to_15[0][3] == VR_0_1):  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
            _coconut_match_set_name_dtype = _coconut_case_match_to_15[0][0]  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
            _coconut_match_set_name_meta = _coconut_case_match_to_15[1]  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
            _coconut_case_match_check_15 = True  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
        if _coconut_case_match_check_15:  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
                dtype = _coconut_case_match_to_15[0][0]  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
                meta = _coconut_case_match_to_15[1]  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
        if _coconut_case_match_check_15:  #                 (rgba_to_xyza,ImageDef(Numpy(dtype,"CHW","XYZA","-1_1"),meta),2,"rgba_to_xyza")
            return ([(rgb_to_xyz, ImageDef(Numpy(dtype, "CHW", "XYZ", "-1_1"), meta), 2, "rgb_to_xyz"), ])  #             return [

def b_xyza_to_rgba(xyza):  # def b_xyza_to_rgba(xyza):
    xyz = xyza[:, :3]  #     xyz = xyza[:,:3]
    a = xyza[:, [3, ]]  #     a = xyza[:,[3]]
    rgb = (xyz + 1) / 2  #     rgb = (xyz+1)/2
    return (np.concatenate((rgb, a), axis=1))  #     return np.concatenate((rgb,a),axis=1)
def b_xyz_to_rgb(xyz):  # def b_xyz_to_rgb(xyz):
    return ((xyz + 1) / 2)  #     return (xyz+1)/2
def b_rgb_to_xyz(rgb):  # def b_rgb_to_xyz(rgb):
    return ((rgb * 2) - 1)  #     return (rgb*2)-1
def b_rgba_to_xyza(rgba):  # def b_rgba_to_xyza(rgba):
    rgb = rgba[:, :3]  #     rgb = rgba[:,:3]
    a = rgba[:, [3, ]]  #     a = rgba[:,[3]]
    xyz = (rgb * 2) - 1  #     xyz = (rgb*2)-1
    return (np.concatenate((xyz, a), axis=1))  #     return np.concatenate((xyz,a),axis=1)

def rule_batch_xyz_to_rgb(imdef):  # def rule_batch_xyz_to_rgb(imdef):
    _coconut_case_match_to_16 = imdef  #     case imdef:
    _coconut_case_match_check_16 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_16, ImageDef)) and (_coconut.len(_coconut_case_match_to_16) == 2) and (_coconut.isinstance(_coconut_case_match_to_16[0], Numpy)) and (_coconut.len(_coconut_case_match_to_16[0]) == 4) and (_coconut_case_match_to_16[0][1] == "BCHW") and (_coconut_case_match_to_16[0][2] == "XYZA") and (_coconut_case_match_to_16[0][3] == "-1_1"):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_16[0][0]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_16[1]  #     case imdef:
        _coconut_case_match_check_16 = True  #     case imdef:
    if _coconut_case_match_check_16:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_16[0][0]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_16[1]  #     case imdef:
    if _coconut_case_match_check_16:  #     case imdef:
        return ([(b_xyza_to_rgba, ImageDef(Numpy(dtype, "BCHW", "RGBA", VR_0_1), meta), 2, "xyza_to_rgba(batch)"), ])  #             return [
    if not _coconut_case_match_check_16:  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
        _coconut_match_set_name_meta = _coconut_sentinel  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
        if (_coconut.isinstance(_coconut_case_match_to_16, ImageDef)) and (_coconut.len(_coconut_case_match_to_16) == 2) and (_coconut.isinstance(_coconut_case_match_to_16[0], Numpy)) and (_coconut.len(_coconut_case_match_to_16[0]) == 4) and (_coconut_case_match_to_16[0][1] == "BCHW") and (_coconut_case_match_to_16[0][2] == "XYZ") and (_coconut_case_match_to_16[0][3] == "-1_1"):  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
            _coconut_match_set_name_dtype = _coconut_case_match_to_16[0][0]  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
            _coconut_match_set_name_meta = _coconut_case_match_to_16[1]  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
            _coconut_case_match_check_16 = True  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
        if _coconut_case_match_check_16:  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
                dtype = _coconut_case_match_to_16[0][0]  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
                meta = _coconut_case_match_to_16[1]  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
        if _coconut_case_match_check_16:  #                 (b_xyza_to_rgba,ImageDef(Numpy(dtype,"BCHW","RGBA",VR_0_1),meta),2,"xyza_to_rgba(batch)")
            return ([(b_xyz_to_rgb, ImageDef(Numpy(dtype, "BCHW", "RGB", VR_0_1), meta), 2, "xyz_to_rgb(batch)"), ])  #             return [
    if not _coconut_case_match_check_16:  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
        _coconut_match_set_name_meta = _coconut_sentinel  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
        if (_coconut.isinstance(_coconut_case_match_to_16, ImageDef)) and (_coconut.len(_coconut_case_match_to_16) == 2) and (_coconut.isinstance(_coconut_case_match_to_16[0], Numpy)) and (_coconut.len(_coconut_case_match_to_16[0]) == 4) and (_coconut_case_match_to_16[0][1] == "BCHW") and (_coconut_case_match_to_16[0][2] == "RGBA") and (_coconut_case_match_to_16[0][3] == VR_0_1):  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
            _coconut_match_set_name_dtype = _coconut_case_match_to_16[0][0]  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
            _coconut_match_set_name_meta = _coconut_case_match_to_16[1]  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
            _coconut_case_match_check_16 = True  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
        if _coconut_case_match_check_16:  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
                dtype = _coconut_case_match_to_16[0][0]  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
                meta = _coconut_case_match_to_16[1]  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
        if _coconut_case_match_check_16:  #                 (b_xyz_to_rgb,ImageDef(Numpy(dtype,"BCHW","RGB",VR_0_1),meta),2,"xyz_to_rgb(batch)")
            return ([(b_rgba_to_xyza, ImageDef(Numpy(dtype, "BCHW", "XYZA", "-1_1"), meta), 2, "rgba_to_xyza(batch)"), ])  #             return [
    if not _coconut_case_match_check_16:  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
        _coconut_match_set_name_meta = _coconut_sentinel  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
        if (_coconut.isinstance(_coconut_case_match_to_16, ImageDef)) and (_coconut.len(_coconut_case_match_to_16) == 2) and (_coconut.isinstance(_coconut_case_match_to_16[0], Numpy)) and (_coconut.len(_coconut_case_match_to_16[0]) == 4) and (_coconut_case_match_to_16[0][1] == "BCHW") and (_coconut_case_match_to_16[0][2] == "RGB") and (_coconut_case_match_to_16[0][3] == VR_0_1):  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
            _coconut_match_set_name_dtype = _coconut_case_match_to_16[0][0]  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
            _coconut_match_set_name_meta = _coconut_case_match_to_16[1]  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
            _coconut_case_match_check_16 = True  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
        if _coconut_case_match_check_16:  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
                dtype = _coconut_case_match_to_16[0][0]  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
                meta = _coconut_case_match_to_16[1]  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
        if _coconut_case_match_check_16:  #                 (b_rgba_to_xyza,ImageDef(Numpy(dtype,"BCHW","XYZA","-1_1"),meta),2,"rgba_to_xyza(batch)")
            return ([(b_rgb_to_xyz, ImageDef(Numpy(dtype, "BCHW", "XYZ", "-1_1"), meta), 2, "rgb_to_xyz(batch)"), ])  #             return [



_conversions = [to_PILImages, to_numpy, to_torch, change_dtype, change_arrange, select_channel, drop_channel, en_batch, change_value_range, drop_alpha, to_rgba, drop_meta, de_batch, RGB_to_YCbCr]  # _conversions =[


@memoize(1024)  # @memoize(1024)
def _edges(imdef):  # def _edges(imdef):
    res = []  #     res = []
    for f in _conversions:  #     for f in _conversions:
        edges = f(imdef)  #         edges = f(imdef)
        if edges is not None:  #         if edges is not None:
            res += edges  #             res += edges
    return (res)  #     return res




@memoize(1024)  # @memoize(1024)
def str_to_img_def(query):  # def str_to_img_def(query):
    """
    ex1: 'numpy,float32,BCHW,RGB,0_255 | hello,world'
    ex2: 'torch,float32,BCHW,RGBA,0_1'
    ex3: 'image,RGBA,RGBA'
    ex4: 'images,RGB,RGB' => ImageDef(PILImage("RGB","RGB"),{shape:(None,None,3)})
    ex5: 'image,RGBA,RGBA,128:128:3' => ImageDef(PILImage("RGB","RGB"),{shape:(128,128,3)})
    """  #     """
    vrs = {"0_255": VR_0_255, "0_1": VR_0_1, "None": VR_None}  #     vrs = {
    query = query.replace(" ", "")  #     query = query.replace(" ","")
    def arng_to_shape(arng, ch_repr):  #     def arng_to_shape(arng,ch_repr):
        ch = len(ch_splitter(ch_repr))  #         ch = len(ch_splitter(ch_repr))
        c_idx = arng.find("C")  #         c_idx = arng.find("C")
        shape = [None, ] * len(arng)  #         shape = [None]*len(arng)
        if c_idx != -1:  #         if c_idx != -1:
            shape[c_idx] = ch  #             shape[c_idx] = ch
#logger.info(f"{arng},{ch_repr}->{tuple(shape)}")
        return (tuple(shape))  #         return tuple(shape)

    def shape_str_to_shape(ss):  #     def shape_str_to_shape(ss):
        tokens = ss.split(":")  #         tokens = ss.split(":")
        return (tuple([int(t) if t != "None" else None for t in tokens]))  #         return tuple([int(t) if t!="None" else None for t in tokens])

    def query_to_data_type(query):  #     def query_to_data_type(query):
        _coconut_case_match_to_17 = query.split(",")  #         case query.split(","):
        _coconut_case_match_check_17 = False  #         case query.split(","):
        _coconut_match_set_name_dtype = _coconut_sentinel  #         case query.split(","):
        _coconut_match_set_name_arng = _coconut_sentinel  #         case query.split(","):
        _coconut_match_set_name_ch = _coconut_sentinel  #         case query.split(","):
        _coconut_match_set_name_vr = _coconut_sentinel  #         case query.split(","):
        if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 5) and (_coconut_case_match_to_17[0] == "numpy"):  #         case query.split(","):
            _coconut_match_set_name_dtype = _coconut_case_match_to_17[1]  #         case query.split(","):
            _coconut_match_set_name_arng = _coconut_case_match_to_17[2]  #         case query.split(","):
            _coconut_match_set_name_ch = _coconut_case_match_to_17[3]  #         case query.split(","):
            _coconut_match_set_name_vr = _coconut_case_match_to_17[4]  #         case query.split(","):
            _coconut_case_match_check_17 = True  #         case query.split(","):
        if _coconut_case_match_check_17:  #         case query.split(","):
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #         case query.split(","):
                dtype = _coconut_case_match_to_17[1]  #         case query.split(","):
            if _coconut_match_set_name_arng is not _coconut_sentinel:  #         case query.split(","):
                arng = _coconut_case_match_to_17[2]  #         case query.split(","):
            if _coconut_match_set_name_ch is not _coconut_sentinel:  #         case query.split(","):
                ch = _coconut_case_match_to_17[3]  #         case query.split(","):
            if _coconut_match_set_name_vr is not _coconut_sentinel:  #         case query.split(","):
                vr = _coconut_case_match_to_17[4]  #         case query.split(","):
        if _coconut_case_match_check_17:  #         case query.split(","):
            return (Numpy(dtype, arng, ch, vrs[vr] if vr in vrs else vr), arng_to_shape(arng, ch))  #                 return Numpy(dtype,arng,ch,vrs[vr] if vr in vrs else vr),arng_to_shape(arng,ch)
        if not _coconut_case_match_check_17:  #             match ["numpy",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_dtype = _coconut_sentinel  #             match ["numpy",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_arng = _coconut_sentinel  #             match ["numpy",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["numpy",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_vr = _coconut_sentinel  #             match ["numpy",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_shape = _coconut_sentinel  #             match ["numpy",dtype,arng,ch,vr,shape]:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 6) and (_coconut_case_match_to_17[0] == "numpy"):  #             match ["numpy",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_dtype = _coconut_case_match_to_17[1]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_arng = _coconut_case_match_to_17[2]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[3]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_vr = _coconut_case_match_to_17[4]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_shape = _coconut_case_match_to_17[5]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                _coconut_case_match_check_17 = True  #             match ["numpy",dtype,arng,ch,vr,shape]:
            if _coconut_case_match_check_17:  #             match ["numpy",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_dtype is not _coconut_sentinel:  #             match ["numpy",dtype,arng,ch,vr,shape]:
                    dtype = _coconut_case_match_to_17[1]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match ["numpy",dtype,arng,ch,vr,shape]:
                    arng = _coconut_case_match_to_17[2]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["numpy",dtype,arng,ch,vr,shape]:
                    ch = _coconut_case_match_to_17[3]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_vr is not _coconut_sentinel:  #             match ["numpy",dtype,arng,ch,vr,shape]:
                    vr = _coconut_case_match_to_17[4]  #             match ["numpy",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_shape is not _coconut_sentinel:  #             match ["numpy",dtype,arng,ch,vr,shape]:
                    shape = _coconut_case_match_to_17[5]  #             match ["numpy",dtype,arng,ch,vr,shape]:
            if _coconut_case_match_check_17:  #             match ["numpy",dtype,arng,ch,vr,shape]:
                return (Numpy(dtype, arng, ch, vrs[vr] if vr in vrs else vr), shape_str_to_shape(shape))  #                 return Numpy(dtype,arng,ch,vrs[vr] if vr in vrs else vr),shape_str_to_shape(shape)
        if not _coconut_case_match_check_17:  #             match ["torch",dtype,arng,ch,vr]:
            _coconut_match_set_name_dtype = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr]:
            _coconut_match_set_name_arng = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr]:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr]:
            _coconut_match_set_name_vr = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr]:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 5) and (_coconut_case_match_to_17[0] == "torch"):  #             match ["torch",dtype,arng,ch,vr]:
                _coconut_match_set_name_dtype = _coconut_case_match_to_17[1]  #             match ["torch",dtype,arng,ch,vr]:
                _coconut_match_set_name_arng = _coconut_case_match_to_17[2]  #             match ["torch",dtype,arng,ch,vr]:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[3]  #             match ["torch",dtype,arng,ch,vr]:
                _coconut_match_set_name_vr = _coconut_case_match_to_17[4]  #             match ["torch",dtype,arng,ch,vr]:
                _coconut_case_match_check_17 = True  #             match ["torch",dtype,arng,ch,vr]:
            if _coconut_case_match_check_17:  #             match ["torch",dtype,arng,ch,vr]:
                if _coconut_match_set_name_dtype is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr]:
                    dtype = _coconut_case_match_to_17[1]  #             match ["torch",dtype,arng,ch,vr]:
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr]:
                    arng = _coconut_case_match_to_17[2]  #             match ["torch",dtype,arng,ch,vr]:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr]:
                    ch = _coconut_case_match_to_17[3]  #             match ["torch",dtype,arng,ch,vr]:
                if _coconut_match_set_name_vr is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr]:
                    vr = _coconut_case_match_to_17[4]  #             match ["torch",dtype,arng,ch,vr]:
            if _coconut_case_match_check_17:  #             match ["torch",dtype,arng,ch,vr]:
                return (Torch(dtype, arng, ch, vrs[vr] if vr in vrs else vr), arng_to_shape(arng, ch))  #                 return Torch(dtype,arng,ch,vrs[vr] if vr in vrs else vr),arng_to_shape(arng,ch)
        if not _coconut_case_match_check_17:  #             match ["torch",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_dtype = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_arng = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_vr = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr,shape]:
            _coconut_match_set_name_shape = _coconut_sentinel  #             match ["torch",dtype,arng,ch,vr,shape]:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 6) and (_coconut_case_match_to_17[0] == "torch"):  #             match ["torch",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_dtype = _coconut_case_match_to_17[1]  #             match ["torch",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_arng = _coconut_case_match_to_17[2]  #             match ["torch",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[3]  #             match ["torch",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_vr = _coconut_case_match_to_17[4]  #             match ["torch",dtype,arng,ch,vr,shape]:
                _coconut_match_set_name_shape = _coconut_case_match_to_17[5]  #             match ["torch",dtype,arng,ch,vr,shape]:
                _coconut_case_match_check_17 = True  #             match ["torch",dtype,arng,ch,vr,shape]:
            if _coconut_case_match_check_17:  #             match ["torch",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_dtype is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr,shape]:
                    dtype = _coconut_case_match_to_17[1]  #             match ["torch",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr,shape]:
                    arng = _coconut_case_match_to_17[2]  #             match ["torch",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr,shape]:
                    ch = _coconut_case_match_to_17[3]  #             match ["torch",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_vr is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr,shape]:
                    vr = _coconut_case_match_to_17[4]  #             match ["torch",dtype,arng,ch,vr,shape]:
                if _coconut_match_set_name_shape is not _coconut_sentinel:  #             match ["torch",dtype,arng,ch,vr,shape]:
                    shape = _coconut_case_match_to_17[5]  #             match ["torch",dtype,arng,ch,vr,shape]:
            if _coconut_case_match_check_17:  #             match ["torch",dtype,arng,ch,vr,shape]:
                return (Torch(dtype, arng, ch, vrs[vr] if vr in vrs else vr), shape_str_to_shape(shape))  #                 return Torch(dtype,arng,ch,vrs[vr] if vr in vrs else vr),shape_str_to_shape(shape)
        if not _coconut_case_match_check_17:  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
            _coconut_match_set_name_mode = _coconut_sentinel  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 3) and (_coconut_case_match_to_17[0] == "image"):  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_match_set_name_mode = _coconut_case_match_to_17[1]  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[2]  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_case_match_check_17 = True  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
            if _coconut_case_match_check_17:  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                    mode = _coconut_case_match_to_17[1]  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                    ch = _coconut_case_match_to_17[2]  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
            if _coconut_case_match_check_17 and not (len(ch_splitter(ch)) > 1):  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_case_match_check_17 = False  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
            if _coconut_case_match_check_17:  #             match ["image",mode,ch] if len(ch_splitter(ch)) > 1:
                return (PILImage(mode, ch), (None, None, len(ch_splitter(ch))))  #                 return PILImage(mode,ch),(None,None,len(ch_splitter(ch)))
        if not _coconut_case_match_check_17:  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
            _coconut_match_set_name_mode = _coconut_sentinel  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 3) and (_coconut_case_match_to_17[0] == "image"):  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                _coconut_match_set_name_mode = _coconut_case_match_to_17[1]  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[2]  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                _coconut_case_match_check_17 = True  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
            if _coconut_case_match_check_17:  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                    mode = _coconut_case_match_to_17[1]  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                    ch = _coconut_case_match_to_17[2]  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
            if _coconut_case_match_check_17 and not (len(ch_splitter(ch)) == 1):  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                _coconut_case_match_check_17 = False  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
            if _coconut_case_match_check_17:  #             match ["image",mode,ch] if len(ch_splitter(ch)) ==1:
                return (PILImage(mode, ch), (None, None))  #                 return PILImage(mode,ch),(None,None)
        if not _coconut_case_match_check_17:  #             match ["image",mode,ch,shape]:
            _coconut_match_set_name_mode = _coconut_sentinel  #             match ["image",mode,ch,shape]:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["image",mode,ch,shape]:
            _coconut_match_set_name_shape = _coconut_sentinel  #             match ["image",mode,ch,shape]:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 4) and (_coconut_case_match_to_17[0] == "image"):  #             match ["image",mode,ch,shape]:
                _coconut_match_set_name_mode = _coconut_case_match_to_17[1]  #             match ["image",mode,ch,shape]:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[2]  #             match ["image",mode,ch,shape]:
                _coconut_match_set_name_shape = _coconut_case_match_to_17[3]  #             match ["image",mode,ch,shape]:
                _coconut_case_match_check_17 = True  #             match ["image",mode,ch,shape]:
            if _coconut_case_match_check_17:  #             match ["image",mode,ch,shape]:
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match ["image",mode,ch,shape]:
                    mode = _coconut_case_match_to_17[1]  #             match ["image",mode,ch,shape]:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["image",mode,ch,shape]:
                    ch = _coconut_case_match_to_17[2]  #             match ["image",mode,ch,shape]:
                if _coconut_match_set_name_shape is not _coconut_sentinel:  #             match ["image",mode,ch,shape]:
                    shape = _coconut_case_match_to_17[3]  #             match ["image",mode,ch,shape]:
            if _coconut_case_match_check_17:  #             match ["image",mode,ch,shape]:
                return (PILImage(mode, ch), shape_str_to_shape(shape))  #                 return PILImage(mode,ch),shape_str_to_shape(shape)
        if not _coconut_case_match_check_17:  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
            _coconut_match_set_name_mode = _coconut_sentinel  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 3) and (_coconut_case_match_to_17[0] == "images"):  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_match_set_name_mode = _coconut_case_match_to_17[1]  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[2]  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_case_match_check_17 = True  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
            if _coconut_case_match_check_17:  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                    mode = _coconut_case_match_to_17[1]  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                    ch = _coconut_case_match_to_17[2]  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
            if _coconut_case_match_check_17 and not (len(ch_splitter(ch)) > 1):  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                _coconut_case_match_check_17 = False  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
            if _coconut_case_match_check_17:  #             match ["images",mode,ch] if len(ch_splitter(ch)) > 1:
                return (PILImages(mode, ch), (None, None, None, len(ch_splitter(ch))))  #                 return PILImages(mode,ch),(None,None,None,len(ch_splitter(ch)))
        if not _coconut_case_match_check_17:  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
            _coconut_match_set_name_mode = _coconut_sentinel  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 3) and (_coconut_case_match_to_17[0] == "images"):  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                _coconut_match_set_name_mode = _coconut_case_match_to_17[1]  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[2]  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                _coconut_case_match_check_17 = True  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
            if _coconut_case_match_check_17:  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                    mode = _coconut_case_match_to_17[1]  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                    ch = _coconut_case_match_to_17[2]  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
            if _coconut_case_match_check_17 and not (len(ch_splitter(ch)) == 1):  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                _coconut_case_match_check_17 = False  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
            if _coconut_case_match_check_17:  #             match ["images",mode,ch] if len(ch_splitter(ch)) == 1:
                return (PILImages(mode, ch), (None, None, None))  #                 return PILImages(mode,ch),(None,None,None)
        if not _coconut_case_match_check_17:  #             match ["images",mode,ch,shape]:
            _coconut_match_set_name_mode = _coconut_sentinel  #             match ["images",mode,ch,shape]:
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ["images",mode,ch,shape]:
            _coconut_match_set_name_shape = _coconut_sentinel  #             match ["images",mode,ch,shape]:
            if (_coconut.isinstance(_coconut_case_match_to_17, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_17) == 4) and (_coconut_case_match_to_17[0] == "images"):  #             match ["images",mode,ch,shape]:
                _coconut_match_set_name_mode = _coconut_case_match_to_17[1]  #             match ["images",mode,ch,shape]:
                _coconut_match_set_name_ch = _coconut_case_match_to_17[2]  #             match ["images",mode,ch,shape]:
                _coconut_match_set_name_shape = _coconut_case_match_to_17[3]  #             match ["images",mode,ch,shape]:
                _coconut_case_match_check_17 = True  #             match ["images",mode,ch,shape]:
            if _coconut_case_match_check_17:  #             match ["images",mode,ch,shape]:
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match ["images",mode,ch,shape]:
                    mode = _coconut_case_match_to_17[1]  #             match ["images",mode,ch,shape]:
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ["images",mode,ch,shape]:
                    ch = _coconut_case_match_to_17[2]  #             match ["images",mode,ch,shape]:
                if _coconut_match_set_name_shape is not _coconut_sentinel:  #             match ["images",mode,ch,shape]:
                    shape = _coconut_case_match_to_17[3]  #             match ["images",mode,ch,shape]:
            if _coconut_case_match_check_17:  #             match ["images",mode,ch,shape]:
                return (PILImages(mode, ch), shape_str_to_shape(shape))  #                 return PILImages(mode,ch),shape_str_to_shape(shape)
    _coconut_case_match_to_18 = query_to_data_type(query)  #     case query_to_data_type(query):
    _coconut_case_match_check_18 = False  #     case query_to_data_type(query):
    _coconut_match_set_name_data_type = _coconut_sentinel  #     case query_to_data_type(query):
    _coconut_match_set_name_shape = _coconut_sentinel  #     case query_to_data_type(query):
    if (_coconut.isinstance(_coconut_case_match_to_18, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_18) == 2):  #     case query_to_data_type(query):
        _coconut_match_set_name_data_type = _coconut_case_match_to_18[0]  #     case query_to_data_type(query):
        _coconut_match_set_name_shape = _coconut_case_match_to_18[1]  #     case query_to_data_type(query):
        _coconut_case_match_check_18 = True  #     case query_to_data_type(query):
    if _coconut_case_match_check_18:  #     case query_to_data_type(query):
        if _coconut_match_set_name_data_type is not _coconut_sentinel:  #     case query_to_data_type(query):
            data_type = _coconut_case_match_to_18[0]  #     case query_to_data_type(query):
        if _coconut_match_set_name_shape is not _coconut_sentinel:  #     case query_to_data_type(query):
            shape = _coconut_case_match_to_18[1]  #     case query_to_data_type(query):
    if _coconut_case_match_check_18:  #     case query_to_data_type(query):
        return (ImageDef(data_type, fdict(shape=shape)))  #             return ImageDef(data_type,fdict(shape=shape))
    if not _coconut_case_match_check_18:  #     else:
        raise RuntimeError("could not parse image def string!:{_coconut_format_0}".format(_coconut_format_0=(query)))  #         raise RuntimeError(f"could not parse image def string!:{query}")

def parse_def(img_def):  # def parse_def(img_def):
    try:  #     try:
        return (str_to_img_def(img_def) if (isinstance)(img_def, str) else img_def)  #         return str_to_img_def(img_def) if img_def `isinstance` str else img_def
    except Exception as e:  #     except Exception as e:
        return (img_def)  #         return img_def




accept_def_str = lambda f: _coconut_base_compose(parse_def, (f, 0))  # accept_def_str = f -> parse_def ..> f
def imdef_neighbors(imdef):  # def imdef_neighbors(imdef):
    return ([(e.f, e.b, e.cost, e.name) for e in _edges(imdef)])  #     return [(e.f,e.b,e.cost,e.name) for e in _edges(imdef)]

#from data_tree.coconut.convert import AutoImage,PILImage,str_to_img_def,PILImages
#from data_tree.coconut.convert import ImageDef,Torch,Numpy,TensorLike,VR_0_1,VR_None,VR_0_255

def normalize_numpy_img(ary):  # def normalize_numpy_img(ary):
    _min = ary.min()  #     _min = ary.min()
    _max = ary.max()  #     _max = ary.max()
    return (((ary - _min) / (_max - _min)))  #     return ((ary-_min)/(_max-_min))

def rule_VR_None_to_normalized(imdef):  # def rule_VR_None_to_normalized(imdef):
    _coconut_case_match_to_19 = imdef  #     case imdef:
    _coconut_case_match_check_19 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_arng = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_19, ImageDef)) and (_coconut.len(_coconut_case_match_to_19) == 2) and (_coconut.isinstance(_coconut_case_match_to_19[0], Numpy)) and (_coconut.len(_coconut_case_match_to_19[0]) == 4) and (_coconut_case_match_to_19[0][3] == VR_None):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_19[0][0]  #     case imdef:
        _coconut_match_set_name_arng = _coconut_case_match_to_19[0][1]  #     case imdef:
        _coconut_match_set_name_ch = _coconut_case_match_to_19[0][2]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_19[1]  #     case imdef:
        _coconut_case_match_check_19 = True  #     case imdef:
    if _coconut_case_match_check_19:  #     case imdef:
        _coconut_case_match_check_19 = False  #     case imdef:
        if (not _coconut_case_match_check_19) and (_coconut.isinstance(_coconut_case_match_to_19, ImageDef)) and (_coconut.len(_coconut_case_match_to_19) == 2) and (_coconut.isinstance(_coconut_case_match_to_19[0], Numpy)) and (_coconut.len(_coconut_case_match_to_19[0]) == 4) and (_coconut_case_match_to_19[0][1] == "CHW"):  #     case imdef:
            _coconut_match_set_name_dtype = _coconut_case_match_to_19[0][0]  #     case imdef:
            _coconut_match_set_name_arng = _coconut_case_match_to_19[0][1]  #     case imdef:
            _coconut_case_match_check_19 = True  #     case imdef:
        if (not _coconut_case_match_check_19) and (_coconut.isinstance(_coconut_case_match_to_19, ImageDef)) and (_coconut.len(_coconut_case_match_to_19) == 2) and (_coconut.isinstance(_coconut_case_match_to_19[0], Numpy)) and (_coconut.len(_coconut_case_match_to_19[0]) == 4) and (_coconut_case_match_to_19[0][1] == "HW"):  #     case imdef:
            _coconut_match_set_name_dtype = _coconut_case_match_to_19[0][0]  #     case imdef:
            _coconut_match_set_name_arng = _coconut_case_match_to_19[0][1]  #     case imdef:
            _coconut_case_match_check_19 = True  #     case imdef:

    if _coconut_case_match_check_19:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_19[0][0]  #     case imdef:
        if _coconut_match_set_name_arng is not _coconut_sentinel:  #     case imdef:
            arng = _coconut_case_match_to_19[0][1]  #     case imdef:
        if _coconut_match_set_name_ch is not _coconut_sentinel:  #     case imdef:
            ch = _coconut_case_match_to_19[0][2]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_19[1]  #     case imdef:
    if _coconut_case_match_check_19:  #     case imdef:
        return ([(normalize_numpy_img, ImageDef(Numpy(dtype, arng, ch, VR_0_1), meta), 1, "minmax_0_1_numpy_img"), ])  #             return [(
    if not _coconut_case_match_check_19:  #                 normalize_numpy_img,
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 normalize_numpy_img,
        _coconut_match_set_name_ch = _coconut_sentinel  #                 normalize_numpy_img,
        _coconut_match_set_name_meta = _coconut_sentinel  #                 normalize_numpy_img,
        if (_coconut.isinstance(_coconut_case_match_to_19, ImageDef)) and (_coconut.len(_coconut_case_match_to_19) == 2) and (_coconut.isinstance(_coconut_case_match_to_19[0], Numpy)) and (_coconut.len(_coconut_case_match_to_19[0]) == 4) and (_coconut_case_match_to_19[0][1] == "BCHW") and (_coconut_case_match_to_19[0][3] == VR_None):  #                 normalize_numpy_img,
            _coconut_match_set_name_dtype = _coconut_case_match_to_19[0][0]  #                 normalize_numpy_img,
            _coconut_match_set_name_ch = _coconut_case_match_to_19[0][2]  #                 normalize_numpy_img,
            _coconut_match_set_name_meta = _coconut_case_match_to_19[1]  #                 normalize_numpy_img,
            _coconut_case_match_check_19 = True  #                 normalize_numpy_img,
        if _coconut_case_match_check_19:  #                 normalize_numpy_img,
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 normalize_numpy_img,
                dtype = _coconut_case_match_to_19[0][0]  #                 normalize_numpy_img,
            if _coconut_match_set_name_ch is not _coconut_sentinel:  #                 normalize_numpy_img,
                ch = _coconut_case_match_to_19[0][2]  #                 normalize_numpy_img,
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 normalize_numpy_img,
                meta = _coconut_case_match_to_19[1]  #                 normalize_numpy_img,
        if _coconut_case_match_check_19:  #                 normalize_numpy_img,
            return ([(lambda batch: np.array([normalize_numpy_img(img) for img in batch]), ImageDef(Numpy(dtype, "BCHW", ch, VR_0_1), meta), 1, "batch_minmax_0_1_numpy_img"), ])  #             return [(
ms_add_ch_to_hw = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (1, s[0], s[1])))  # ms_add_ch_to_hw = (s->(1,s[0],s[1])) |> ss_to_ms$
ms_add_ch_to_bhw = (_coconut.functools.partial(_coconut.functools.partial, ss_to_ms))((lambda s: (s[0], 1, s[1], s[2])))  # ms_add_ch_to_bhw = (s->(s[0],1,s[1],s[2])) |> ss_to_ms$
def rule_add_channel(imdef):  # def rule_add_channel(imdef):
    _coconut_case_match_to_20 = imdef  #     case imdef:
    _coconut_case_match_check_20 = False  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_ch = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_20, ImageDef)) and (_coconut.len(_coconut_case_match_to_20) == 2) and (_coconut.isinstance(_coconut_case_match_to_20[0], Numpy)) and (_coconut.len(_coconut_case_match_to_20[0]) == 4) and (_coconut_case_match_to_20[0][1] == "HW"):  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_20[0][0]  #     case imdef:
        _coconut_match_set_name_ch = _coconut_case_match_to_20[0][2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_20[0][3]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_20[1]  #     case imdef:
        _coconut_case_match_check_20 = True  #     case imdef:
    if _coconut_case_match_check_20:  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_20[0][0]  #     case imdef:
        if _coconut_match_set_name_ch is not _coconut_sentinel:  #     case imdef:
            ch = _coconut_case_match_to_20[0][2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_20[0][3]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_20[1]  #     case imdef:
    if _coconut_case_match_check_20:  #     case imdef:
        return ([(lambda a: a[None], ImageDef(Numpy(dtype, "CHW", ch, vr), (ms_add_ch_to_hw)(meta)), 1, "add_channel_dim"), ])  #             return [(
    if not _coconut_case_match_check_20:  #                 a->a[None],
        _coconut_match_set_name_dtype = _coconut_sentinel  #                 a->a[None],
        _coconut_match_set_name_ch = _coconut_sentinel  #                 a->a[None],
        _coconut_match_set_name_vr = _coconut_sentinel  #                 a->a[None],
        _coconut_match_set_name_meta = _coconut_sentinel  #                 a->a[None],
        if (_coconut.isinstance(_coconut_case_match_to_20, ImageDef)) and (_coconut.len(_coconut_case_match_to_20) == 2) and (_coconut.isinstance(_coconut_case_match_to_20[0], Numpy)) and (_coconut.len(_coconut_case_match_to_20[0]) == 4) and (_coconut_case_match_to_20[0][1] == "BHW"):  #                 a->a[None],
            _coconut_match_set_name_dtype = _coconut_case_match_to_20[0][0]  #                 a->a[None],
            _coconut_match_set_name_ch = _coconut_case_match_to_20[0][2]  #                 a->a[None],
            _coconut_match_set_name_vr = _coconut_case_match_to_20[0][3]  #                 a->a[None],
            _coconut_match_set_name_meta = _coconut_case_match_to_20[1]  #                 a->a[None],
            _coconut_case_match_check_20 = True  #                 a->a[None],
        if _coconut_case_match_check_20:  #                 a->a[None],
            if _coconut_match_set_name_dtype is not _coconut_sentinel:  #                 a->a[None],
                dtype = _coconut_case_match_to_20[0][0]  #                 a->a[None],
            if _coconut_match_set_name_ch is not _coconut_sentinel:  #                 a->a[None],
                ch = _coconut_case_match_to_20[0][2]  #                 a->a[None],
            if _coconut_match_set_name_vr is not _coconut_sentinel:  #                 a->a[None],
                vr = _coconut_case_match_to_20[0][3]  #                 a->a[None],
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 a->a[None],
                meta = _coconut_case_match_to_20[1]  #                 a->a[None],
        if _coconut_case_match_check_20:  #                 a->a[None],
            return ([(lambda a: a[:, None], ImageDef(Numpy(dtype, "BCHW", ch, vr), (ms_add_ch_to_bhw)(meta)), 1, "add_channel_dim"), ])  #             return [(

def rule_swap_RGB_BGR(imdef):  # def rule_swap_RGB_BGR(imdef):
    _coconut_case_match_to_21 = imdef  #     case imdef:
    _coconut_case_match_check_21 = False  #     case imdef:
    _coconut_match_set_name_tl = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_dtype = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_rgb_order = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_vr = _coconut_sentinel  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_21, ImageDef)) and (_coconut.len(_coconut_case_match_to_21) == 2) and (_coconut.isinstance(_coconut_case_match_to_21[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_21[0]) == 4) and (_coconut_case_match_to_21[0][1] == "BHWC"):  #     case imdef:
        _coconut_match_set_name_tl = _coconut_case_match_to_21[0]  #     case imdef:
        _coconut_match_set_name_dtype = _coconut_case_match_to_21[0][0]  #     case imdef:
        _coconut_match_set_name_rgb_order = _coconut_case_match_to_21[0][2]  #     case imdef:
        _coconut_match_set_name_vr = _coconut_case_match_to_21[0][3]  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_21[1]  #     case imdef:
        _coconut_case_match_check_21 = True  #     case imdef:
    if _coconut_case_match_check_21:  #     case imdef:
        _coconut_case_match_check_21 = False  #     case imdef:
        if (not _coconut_case_match_check_21) and (_coconut.isinstance(_coconut_case_match_to_21, ImageDef)) and (_coconut.len(_coconut_case_match_to_21) == 2) and (_coconut.isinstance(_coconut_case_match_to_21[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_21[0]) == 4) and (_coconut_case_match_to_21[0][1] == "BHWC") and (_coconut_case_match_to_21[0][2] == "RGB"):  #     case imdef:
            _coconut_match_set_name_tl = _coconut_case_match_to_21[0]  #     case imdef:
            _coconut_match_set_name_dtype = _coconut_case_match_to_21[0][0]  #     case imdef:
            _coconut_match_set_name_rgb_order = _coconut_case_match_to_21[0][2]  #     case imdef:
            _coconut_case_match_check_21 = True  #     case imdef:
        if (not _coconut_case_match_check_21) and (_coconut.isinstance(_coconut_case_match_to_21, ImageDef)) and (_coconut.len(_coconut_case_match_to_21) == 2) and (_coconut.isinstance(_coconut_case_match_to_21[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_21[0]) == 4) and (_coconut_case_match_to_21[0][1] == "BHWC") and (_coconut_case_match_to_21[0][2] == "BGR"):  #     case imdef:
            _coconut_match_set_name_tl = _coconut_case_match_to_21[0]  #     case imdef:
            _coconut_match_set_name_dtype = _coconut_case_match_to_21[0][0]  #     case imdef:
            _coconut_match_set_name_rgb_order = _coconut_case_match_to_21[0][2]  #     case imdef:
            _coconut_case_match_check_21 = True  #     case imdef:

    if _coconut_case_match_check_21:  #     case imdef:
        if _coconut_match_set_name_tl is not _coconut_sentinel:  #     case imdef:
            tl = _coconut_case_match_to_21[0]  #     case imdef:
        if _coconut_match_set_name_dtype is not _coconut_sentinel:  #     case imdef:
            dtype = _coconut_case_match_to_21[0][0]  #     case imdef:
        if _coconut_match_set_name_rgb_order is not _coconut_sentinel:  #     case imdef:
            rgb_order = _coconut_case_match_to_21[0][2]  #     case imdef:
        if _coconut_match_set_name_vr is not _coconut_sentinel:  #     case imdef:
            vr = _coconut_case_match_to_21[0][3]  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_21[1]  #     case imdef:
    if _coconut_case_match_check_21:  #     case imdef:
        return ([(lambda a: a[:, :, :, [2, 1, 0]], ImageDef(tl.__class__(dtype, "BHWC", "RGB" if rgb_order.startswith("B") else "BGR", vr), meta), 1, "swap rgb or bgr"), ])  #             return [(

def rule_BGR_to_LAB(imdef):  # def rule_BGR_to_LAB(imdef):
#logger.info(f"trying to load skimage")
#from skimage import color
#from skimage.color.colorconv import lab2rgb,rgb2lab
#logger.info("loaded skimage")
    from data_tree.colorconv import rgb2lab  #     from data_tree.colorconv import rgb2lab,lab2rgb
    from data_tree.colorconv import lab2rgb  #     from data_tree.colorconv import rgb2lab,lab2rgb
    _coconut_case_match_to_22 = imdef  #     case imdef:
    _coconut_case_match_check_22 = False  #     case imdef:
    _coconut_match_set_name_meta = _coconut_sentinel  #     case imdef:
    if (_coconut.isinstance(_coconut_case_match_to_22, ImageDef)) and (_coconut.len(_coconut_case_match_to_22) == 2) and (_coconut.isinstance(_coconut_case_match_to_22[0], Numpy)) and (_coconut.len(_coconut_case_match_to_22[0]) == 4) and (_coconut_case_match_to_22[0][0] == "float32") and (_coconut_case_match_to_22[0][1] == "HWC") and (_coconut_case_match_to_22[0][2] == "BGR") and (_coconut_case_match_to_22[0][3] == VR_0_1):  #     case imdef:
        _coconut_match_set_name_meta = _coconut_case_match_to_22[1]  #     case imdef:
        _coconut_case_match_check_22 = True  #     case imdef:
    if _coconut_case_match_check_22:  #     case imdef:
        if _coconut_match_set_name_meta is not _coconut_sentinel:  #     case imdef:
            meta = _coconut_case_match_to_22[1]  #     case imdef:
    if _coconut_case_match_check_22:  #     case imdef:
        return ([(rgb2lab, ImageDef(Numpy("float32", "HWC", "LAB", "VR_LAB"), meta), 1, "bgr_0_1 to lab"), ])  #             return[(
    if not _coconut_case_match_check_22:  #                 rgb2lab,
        _coconut_match_set_name_meta = _coconut_sentinel  #                 rgb2lab,
        if (_coconut.isinstance(_coconut_case_match_to_22, ImageDef)) and (_coconut.len(_coconut_case_match_to_22) == 2) and (_coconut.isinstance(_coconut_case_match_to_22[0], Numpy)) and (_coconut.len(_coconut_case_match_to_22[0]) == 4) and (_coconut_case_match_to_22[0][0] == "float32") and (_coconut_case_match_to_22[0][1] == "HWC") and (_coconut_case_match_to_22[0][2] == "LAB") and (_coconut_case_match_to_22[0][3] == "VR_LAB"):  #                 rgb2lab,
            _coconut_match_set_name_meta = _coconut_case_match_to_22[1]  #                 rgb2lab,
            _coconut_case_match_check_22 = True  #                 rgb2lab,
        if _coconut_case_match_check_22:  #                 rgb2lab,
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #                 rgb2lab,
                meta = _coconut_case_match_to_22[1]  #                 rgb2lab,
        if _coconut_case_match_check_22:  #                 rgb2lab,
            return ([(lab2rgb, ImageDef(Numpy("float32", "HWC", "BGR", VR_0_1), meta), 1, "lab to bgr_0_1"), ])  #             return [(




class AutoImage:  # class AutoImage:
    default_rules = [
        imdef_neighbors,
        rule_xyz_to_rgb,
        rule_batch_xyz_to_rgb,
        rule_VR_None_to_normalized,
        rule_add_channel,
        rule_swap_RGB_BGR]  #     default_rules = [
    solver = AStarSolver(rules=default_rules.copy())  #     solver = AStarSolver(rules=default_rules.copy())

    @staticmethod  #     @staticmethod
    def reset_solver():  #     def reset_solver():
        AutoImage.solver = AStarSolver(rules=AutoImage.default_rules.copy())  #         AutoImage.solver = AStarSolver(rules = AutoImage.default_rules.copy())

    @staticmethod  #     @staticmethod
    def debug_conversion(a, b, samples):  #     def debug_conversion(a,b,samples):
        x = samples  #         x = samples
        edges = AutoImage.solver.search_direct(a, b).edges  #         edges = AutoImage.solver.search_direct(a,b).edges
        for edge in edges:  #         for edge in edges:
            print(edge)  #             print(edge)
            print(edge.f)  #             print(edge.f)
            x = edge.f(x)  #             x = edge.f(x)
            print("converted to type:{_coconut_format_0}".format(_coconut_format_0=(type(x))))  #             print(f"converted to type:{type(x)}")
            if (isinstance)(x, np.ndarray):  #             if x `isinstance` np.ndarray:
                print(x.shape)  #                 print(x.shape)
            print("converted:{_coconut_format_0}".format(_coconut_format_0=(x)))  #             print(f"converted:{x}")
        return (x)  #         return x

    def to_debug(self, img_def):  #     def to_debug(self,img_def):
        img_def = parse_def(img_def)  #         img_def = parse_def(img_def)
        return (AutoImage.debug_conversion(self.img_def, img_def, self.data))  #         return AutoImage.debug_conversion(self.img_def,img_def,self.data)

    def __init__(self, data, img_def):  #     def __init__(self,data,img_def):
        img_def = parse_def(img_def)  #         img_def = parse_def(img_def)
        self.data = data  #         self.data = data
        self.img_def = img_def  #         self.img_def = img_def

    def converter(self, img_def):  #     def converter(self,img_def):
        img_def = parse_def(img_def)  #         img_def = parse_def(img_def)
        return (AutoImage.solver.search_direct(self.img_def, img_def))  #         return AutoImage.solver.search_direct(self.img_def,img_def)

    def any_converter(self, img_defs):  #     def any_converter(self,img_defs):
        imdefs = [parse_def(imdef) for imdef in img_defs]  #         imdefs = [parse_def(imdef) for imdef in img_defs]
        return (AutoImage.solver.search_direct_any(self.img_def, imdefs))  #         return AutoImage.solver.search_direct_any(self.img_def,imdefs)

    def convert(self, img_def):  #     def convert(self,img_def):
        convert = self.converter(img_def)  #         convert = self.converter(img_def)
        if convert.edges:  #         if convert.edges:
            return (AutoImage(convert(self.data), convert.edges[-1].dst))  #             return AutoImage(convert(self.data),convert.edges[-1].dst)
        else:  #         else:
            return (self)  #             return self

    def any_convert(self, imdefs):  #     def any_convert(self,imdefs):
        converter = self.any_converter(imdefs)  #         converter = self.any_converter(imdefs)
        if converter.edges:  #         if converter.edges:
            return (AutoImage(converter(self.data), converter.edges[-1].dst))  #             return AutoImage(converter(self.data),converter.edges[-1].dst)
        else:  #         else:
            return (self)  #             return self

    @_coconut_mark_as_match  #     def to(self,img_def is (str,ImageDef),log_trace=False):
    def to(*_coconut_match_args, **_coconut_match_kwargs):  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        _coconut_match_check_5 = False  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        _coconut_match_set_name_self = _coconut_sentinel  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        _coconut_match_set_name_img_def = _coconut_sentinel  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        _coconut_match_set_name_log_trace = _coconut_sentinel  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        _coconut_FunctionMatchError = _coconut_get_function_match_error()  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        if (_coconut.len(_coconut_match_args) <= 3) and (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "self" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "img_def" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 2, "log_trace" in _coconut_match_kwargs)) <= 1):  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("self")  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            _coconut_match_temp_1 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("img_def")  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            _coconut_match_temp_2 = _coconut_match_args[2] if _coconut.len(_coconut_match_args) > 2 else _coconut_match_kwargs.pop("log_trace") if "log_trace" in _coconut_match_kwargs else False  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            if (_coconut.isinstance(_coconut_match_temp_1, (str, ImageDef))) and (not _coconut_match_kwargs):  #     def to(self,img_def is (str,ImageDef),log_trace=False):
                _coconut_match_set_name_self = _coconut_match_temp_0  #     def to(self,img_def is (str,ImageDef),log_trace=False):
                _coconut_match_set_name_img_def = _coconut_match_temp_1  #     def to(self,img_def is (str,ImageDef),log_trace=False):
                _coconut_match_set_name_log_trace = _coconut_match_temp_2  #     def to(self,img_def is (str,ImageDef),log_trace=False):
                _coconut_match_check_5 = True  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        if _coconut_match_check_5:  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            if _coconut_match_set_name_self is not _coconut_sentinel:  #     def to(self,img_def is (str,ImageDef),log_trace=False):
                self = _coconut_match_temp_0  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            if _coconut_match_set_name_img_def is not _coconut_sentinel:  #     def to(self,img_def is (str,ImageDef),log_trace=False):
                img_def = _coconut_match_temp_1  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            if _coconut_match_set_name_log_trace is not _coconut_sentinel:  #     def to(self,img_def is (str,ImageDef),log_trace=False):
                log_trace = _coconut_match_temp_2  #     def to(self,img_def is (str,ImageDef),log_trace=False):
        if not _coconut_match_check_5:  #     def to(self,img_def is (str,ImageDef),log_trace=False):
            raise _coconut_FunctionMatchError('def to(self,img_def is (str,ImageDef),log_trace=False):', _coconut_match_args)  #     def to(self,img_def is (str,ImageDef),log_trace=False):

        return (self.convert(img_def).data)  #         return self.convert(img_def).data

    def any_to(self, imdefs):  #     def any_to(self,imdefs):
        return (self.any_convert(imdefs).data)  #         return self.any_convert(imdefs).data

    def to_widget(self):  #     def to_widget(self):
        _coconut_case_match_to_23 = self.img_def.data_type  #         case self.img_def.data_type:
        _coconut_case_match_check_23 = False  #         case self.img_def.data_type:
        _coconut_match_set_name_item = _coconut_sentinel  #         case self.img_def.data_type:
        if _coconut.isinstance(_coconut_case_match_to_23, PILImages):  #         case self.img_def.data_type:
            _coconut_match_set_name_item = _coconut_case_match_to_23  #         case self.img_def.data_type:
            _coconut_case_match_check_23 = True  #         case self.img_def.data_type:
        if _coconut_case_match_check_23:  #         case self.img_def.data_type:
            if _coconut_match_set_name_item is not _coconut_sentinel:  #         case self.img_def.data_type:
                item = _coconut_case_match_to_23  #         case self.img_def.data_type:
        if _coconut_case_match_check_23:  #         case self.img_def.data_type:
            return (self.tile_image().to_widget())  #                 return self.tile_image().to_widget()
        if not _coconut_case_match_check_23:  #             match TensorLike(_,arng,*_) if "B" in arng:
            _coconut_match_set_name_arng = _coconut_sentinel  #             match TensorLike(_,arng,*_) if "B" in arng:
            if (_coconut.isinstance(_coconut_case_match_to_23, TensorLike)) and (_coconut.len(_coconut_case_match_to_23) >= 2):  #             match TensorLike(_,arng,*_) if "B" in arng:
                _coconut_match_set_name_arng = _coconut_case_match_to_23[1]  #             match TensorLike(_,arng,*_) if "B" in arng:
                _coconut_case_match_check_23 = True  #             match TensorLike(_,arng,*_) if "B" in arng:
            if _coconut_case_match_check_23:  #             match TensorLike(_,arng,*_) if "B" in arng:
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match TensorLike(_,arng,*_) if "B" in arng:
                    arng = _coconut_case_match_to_23[1]  #             match TensorLike(_,arng,*_) if "B" in arng:
            if _coconut_case_match_check_23 and not ("B" in arng):  #             match TensorLike(_,arng,*_) if "B" in arng:
                _coconut_case_match_check_23 = False  #             match TensorLike(_,arng,*_) if "B" in arng:
            if _coconut_case_match_check_23:  #             match TensorLike(_,arng,*_) if "B" in arng:
                return (self.tile_image().to_widget())  #                 return self.tile_image().to_widget()
        if not _coconut_case_match_check_23:  #         else:
            convert = self.converter(self.to_images_def())  #             convert = self.converter(self.to_images_def())
        return ((infer_widget)(convert(self.data)))  #         return convert(self.data) |> infer_widget

    def _repr_html_(self):  #     def _repr_html_(self):
        return ((display)(self.to_widget()))  #         return self.to_widget() |> display

    def to_images_def(self):  #     def to_images_def(self):
        """
        you have to add en_batched tag when data is not batch.
        """  #         """
        tag_opt = frozenset()  #         tag_opt=frozenset()
        img_cls = PILImages  #         img_cls = PILImages
        _coconut_case_match_to_24 = self.img_def  #         case self.img_def:
        _coconut_case_match_check_24 = False  #         case self.img_def:
        _coconut_match_set_name_arng = _coconut_sentinel  #         case self.img_def:
        _coconut_match_set_name_meta = _coconut_sentinel  #         case self.img_def:
        if (_coconut.isinstance(_coconut_case_match_to_24, ImageDef)) and (_coconut.len(_coconut_case_match_to_24) == 2) and (_coconut.isinstance(_coconut_case_match_to_24[0], TensorLike)) and (_coconut.len(_coconut_case_match_to_24[0]) == 4):  #         case self.img_def:
            _coconut_match_set_name_arng = _coconut_case_match_to_24[0][1]  #         case self.img_def:
            _coconut_match_set_name_meta = _coconut_case_match_to_24[1]  #         case self.img_def:
            _coconut_case_match_check_24 = True  #         case self.img_def:
        if _coconut_case_match_check_24:  #         case self.img_def:
            if _coconut_match_set_name_arng is not _coconut_sentinel:  #         case self.img_def:
                arng = _coconut_case_match_to_24[0][1]  #         case self.img_def:
            if _coconut_match_set_name_meta is not _coconut_sentinel:  #         case self.img_def:
                meta = _coconut_case_match_to_24[1]  #         case self.img_def:
        if _coconut_case_match_check_24 and not ("B" not in arng):  #         case self.img_def:
            _coconut_case_match_check_24 = False  #         case self.img_def:
        if _coconut_case_match_check_24:  #         case self.img_def:
            tag_opt = frozenset(("en_batched", ))  #                 tag_opt = frozenset(("en_batched",))
        if not _coconut_case_match_check_24:  #             match ImageDef(PILImage(mode,ch),meta):
            _coconut_match_set_name_mode = _coconut_sentinel  #             match ImageDef(PILImage(mode,ch),meta):
            _coconut_match_set_name_ch = _coconut_sentinel  #             match ImageDef(PILImage(mode,ch),meta):
            _coconut_match_set_name_meta = _coconut_sentinel  #             match ImageDef(PILImage(mode,ch),meta):
            if (_coconut.isinstance(_coconut_case_match_to_24, ImageDef)) and (_coconut.len(_coconut_case_match_to_24) == 2) and (_coconut.isinstance(_coconut_case_match_to_24[0], PILImage)) and (_coconut.len(_coconut_case_match_to_24[0]) == 2):  #             match ImageDef(PILImage(mode,ch),meta):
                _coconut_match_set_name_mode = _coconut_case_match_to_24[0][0]  #             match ImageDef(PILImage(mode,ch),meta):
                _coconut_match_set_name_ch = _coconut_case_match_to_24[0][1]  #             match ImageDef(PILImage(mode,ch),meta):
                _coconut_match_set_name_meta = _coconut_case_match_to_24[1]  #             match ImageDef(PILImage(mode,ch),meta):
                _coconut_case_match_check_24 = True  #             match ImageDef(PILImage(mode,ch),meta):
            if _coconut_case_match_check_24:  #             match ImageDef(PILImage(mode,ch),meta):
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match ImageDef(PILImage(mode,ch),meta):
                    mode = _coconut_case_match_to_24[0][0]  #             match ImageDef(PILImage(mode,ch),meta):
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match ImageDef(PILImage(mode,ch),meta):
                    ch = _coconut_case_match_to_24[0][1]  #             match ImageDef(PILImage(mode,ch),meta):
                if _coconut_match_set_name_meta is not _coconut_sentinel:  #             match ImageDef(PILImage(mode,ch),meta):
                    meta = _coconut_case_match_to_24[1]  #             match ImageDef(PILImage(mode,ch),meta):
            if _coconut_case_match_check_24:  #             match ImageDef(PILImage(mode,ch),meta):
                tag_opt = frozenset(("en_batched", ))  #                 tag_opt = frozenset(("en_batched",))

        _coconut_case_match_to_25 = self.img_def.data_type  #         case self.img_def.data_type:
        _coconut_case_match_check_25 = False  #         case self.img_def.data_type:
        _coconut_match_set_name_mode = _coconut_sentinel  #         case self.img_def.data_type:
        _coconut_match_set_name_chrepr = _coconut_sentinel  #         case self.img_def.data_type:
        if (_coconut.isinstance(_coconut_case_match_to_25, PILImage)) and (_coconut.len(_coconut_case_match_to_25) == 2):  #         case self.img_def.data_type:
            _coconut_match_set_name_mode = _coconut_case_match_to_25[0]  #         case self.img_def.data_type:
            _coconut_match_set_name_chrepr = _coconut_case_match_to_25[1]  #         case self.img_def.data_type:
            _coconut_case_match_check_25 = True  #         case self.img_def.data_type:
        if _coconut_case_match_check_25:  #         case self.img_def.data_type:
            if _coconut_match_set_name_mode is not _coconut_sentinel:  #         case self.img_def.data_type:
                mode = _coconut_case_match_to_25[0]  #         case self.img_def.data_type:
            if _coconut_match_set_name_chrepr is not _coconut_sentinel:  #         case self.img_def.data_type:
                chrepr = _coconut_case_match_to_25[1]  #         case self.img_def.data_type:
        if _coconut_case_match_check_25:  #         case self.img_def.data_type:
            return (ImageDef(PILImages(mode, chrepr), self.img_def.meta | tag_opt))  #                 return ImageDef(PILImages(mode,chrepr),self.img_def.meta | tag_opt)
        if not _coconut_case_match_check_25:  #             match PILImages(mode,chrepr):
            _coconut_match_set_name_mode = _coconut_sentinel  #             match PILImages(mode,chrepr):
            _coconut_match_set_name_chrepr = _coconut_sentinel  #             match PILImages(mode,chrepr):
            if (_coconut.isinstance(_coconut_case_match_to_25, PILImages)) and (_coconut.len(_coconut_case_match_to_25) == 2):  #             match PILImages(mode,chrepr):
                _coconut_match_set_name_mode = _coconut_case_match_to_25[0]  #             match PILImages(mode,chrepr):
                _coconut_match_set_name_chrepr = _coconut_case_match_to_25[1]  #             match PILImages(mode,chrepr):
                _coconut_case_match_check_25 = True  #             match PILImages(mode,chrepr):
            if _coconut_case_match_check_25:  #             match PILImages(mode,chrepr):
                if _coconut_match_set_name_mode is not _coconut_sentinel:  #             match PILImages(mode,chrepr):
                    mode = _coconut_case_match_to_25[0]  #             match PILImages(mode,chrepr):
                if _coconut_match_set_name_chrepr is not _coconut_sentinel:  #             match PILImages(mode,chrepr):
                    chrepr = _coconut_case_match_to_25[1]  #             match PILImages(mode,chrepr):
            if _coconut_case_match_check_25:  #             match PILImages(mode,chrepr):
                return (self.img_def)  #                 return self.img_def
        if not _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            _coconut_match_set_name_dtype = _coconut_sentinel  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            _coconut_match_set_name_arng = _coconut_sentinel  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            _coconut_match_set_name_c = _coconut_sentinel  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            _coconut_match_set_name_vr = _coconut_sentinel  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            if (_coconut.isinstance(_coconut_case_match_to_25, TensorLike)) and (_coconut.len(_coconut_case_match_to_25) == 4):  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                _coconut_match_set_name_dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                _coconut_match_set_name_arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                _coconut_match_set_name_c = _coconut_case_match_to_25[2]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                _coconut_match_set_name_vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                _coconut_case_match_check_25 = True  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                if _coconut_match_set_name_dtype is not _coconut_sentinel:  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                    dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                    arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                if _coconut_match_set_name_c is not _coconut_sentinel:  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                    c = _coconut_case_match_to_25[2]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                if _coconut_match_set_name_vr is not _coconut_sentinel:  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                    vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            if _coconut_case_match_check_25 and not (len(c) == 1):  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                _coconut_case_match_check_25 = False  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,c,vr) if len(c) == 1:
                return (ImageDef(img_cls("L", c), self.img_def.meta | tag_opt))  #                 return ImageDef(img_cls("L",c),self.img_def.meta | tag_opt)
        if not _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,"RGBA",vr):
            _coconut_match_set_name_dtype = _coconut_sentinel  #             match TensorLike(dtype,arng,"RGBA",vr):
            _coconut_match_set_name_arng = _coconut_sentinel  #             match TensorLike(dtype,arng,"RGBA",vr):
            _coconut_match_set_name_vr = _coconut_sentinel  #             match TensorLike(dtype,arng,"RGBA",vr):
            if (_coconut.isinstance(_coconut_case_match_to_25, TensorLike)) and (_coconut.len(_coconut_case_match_to_25) == 4) and (_coconut_case_match_to_25[2] == "RGBA"):  #             match TensorLike(dtype,arng,"RGBA",vr):
                _coconut_match_set_name_dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,"RGBA",vr):
                _coconut_match_set_name_arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,"RGBA",vr):
                _coconut_match_set_name_vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,"RGBA",vr):
                _coconut_case_match_check_25 = True  #             match TensorLike(dtype,arng,"RGBA",vr):
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,"RGBA",vr):
                if _coconut_match_set_name_dtype is not _coconut_sentinel:  #             match TensorLike(dtype,arng,"RGBA",vr):
                    dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,"RGBA",vr):
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match TensorLike(dtype,arng,"RGBA",vr):
                    arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,"RGBA",vr):
                if _coconut_match_set_name_vr is not _coconut_sentinel:  #             match TensorLike(dtype,arng,"RGBA",vr):
                    vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,"RGBA",vr):
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,"RGBA",vr):
                return (ImageDef(img_cls("RGBA", "RGBA"), self.img_def.meta | tag_opt))  #                 return ImageDef(img_cls("RGBA","RGBA"),self.img_def.meta | tag_opt)
        if not _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,"RGB",vr):
            _coconut_match_set_name_dtype = _coconut_sentinel  #             match TensorLike(dtype,arng,"RGB",vr):
            _coconut_match_set_name_arng = _coconut_sentinel  #             match TensorLike(dtype,arng,"RGB",vr):
            _coconut_match_set_name_vr = _coconut_sentinel  #             match TensorLike(dtype,arng,"RGB",vr):
            if (_coconut.isinstance(_coconut_case_match_to_25, TensorLike)) and (_coconut.len(_coconut_case_match_to_25) == 4) and (_coconut_case_match_to_25[2] == "RGB"):  #             match TensorLike(dtype,arng,"RGB",vr):
                _coconut_match_set_name_dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,"RGB",vr):
                _coconut_match_set_name_arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,"RGB",vr):
                _coconut_match_set_name_vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,"RGB",vr):
                _coconut_case_match_check_25 = True  #             match TensorLike(dtype,arng,"RGB",vr):
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,"RGB",vr):
                if _coconut_match_set_name_dtype is not _coconut_sentinel:  #             match TensorLike(dtype,arng,"RGB",vr):
                    dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,"RGB",vr):
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match TensorLike(dtype,arng,"RGB",vr):
                    arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,"RGB",vr):
                if _coconut_match_set_name_vr is not _coconut_sentinel:  #             match TensorLike(dtype,arng,"RGB",vr):
                    vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,"RGB",vr):
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,"RGB",vr):
                return (ImageDef(img_cls("RGB", "RGB"), self.img_def.meta | tag_opt))  #                 return ImageDef(img_cls("RGB","RGB"),self.img_def.meta | tag_opt)
        if not _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            _coconut_match_set_name_dtype = _coconut_sentinel  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            _coconut_match_set_name_arng = _coconut_sentinel  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            _coconut_match_set_name_ch = _coconut_sentinel  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            _coconut_match_set_name_vr = _coconut_sentinel  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            if (_coconut.isinstance(_coconut_case_match_to_25, TensorLike)) and (_coconut.len(_coconut_case_match_to_25) == 4):  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                _coconut_match_set_name_dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                _coconut_match_set_name_arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                _coconut_match_set_name_ch = _coconut_case_match_to_25[2]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                _coconut_match_set_name_vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                _coconut_case_match_check_25 = True  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                if _coconut_match_set_name_dtype is not _coconut_sentinel:  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                    dtype = _coconut_case_match_to_25[0]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                if _coconut_match_set_name_arng is not _coconut_sentinel:  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                    arng = _coconut_case_match_to_25[1]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                if _coconut_match_set_name_ch is not _coconut_sentinel:  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                    ch = _coconut_case_match_to_25[2]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                if _coconut_match_set_name_vr is not _coconut_sentinel:  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                    vr = _coconut_case_match_to_25[3]  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            if _coconut_case_match_check_25 and not ("A" in ch and ch != "LAB"):  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                _coconut_case_match_check_25 = False  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
            if _coconut_case_match_check_25:  #             match TensorLike(dtype,arng,ch,vr) if "A" in ch and ch != "LAB":
                return (ImageDef(img_cls("RGBA", "RGBA"), self.img_def.meta | tag_opt))  #                 return ImageDef(img_cls("RGBA","RGBA"),self.img_def.meta | tag_opt)
        if not _coconut_case_match_check_25:  #         else:
            attempts = ["image,RGBA,RGBA", "images,RGBA,RGBA", "image,RGB,RGB", "images,RGB,RGB", "image,L,L", "images,L,L"]  #             attempts=[
            for tgt in attempts:  #             for tgt in attempts:
                imdef = str_to_img_def(tgt)  #                 imdef = str_to_img_def(tgt)
                try:  #                 try:
                    converter = self.converter(imdef)  #                     converter = self.converter(imdef)
                    return (imdef)  #                     return imdef
                except NoRouteException as e:  #                 except NoRouteException as e:
#logger.warning(f"no route for:{imdef}. trying next imdef.")
                    pass  #                     pass
            raise RuntimeError("cannot convert to image:{_coconut_format_0} to any image_like imdef}".format(_coconut_format_0=(self.img_def)))  #             raise RuntimeError(f"cannot convert to image:{self.img_def} to any image_like imdef}")


    def image_op(self, f: '_coconut.typing.Callable[[Image], Image]'):  #     def image_op(self,f:Image->Image):
        images_def = self.to_images_def()  #         images_def = self.to_images_def()
        images = self.to(images_def)  #         images = self.to(images_def)
        new_images = [f(i) for i in images]  # do some resizing or something  #         new_images=[f(i) for i in images ] # do some resizing or something
        new_ai = AutoImage(new_images, images_def)  #         new_ai = AutoImage(new_images,images_def)
        return (new_ai.convert(self.img_def))  # go back to last state.  #         return new_ai.convert(self.img_def) # go back to last state.

    def visdom(self, visdom=None, **kwargs):  #     def visdom(self,visdom=None,**kwargs):
        if visdom is None:  #         if visdom is None:
            from data_tree.visdom import VISDOM  #             from data_tree.visdom import VISDOM
            visdom = VISDOM  #             visdom = VISDOM
        candidates = ["numpy,float32,CHW,RGB,0_1", "numpy,float32,CHW,L,0_1", "numpy,float32,BCHW,RGB,0_1", "numpy,float32,BCHW,L,0_1"]  #         candidates = [
        img = self.any_convert(candidates)  #         img = self.any_convert(candidates)
        data_type = img.img_def.data_type  #         data_type = img.img_def.data_type
        _coconut_case_match_to_26 = data_type  #         case data_type:
        _coconut_case_match_check_26 = False  #         case data_type:
        if (_coconut.isinstance(_coconut_case_match_to_26, Numpy)) and (_coconut.len(_coconut_case_match_to_26) == 4) and (_coconut_case_match_to_26[1] == "CHW"):  #         case data_type:
            _coconut_case_match_check_26 = True  #         case data_type:
        if _coconut_case_match_check_26:  #         case data_type:
            res = visdom.image(img.data, **kwargs)  #                res = visdom.image(img.data,**kwargs)
        if not _coconut_case_match_check_26:  #            match Numpy(_,"BCHW",_,_):
            if (_coconut.isinstance(_coconut_case_match_to_26, Numpy)) and (_coconut.len(_coconut_case_match_to_26) == 4) and (_coconut_case_match_to_26[1] == "BCHW"):  #            match Numpy(_,"BCHW",_,_):
                _coconut_case_match_check_26 = True  #            match Numpy(_,"BCHW",_,_):
            if _coconut_case_match_check_26:  #            match Numpy(_,"BCHW",_,_):
                res = visdom.images(img.data, **kwargs)  #                res = visdom.images(img.data,**kwargs)
        return (res)  #         return res


_coconut_call_set_names(AutoImage)  # @property
try:  # @property
    _coconut_name_store_0 = images  # @property
except _coconut.NameError:  # @property
    _coconut_name_store_0 = _coconut_sentinel  # @property
@property  # @property
def images(self):  # def AutoImage.images(self):
    return (self.to(self.to_images_def()))  #     return self.to(self.to_images_def())

AutoImage.images = images  # @property
if _coconut_name_store_0 is not _coconut_sentinel:  # @property
    images = _coconut_name_store_0  # @property
try:  # @property
    _coconut_name_store_1 = image_size  # @property
except _coconut.NameError:  # @property
    _coconut_name_store_1 = _coconut_sentinel  # @property
@property  # @property
def image_size(self):  # def AutoImage.image_size(self):
    return (self.images[0].size)  #     return self.images[0].size

AutoImage.image_size = image_size  # def make_grid(imgs, nrow, padding=0):
if _coconut_name_store_1 is not _coconut_sentinel:  # def make_grid(imgs, nrow, padding=0):
    image_size = _coconut_name_store_1  # def make_grid(imgs, nrow, padding=0):
def make_grid(imgs, nrow, padding=0):  # def make_grid(imgs, nrow, padding=0):
    """Numpy配列の複数枚の画像を、1枚の画像にタイルします

    Arguments:
        imgs {np.ndarray} -- 複数枚の画像からなるテンソル
        nrow {int} -- 1行あたりにタイルする枚数

    Keyword Arguments:
        padding {int} -- グリッドの間隔 (default: {0})

    Returns:
        [np.ndarray] -- 3階テンソル。1枚の画像
    """  #     """
    assert imgs.ndim == 4 and nrow > 0  #     assert imgs.ndim == 4 and nrow > 0
    batch, height, width, ch = imgs.shape  #     batch, height, width, ch = imgs.shape
    n = nrow * (batch // nrow + np.sign(batch % nrow))  #     n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow  #     ncol = n // nrow
    pad = np.zeros((n - batch, height, width, ch), imgs.dtype)  #     pad = np.zeros((n - batch, height, width, ch), imgs.dtype)
    x = np.concatenate([imgs, pad], axis=0)  #     x = np.concatenate([imgs, pad], axis=0)
# border padding if required
    if padding > 0:  #     if padding > 0:
        x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)), "constant", constant_values=(0, 0))  # 下と右だけにpaddingを入れる  #         x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)),
        height += padding  #         height += padding
        width += padding  #         width += padding
    x = x.reshape(ncol, nrow, height, width, ch)  #     x = x.reshape(ncol, nrow, height, width, ch)
    x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, ch)  #     x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, ch)
    x = x.reshape(height * ncol, width * nrow, ch)  #     x = x.reshape(height * ncol, width * nrow, ch)
    if padding > 0:  #     if padding > 0:
        x = x[:(height * ncol - padding), :(width * nrow - padding), :]  # 右端と下端のpaddingを削除  #         x = x[:(height * ncol - padding),:(width * nrow - padding),:] # 右端と下端のpaddingを削除
    return (x)  #     return x


try:  # def AutoImage.tile_image(self,w=1024,h=1024,max_image=100,padding=1):
    _coconut_name_store_2 = tile_image  # def AutoImage.tile_image(self,w=1024,h=1024,max_image=100,padding=1):
except _coconut.NameError:  # def AutoImage.tile_image(self,w=1024,h=1024,max_image=100,padding=1):
    _coconut_name_store_2 = _coconut_sentinel  # def AutoImage.tile_image(self,w=1024,h=1024,max_image=100,padding=1):
def tile_image(self, w=1024, h=1024, max_image=100, padding=1):  # def AutoImage.tile_image(self,w=1024,h=1024,max_image=100,padding=1):
    ch = self.to_images_def().data_type.channel_repr  #     ch = self.to_images_def().data_type.channel_repr
    if len(ch) == 1:  #     if len(ch) == 1:
        codec = "numpy,uint8,BHW,{_coconut_format_0},0_255".format(_coconut_format_0=(ch))  #         codec = f"numpy,uint8,BHW,{ch},0_255"
    else:  #     else:
        codec = "numpy,uint8,BHWC,{_coconut_format_0},0_255".format(_coconut_format_0=(ch))  #         codec = f"numpy,uint8,BHWC,{ch},0_255"
    imgs = self.to(codec)[:max_image]  #     imgs = self.to(codec)[:max_image]
    nrow = int(sqrt(len(imgs)) + 0.5)  #     nrow = int(sqrt(len(imgs))+0.5)
    r = int((w - ((nrow + 1) * padding)) / nrow)  #     r = int((w-((nrow+1)*padding))/nrow)
    imgs = np.array([((np.array)(Image.fromarray(img).resize((r, r)))) for img in imgs])  #     imgs = np.array([(Image.fromarray(img).resize((r,r)) |> np.array) for img in imgs])
    if len(ch) == 1:  #     if len(ch) == 1:
        imgs = imgs[:, :, :, None]  #         imgs = imgs[:,:,:,None]
    return (AutoImage(make_grid(imgs, nrow, padding=1), "numpy,uint8,HWC,{_coconut_format_0},0_255".format(_coconut_format_0=(ch))))  #     return AutoImage(make_grid(imgs,nrow,padding=1),f"numpy,uint8,HWC,{ch},0_255")

AutoImage.tile_image = tile_image  # img_to_shifting_grids = img->make_grids(*img.image_size)|> shifting_grids
if _coconut_name_store_2 is not _coconut_sentinel:  # img_to_shifting_grids = img->make_grids(*img.image_size)|> shifting_grids
    tile_image = _coconut_name_store_2  # img_to_shifting_grids = img->make_grids(*img.image_size)|> shifting_grids
img_to_shifting_grids = lambda img: (shifting_grids)(make_grids(*img.image_size))  # img_to_shifting_grids = img->make_grids(*img.image_size)|> shifting_grids
def auto_to_3res(img, cx, cy, r=256) -> '"AutoImage"':  # def auto_to_3res(img,cx,cy,r=256)->"AutoImage":
    img = img.to("image,L,L")  #     img = img.to("image,L,L")
#img = img.resize((2048,2048))
    chs = [crop_square(img, cx, cy, _r).resize((r, r)) for _r in [r * 4, r * 2, r]]  #     chs = [crop_square(img,cx,cy,_r).resize((r,r)) for _r in [r*4,r*2,r]]
    return (AutoImage(np.concatenate([np.array(i)[:, :, None] for i in chs], axis=2), "numpy,float32,HWC,RGB,0_255"))  #     return AutoImage(np.concatenate([np.array(i)[:,:,None] for i in chs],axis=2),"numpy,float32,HWC,RGB,0_255")

def img_to_grid_batch(img: 'AutoImage'):  # def img_to_grid_batch(img:AutoImage):
    grids = (series)((img_to_shifting_grids(img)).astype("int32"))  #     grids = img_to_shifting_grids(img) |> .astype("int32") |> series
    batch = ((array)(grids.map(lambda xy: auto_to_3res(img, xy[0] + 128, xy[1] + 128, r=256).to("numpy,float32,HWC,RGB,0_1")).values)).astype("float32")  #     batch = grids.map((xy)->auto_to_3res(img,xy[0]+128,xy[1]+128,r=256).to("numpy,float32,HWC,RGB,0_1")).values |> array |> .astype("float32")
    return (grids.values, AutoImage(batch, "numpy,float32,BHWC,RGB,0_1"))  #     return grids.values,AutoImage(batch,"numpy,float32,BHWC,RGB,0_1")


try:  # def AutoImage.cast(self,imgdef):
    _coconut_name_store_3 = cast  # def AutoImage.cast(self,imgdef):
except _coconut.NameError:  # def AutoImage.cast(self,imgdef):
    _coconut_name_store_3 = _coconut_sentinel  # def AutoImage.cast(self,imgdef):
def cast(self, imgdef):  # def AutoImage.cast(self,imgdef):
    return (AutoImage(self.data, imgdef))  #     return AutoImage(self.data,imgdef)



AutoImage.cast = cast  # 
if _coconut_name_store_3 is not _coconut_sentinel:  # 
    cast = _coconut_name_store_3  #
