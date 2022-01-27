#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xae77580

# Compiled with Coconut version 1.6.0 [Vocational Guidance Counsellor]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop
import sys as _coconut_sys, os as _coconut_os
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
from __coconut__ import _coconut_call_set_names, _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------


class Try(_coconut.collections.namedtuple("Try", ())):  # data Try
    __slots__ = ()  # data Try
    __ne__ = _coconut.object.__ne__  # data Try
    def __eq__(self, other):  # data Try
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Try
    def __hash__(self):  # data Try
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Try
    __match_args__ = ()  # data Try
_coconut_call_set_names(Try)  # data Success(result) from Try
class Success(_coconut.collections.namedtuple("Success", ('result',)), Try):  # data Success(result) from Try
    __slots__ = ()  # data Success(result) from Try
    __ne__ = _coconut.object.__ne__  # data Success(result) from Try
    def __eq__(self, other):  # data Success(result) from Try
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Success(result) from Try
    def __hash__(self):  # data Success(result) from Try
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Success(result) from Try
    __match_args__ = ('result',)  # data Success(result) from Try
_coconut_call_set_names(Success)  # data Failure(exception,trace) from Try
class Failure(_coconut.collections.namedtuple("Failure", ('exception', 'trace')), Try):  # data Failure(exception,trace) from Try
    __slots__ = ()  # data Failure(exception,trace) from Try
    __ne__ = _coconut.object.__ne__  # data Failure(exception,trace) from Try
    def __eq__(self, other):  # data Failure(exception,trace) from Try
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Failure(exception,trace) from Try
    def __hash__(self):  # data Failure(exception,trace) from Try
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Failure(exception,trace) from Try
    __match_args__ = ('exception', 'trace')  # data Failure(exception,trace) from Try

_coconut_call_set_names(Failure)  # def try_monad(f):
def try_monad(f):  # def try_monad(f):
    def _inner(*args, **kwargs):  #     def _inner(*args,**kwargs):
        try:  #         try:
            res = f(*args, **kwargs)  #             res = f(*args,**kwargs)
            return (Success(res))  #             return Success(res)
        except Exception as e:  #         except Exception as e:
            import traceback  #             import traceback
            return (Failure(e, traceback.format_exc()))  #             return Failure(e,traceback.format_exc())
    return (_inner)  #     return _inner
