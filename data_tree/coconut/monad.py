#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xe2735dfb

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


class Try(_coconut.collections.namedtuple("Try", "")):  # data Try
    __slots__ = ()  # data Try
    __ne__ = _coconut.object.__ne__  # data Try
    def __eq__(self, other):  # data Try
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Try
    def __hash__(self):  # data Try
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Try

class Success(_coconut.collections.namedtuple("Success", "result"), Try):  # data Success(result) from Try
    __slots__ = ()  # data Success(result) from Try
    __ne__ = _coconut.object.__ne__  # data Success(result) from Try
    def __eq__(self, other):  # data Success(result) from Try
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Success(result) from Try
    def __hash__(self):  # data Success(result) from Try
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Success(result) from Try

class Failure(_coconut.collections.namedtuple("Failure", "exception trace"), Try):  # data Failure(exception,trace) from Try
    __slots__ = ()  # data Failure(exception,trace) from Try
    __ne__ = _coconut.object.__ne__  # data Failure(exception,trace) from Try
    def __eq__(self, other):  # data Failure(exception,trace) from Try
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Failure(exception,trace) from Try
    def __hash__(self):  # data Failure(exception,trace) from Try
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Failure(exception,trace) from Try


def try_monad(f):  # def try_monad(f):
    def _inner(*args, **kwargs):  #     def _inner(*args,**kwargs):
        try:  #         try:
            res = f(*args, **kwargs)  #             res = f(*args,**kwargs)
            return Success(res)  #             return Success(res)
        except Exception as e:  #         except Exception as e:
            import traceback  #             import traceback
            return Failure(e, traceback.format_exc())  #             return Failure(e,traceback.format_exc())
    return _inner  #     return _inner
