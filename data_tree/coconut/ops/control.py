#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xdabfa432

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

from data_tree._series import Series  # from data_tree._series import Series,SourcedSeries
from data_tree._series import SourcedSeries  # from data_tree._series import Series,SourcedSeries
from data_tree.coconut.controllable import ControllableWidget  # from data_tree.coconut.controllable import ControllableWidget,iwidget
from data_tree.coconut.controllable import iwidget  # from data_tree.coconut.controllable import ControllableWidget,iwidget
from data_tree.indexer import IdentityIndexer  # from data_tree.indexer import IdentityIndexer

class SliceFunction(_coconut.collections.namedtuple("SliceFunction", "f vec_f")):  # data SliceFunction(f,vec_f):
    __slots__ = ()  # data SliceFunction(f,vec_f):
    __ne__ = _coconut.object.__ne__  # data SliceFunction(f,vec_f):
    def __eq__(self, other):  # data SliceFunction(f,vec_f):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data SliceFunction(f,vec_f):
    def __hash__(self):  # data SliceFunction(f,vec_f):
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data SliceFunction(f,vec_f):
    def partial(self, **kwargs):  #     def partial(self,**kwargs):
        return SliceFunction(_coconut.functools.partial(self.f, **kwargs), _coconut.functools.partial(self.vec_f, **kwargs))  #         return SliceFunction(self.f$(**kwargs),self.vec_f$(**kwargs))


def f_to_vec_f(f):  # def f_to_vec_f(f):
    return lambda items, *args, **kwargs: [f(i, *args, **kwargs) for i in items]  #     return (items,*args,**kwargs)->[f(i,*args,**kwargs) for i in items]

def vecf_to_f(vecf):  # def vecf_to_f(vecf):
    return lambda items, *args, **kwargs: vecf(item, *args, **kwargs)[0]  #     return (items,*args,**kwargs)->vecf(item,*args,**kwargs)[0]

def get_slice_function(f=None, vec_f=None):  # def get_slice_function(f=None,vec_f=None):
    if f is None and vec_f is None:  #     if f is None and vec_f is None:
        raise RuntimeError("either f or vec_f must be a function! {_coconut_format_0} and {_coconut_format_1}".format(_coconut_format_0=(f), _coconut_format_1=(vec_f)))  #         raise RuntimeError(f"either f or vec_f must be a function! {f} and {vec_f}")
    if f is None:  #     if f is None:
        f = vecf_to_f(vec_f)  #         f = vecf_to_f(vec_f)
    elif vec_f is None:  #     elif vec_f is None:
        vec_f = f_to_vec_f(f)  #         vec_f = f_to_vec_f(f)
    return SliceFunction(f, vec_f)  #     return SliceFunction(f,vec_f)


class CMapTemplate(SourcedSeries):  # class CMapTemplate(SourcedSeries):
    def __init__(self, src, mapper, slice_mapper, param_controllers: 'Mapping[str, tuple]'):  #     def __init__(self,
        super().__init__()  #         super().__init__()
        self._src = src  #         self._src = src
        self._indexer = IdentityIndexer(self.total)  #         self._indexer = IdentityIndexer(self.total)
        self.param_controllers = param_controllers  #         self.param_controllers = param_controllers
        self.default_params = {k: v[0] for k, v in param_controllers.items()}  #         self.default_params = {k:v[0] for k,v in param_controllers.items()}
        self.unapplied = get_slice_function(mapper, slice_mapper)  #         self.unapplied = get_slice_function(mapper,slice_mapper)
        self.applied = self.unapplied.partial(**self.default_params)  #         self.applied = self.unapplied.partial(**self.default_params)
        self.mapped = src.map(f=self.applied.f, batch_f=self.applied.vec_f)  #         self.mapped = src.map(f=self.applied.f,batch_f=self.applied.vec_f)

    def clone(self, parents):  #     def clone(self, parents):
        return CMapTemplate(self.src, self.unapplied.f, self.unapplied.vec_f, self.param_controllers)  #         return CMapTemplate(

    @property  #     @property
    def indexer(self):  #     def indexer(self):
        return self._indexer  #         return self._indexer

    @property  #     @property
    def src(self) -> 'Series':  #     def src(self) -> Series:
        return self._src  #         return self._src

    @property  #     @property
    def total(self):  #     def total(self):
        return self._src.total  #         return self._src.total

    def _get_item(self, index):  #     def _get_item(self, index):
        return self.mapped._values(index)  #         return self.mapped._values(index)

    def _get_slice(self, _slice):  #     def _get_slice(self, _slice):
        return self.mapped._values(_slice)  #         return self.mapped._values(_slice)

    def _get_indices(self, indices):  #     def _get_indices(self, indices):
        return self.mapped._values(indices)  #         return self.mapped._values(indices)

    def _get_mask(self, mask):  #     def _get_mask(self, mask):
        return self.mapped._values(mask)  #         return self.mapped._values(mask)

    def slice_generator(self, slices, preload=5, en_numpy=False):  #     def slice_generator(self, slices, preload=5, en_numpy=False):
        yield from self.mapped.slice_generator(slices, preload=preload, en_numpy=en_numpy)  #         yield from self.mapped.slice_generator(slices,preload=preload,en_numpy=en_numpy)
from typing import Callable  # from typing import Callable
class CMapInstance(SourcedSeries):  # stateful. beware not to use it  # class CMapInstance(SourcedSeries):# stateful. beware not to use it
    def __init__(self, template: 'CMapTemplate'):  #     def __init__(self,template:CMapTemplate):
        self.template = template  #         self.template = template
        self.controllers = dict()  #         self.controllers = dict()
        for k, v in self.template.param_controllers.items():  #         for k,v in self.template.param_controllers.items():
            _coconut_match_to = v[1]  #             case v[1]:
            _coconut_case_check_0 = False  #             case v[1]:
            if _coconut.isinstance(_coconut_match_to, Callable):  #             case v[1]:
                f = _coconut_match_to  #             case v[1]:
                _coconut_case_check_0 = True  #             case v[1]:
            if _coconut_case_check_0:  #             case v[1]:
                self.controllers[k] = f()  #                     self.controllers[k] = f()
            if not _coconut_case_check_0:  #                 match data is dict:
                if _coconut.isinstance(_coconut_match_to, dict):  #                 match data is dict:
                    data = _coconut_match_to  #                 match data is dict:
                    _coconut_case_check_0 = True  #                 match data is dict:
                if _coconut_case_check_0:  #                 match data is dict:
                    data = data.copy()  #                     data = data.copy()
                    schem = data["schem"]  #                     schem = data["schem"]
                    del data["schem"]  #                     del data["schem"]
                    self.controllers[k] = iwidget(schem, **data)  #                     self.controllers[k] = iwidget(schem,**data)

#self.controllers = {k:v[1] for k,v in self.template.param_controllers.items()}

    @property  #     @property
    def indexer(self):  #     def indexer(self):
        return self.template.indexer  #         return self.template.indexer

    @property  #     @property
    def src(self) -> 'Series':  #     def src(self)->Series:
        return self.template.src  #         return self.template.src

    def clone(self, parents):  #     def clone(self,parents):
        return CMapInstance(self.template)  #         return CMapInstance(self.template)

    @property  #     @property
    def total(self):  #     def total(self):
        return self.template.total  #         return self.template.total

    def current_kwargs(self):  #     def current_kwargs(self):
        return {k: v.value.value for k, v in self.controllers.items()}  #         return {k:v.value.value for k,v in self.controllers.items()}

    def _get_item(self, index):  #     def _get_item(self, index):
        kwargs = self.current_kwargs()  #         kwargs = self.current_kwargs()
        val = self.src[index]  #         val = self.src[index]
        return self.template.unapplied.f(val, **kwargs)  #         return self.template.unapplied.f(val,**kwargs)

    def _get_slice(self, _slice):  #     def _get_slice(self, _slice):
        kwargs = self.current_kwargs()  #         kwargs = self.current_kwargs()
        vals = self.src._values(_slice)  #         vals = self.src._values(_slice)
        return self.template.unaplied.vec_f(vals, **kwargs)  #         return self.template.unaplied.vec_f(vals,**kwargs)

    def _get_indices(self, indices):  #     def _get_indices(self, indices):
        kwargs = self.current_kwargs()  #         kwargs = self.current_kwargs()
        vals = self.src._values(_slice)  #         vals = self.src._values(_slice)
        return self.template.unaplied.vec_f(vals, **kwargs)  #         return self.template.unaplied.vec_f(vals,**kwargs)


    def _get_mask(self, mask):  #     def _get_mask(self, mask):
        kwargs = self.current_kwargs()  #         kwargs = self.current_kwargs()
        vals = self.src._values(_slice)  #         vals = self.src._values(_slice)
        return self.template.unaplied.vec_f(vals, **kwargs)  #         return self.template.unaplied.vec_f(vals,**kwargs)
def remove_caches(s, parents):  # def remove_caches(s,parents):
    from data_tree.ops.cache import CachedSeries  #     from data_tree.ops.cache import CachedSeries
    _coconut_match_to = s  #     case s:
    _coconut_case_check_1 = False  #     case s:
    if _coconut.isinstance(_coconut_match_to, CachedSeries):  #     case s:
        _coconut_case_check_1 = True  #     case s:
    if _coconut_case_check_1:  #     case s:
        return parents[0]  #             return parents[0]

def instantiate_cmap(s, parents):  # def instantiate_cmap(s,parents):
    _coconut_match_to = s  #     case s:
    _coconut_case_check_2 = False  #     case s:
    if _coconut.isinstance(_coconut_match_to, CMapTemplate):  #     case s:
        _coconut_case_check_2 = True  #     case s:
    if _coconut_case_check_2:  #     case s:
        return CMapInstance(s.clone(parents))  #             return CMapInstance(s.clone(parents))

def control_series(s):  # def control_series(s):
    from data_tree import series  #     from data_tree import series
    converted = s.replace(remove_caches).replace(instantiate_cmap)  #     converted = s.replace(remove_caches).replace(instantiate_cmap)
    controllers = converted.traversed.filter(lambda _=None: (isinstance)(_, CMapInstance)).map(lambda _=None: (series)(_.controllers.values())).flatten().values  #     controllers = converted.traversed.filter(->_ `isinstance` CMapInstance ).map(->_.controllers.values()|>series).flatten().values
    controllers = ControllableWidget.zip(*controllers)  #     controllers = ControllableWidget.zip(*controllers)
    index = iwidget((0, len(converted), 1), description="index")  #     index = iwidget((0,len(converted),1),description="index")
    max_depth = len(converted.trace(0).ancestors()) + 1  #     max_depth = len(converted.trace(0).ancestors()) + 1
    depth = iwidget((0, max_depth, 1), value=1, min=1, description="depth")  #     depth = iwidget((0,max_depth,1),value=1,min=1,description="depth")
    def _id(i, d):  #     def _id(i,d):
        return converted.trace(i).show_traceback_html(depth=d)  #         return converted.trace(i).show_traceback_html(depth=d)

    trace = index.zip(depth)  #.star_map(_id)  #     trace = index.zip(depth)#.star_map(_id)
    return controllers.zip(trace).viz(lambda t: _id(*t[1]))  #     return controllers.zip(trace).viz(t->_id(*t[1]))
