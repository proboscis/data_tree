#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x39e367c4

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

from IPython.display import display  # from IPython.display import display
import ipywidgets as widgets  # import ipywidgets as widgets
from typing import Tuple  # from typing import Tuple
from rx.subjects import BehaviorSubject  # from rx.subjects import BehaviorSubject
from lazy import lazy  # from lazy import lazy
from concurrent.futures import Future  # from concurrent.futures import Future
from data_tree.mp_util import mt_run  # from data_tree.mp_util import mt_run
from data_tree.coconut.visualization import infer_widget  # from data_tree.coconut.visualization import infer_widget
from rx import Observable  # from rx import Observable
from itertools import chain  # from itertools import chain
from inspect import signature  # from inspect import signature

class ControllableWidget():  #mutable subject with controller  # class ControllableWidget: #mutable subject with controller
    @property  #     @property
    def controllers(self):  #     def controllers(self):
        return ([])  #         return []


    @property  #     @property
    def value(self) -> 'BehaviorSubject':  #     def value(self)->BehaviorSubject:
        return (None)  #         return None


    @property  #     @property
    def parents(self):  #     def parents(self):
        return ([])  #         return []


    def _infer_widget(self, val):  #     def _infer_widget(self,val):
        return (infer_widget(val))  #         return infer_widget(val)


    @lazy  #     @lazy
    def widget(self):  #     def widget(self):
        out = widgets.Output()  #         out = widgets.Output()
        def _on_change(item):  #         def _on_change(item):
# I want the item to be cleared before item comes.
            out.clear_output(wait=False)  #             out.clear_output(wait=False)
            with out:  #             with out:
                (display)((self._infer_widget)(item))  #                 item |> self._infer_widget |> display

        self.value.subscribe(_on_change, _on_change, _on_change)  #         self.value.subscribe(
        box = widgets.VBox([out,])  #         box = widgets.VBox([out])
        box.layout.border = "solid 2px"  #         box.layout.border="solid 2px"
        return (box)  #         return box


    def _ipython_display_(self):  #     def _ipython_display_(self):
        display(widgets.VBox([*self.controllers, self.widget]))  #         display(widgets.VBox([*self.controllers,self.widget]))


    def map(self, f):  #     def map(self,f):
        return (MappedCW(f, self))  #         return MappedCW(f,self)


    def star_map(self, f):  #     def star_map(self,f):
        return (MappedCW(f, self, expand=True))  #         return MappedCW(f,self,expand=True)


    def zip(self, *others):  #     def zip(self,*others):
        return (ZippedCW(self, *others))  #         return ZippedCW(self,*others)


    def viz(self, visualizer):  #     def viz(self,visualizer):
        return (VizCW(self, visualizer))  #         return VizCW(self,visualizer)


    @lazy  #     @lazy
    def roots(self):  #     def roots(self):
        result = []  #         result = []
        for p in (self.parents):  #         for p in self.parents:
            if not p.parents:  #             if not p.parents:
                result.append(p)  #                 result.append(p)
            else:  #             else:
                result += p.roots  #                 result += p.roots
        return ((list)({id(r): r for r in result}.values()))  #         return {id(r):r for r in result}.values() |> list


    @lazy  #     @lazy
    def traverse_parents(self):  #     def traverse_parents(self):
        result = []  #         result = []
        for p in (self.parents):  #         for p in self.parents:
            result.append(p)  #             result.append(p)
            result += p.traverse_parents  #             result += p.traverse_parents
        return ((list)({id(r): r for r in result}.values()))  #         return {id(r):r for r in result}.values() |> list


    def get(self):  #     def get(self):
        return (self.value.value)  #         return self.value.value



_coconut_call_set_names(ControllableWidget)  # class IpyWidgetAdapter(ControllableWidget):
class IpyWidgetAdapter(ControllableWidget):  # class IpyWidgetAdapter(ControllableWidget):
    def __init__(self, widget):  #     def __init__(self,widget):
        self._widget = widget  #         self._widget=widget
        self._value = BehaviorSubject(self._widget.value)  #.buffer_with_time(1).filter(bool).map(.[-1])  #         self._value = BehaviorSubject(self._widget.value)#.buffer_with_time(1).filter(bool).map(.[-1])
        def emit_if_different(item):  #         def emit_if_different(item):
#if item["old"] != item["new"]:
            self._value.on_next(item["new"])  #             self._value.on_next(item["new"])

        self._widget.observe(emit_if_different, names="value")  #         self._widget.observe(emit_if_different,names="value")
#self._widget.observe(self._value.on_next,names="value")



    @property  #     @property
    def value(self):  #     def value(self):
        return (self._value)  #         return self._value


    @property  #     @property
    def controllers(self):  #     def controllers(self):
        return ([self._widget,])  #         return [self._widget]


_coconut_call_set_names(IpyWidgetAdapter)  # unique_objects = items ->{id(i):i for i in items}.values() |> tuple
unique_objects = lambda items: (tuple)({id(i): i for i in items}.values())  # unique_objects = items ->{id(i):i for i in items}.values() |> tuple
class Sourced(ControllableWidget):  # class Sourced(ControllableWidget):

    @property  #     @property
    def parents(self):  #     def parents(self):
        raise RuntimeError("Not implemented")  #         raise RuntimeError("Not implemented")


    @property  #     @property
    def controllers(self):  #     def controllers(self):
        return ((unique_objects)((chain)(*(map)(_coconut.operator.attrgetter("controllers"), self.parents))))  #         return self.parents |> map$(.controllers) |*> chain |> unique_objects

    """
    @lazy
    def widget(self):
        out = widgets.Output()
        def data_output(data):
            out.clear_output(wait=False)
            with out:
                data |> self._infer_widget |> display
        roots = [r.value for r in self.roots]
        #Observable.combine_latest(roots,(*i)->tuple(i)).subscribe(_->out.clear_output(wait=False))

        def _on_change(item):
            if item `isinstance` Future:
                data_output("waiting for future")
                item.add_done_callback(f->data_output(f.result()))
            else:
                data_output(item)

        self.value.subscribe(
            _on_change,
            _on_change,
            _on_change)
        #out.layout.border="solid 2px"
        box = widgets.VBox([*self.controllers,out])
        box.layout.border="solid 2px"
        return box
    """  #     """

_coconut_call_set_names(Sourced)  # class MappedCW(Sourced):
class MappedCW(Sourced):  # class MappedCW(Sourced):
    def __init__(self, f, src, expand=False):  #     def __init__(self,f,src,expand=False):
        self.src = src  #         self.src = src
        self.f = f  #         self.f = f
        self.expand = expand  #         self.expand=expand


    @property  #     @property
    def parents(self):  #     def parents(self):
        return ([self.src,])  #         return [self.src]


    @lazy  #     @lazy
    def value(self):  #     def value(self):
        subject = BehaviorSubject(None)  #         subject = BehaviorSubject(None)
        if self.expand:  #         if self.expand:
            self.src.value.map(lambda args: self.f(*args)).subscribe(subject.on_next)  #             self.src.value.map(args->self.f(*args)).subscribe(subject.on_next)
        else:  #         else:
            self.src.value.map(self.f).subscribe(subject.on_next)  #             self.src.value.map(self.f).subscribe(subject.on_next)
        return (subject)  #         return subject



_coconut_call_set_names(MappedCW)  # class ZippedCW(Sourced):
class ZippedCW(Sourced):  # class ZippedCW(Sourced):
    def __init__(self, *srcs):  #     def __init__(self,*srcs):
        self.srcs = srcs  #         self.srcs = srcs


    @property  #     @property
    def parents(self):  #     def parents(self):
        return (list(self.srcs))  #         return list(self.srcs)


    @property  #     @property
    def controller(self):  #     def controller(self):
        return (widgets.VBox([w.controller for w in self.srcs]))  #         return widgets.VBox([w.controller for w in self.srcs])


    @lazy  #     @lazy
    def value(self):  #     def value(self):
#root_ids = {id(p) for p in self.traverse_parents}
#tgts = [w.value for w in self.srcs if id(w) not in root_ids]
        tgts = [s.value for s in self.srcs]  #         tgts = [s.value for s in self.srcs]
#current_values = ()->tuple([w.value.value for w in self.srcs])
        subject = BehaviorSubject(None)  #         subject = BehaviorSubject(None)
#if not tgts:
#    tgts = [self.parents[0].value]
        combined = Observable.combine_latest(tgts, lambda *i: tuple(i))  #         combined = Observable.combine_latest(tgts,(*i)->tuple(i))
        combined.subscribe(subject.on_next)  #.subscribe(t -> subject.on_next(current_values()))  #         combined.subscribe(subject.on_next)#.subscribe(t -> subject.on_next(current_values()))
        return (subject)  #         return subject


_coconut_call_set_names(ZippedCW)  # class VizCW(Sourced):
class VizCW(Sourced):  # class VizCW(Sourced):
    def __init__(self, src, visualizer: '_coconut.typing.Callable[[object], "displayable"]'):  #     def __init__(self,src,visualizer:object->"displayable"):
        self.value = src  #         self.value = src
        self.visualizer = visualizer  #         self.visualizer = visualizer

    @lazy  #     @lazy
    def value(self):  #     def value(self):
        return (self.src.value)  #         return self.src.value

    @property  #     @property
    def parents(self):  #     def parents(self):
        return ([self.src,])  #         return [self.src]


    def _infer_widget(self, val):  #     def _infer_widget(self,val):
        return (self.visualizer(val))  #         return self.visualizer(val)


_coconut_call_set_names(VizCW)  # def infer_ipywidget(item,**kwargs):
def infer_ipywidget(item, **kwargs):  # def infer_ipywidget(item,**kwargs):
    _coconut_case_match_to_0 = item  #     case item:
    _coconut_case_match_check_0 = False  #     case item:
    _coconut_match_set_name_start = _coconut_sentinel  #     case item:
    _coconut_match_set_name_end = _coconut_sentinel  #     case item:
    _coconut_match_set_name_step = _coconut_sentinel  #     case item:
    if (_coconut.isinstance(_coconut_case_match_to_0, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_0) == 3) and (_coconut.isinstance(_coconut_case_match_to_0[0], int)) and (_coconut.isinstance(_coconut_case_match_to_0[1], int)) and (_coconut.isinstance(_coconut_case_match_to_0[2], int)):  #     case item:
        _coconut_match_set_name_start = _coconut_case_match_to_0[0]  #     case item:
        _coconut_match_set_name_end = _coconut_case_match_to_0[1]  #     case item:
        _coconut_match_set_name_step = _coconut_case_match_to_0[2]  #     case item:
        _coconut_case_match_check_0 = True  #     case item:
    if _coconut_case_match_check_0:  #     case item:
        if _coconut_match_set_name_start is not _coconut_sentinel:  #     case item:
            start = _coconut_match_set_name_start  #     case item:
        if _coconut_match_set_name_end is not _coconut_sentinel:  #     case item:
            end = _coconut_match_set_name_end  #     case item:
        if _coconut_match_set_name_step is not _coconut_sentinel:  #     case item:
            step = _coconut_match_set_name_step  #     case item:
    if _coconut_case_match_check_0:  #     case item:
        _kwargs = dict(value=(start + end) // 2, max=end, min=start, step=step)  #             _kwargs = dict(
        return (widgets.IntSlider(**{**_kwargs, **kwargs}))  #             return widgets.IntSlider(**{**_kwargs,**kwargs})
    if not _coconut_case_match_check_0:  #         match (start is float,end,step):
        _coconut_match_set_name_start = _coconut_sentinel  #         match (start is float,end,step):
        _coconut_match_set_name_end = _coconut_sentinel  #         match (start is float,end,step):
        _coconut_match_set_name_step = _coconut_sentinel  #         match (start is float,end,step):
        if (_coconut.isinstance(_coconut_case_match_to_0, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_0) == 3) and (_coconut.isinstance(_coconut_case_match_to_0[0], float)):  #         match (start is float,end,step):
            _coconut_match_set_name_start = _coconut_case_match_to_0[0]  #         match (start is float,end,step):
            _coconut_match_set_name_end = _coconut_case_match_to_0[1]  #         match (start is float,end,step):
            _coconut_match_set_name_step = _coconut_case_match_to_0[2]  #         match (start is float,end,step):
            _coconut_case_match_check_0 = True  #         match (start is float,end,step):
        if _coconut_case_match_check_0:  #         match (start is float,end,step):
            if _coconut_match_set_name_start is not _coconut_sentinel:  #         match (start is float,end,step):
                start = _coconut_match_set_name_start  #         match (start is float,end,step):
            if _coconut_match_set_name_end is not _coconut_sentinel:  #         match (start is float,end,step):
                end = _coconut_match_set_name_end  #         match (start is float,end,step):
            if _coconut_match_set_name_step is not _coconut_sentinel:  #         match (start is float,end,step):
                step = _coconut_match_set_name_step  #         match (start is float,end,step):
        if _coconut_case_match_check_0:  #         match (start is float,end,step):
            _kwargs = dict(value=(start + end) / 2, max=end, min=start, step=step)  #             _kwargs = dict(
            return (widgets.FloatSlider(**{**_kwargs, **kwargs}))  #             return widgets.FloatSlider(**{**_kwargs,**kwargs})
    if not _coconut_case_match_check_0:  #         match _ is str:
        if _coconut.isinstance(_coconut_case_match_to_0, str):  #         match _ is str:
            _coconut_case_match_check_0 = True  #         match _ is str:
        if _coconut_case_match_check_0:  #         match _ is str:
            return (widgets.Text(value=item, **kwargs))  #             return widgets.Text(value=item,**kwargs)
    if not _coconut_case_match_check_0:  #         match _ is list:
        if _coconut.isinstance(_coconut_case_match_to_0, list):  #         match _ is list:
            _coconut_case_match_check_0 = True  #         match _ is list:
        if _coconut_case_match_check_0:  #         match _ is list:
            return (widgets.Dropdown(options=item, index=0, value=item[0], **kwargs))  #             return widgets.Dropdown(options=item,index=0,value=item[0],**kwargs)
    if not _coconut_case_match_check_0:  #     else:
        raise RuntimeError("cannot convert {_coconut_format_0} to widget".format(_coconut_format_0=(item)))  #         raise RuntimeError(f"cannot convert {item} to widget")


iwa = IpyWidgetAdapter  # iwa = IpyWidgetAdapter
iwa_helper = _coconut_base_compose(infer_ipywidget, (iwa, 0, False))  # iwa_helper = infer_ipywidget ..> iwa
iwidget = iwa_helper  # iwidget = iwa_helper
def ensure_iwa(key, item):  # def ensure_iwa(key,item):
    _coconut_case_match_to_1 = item  #     case item:
    _coconut_case_match_check_1 = False  #     case item:
    if _coconut.isinstance(_coconut_case_match_to_1, ControllableWidget):  #     case item:
        _coconut_case_match_check_1 = True  #     case item:
    if _coconut_case_match_check_1:  #     case item:
        return (item)  #             return item
    if not _coconut_case_match_check_1:  #     else:
        return (iwa_helper(item, description=key))  #         return iwa_helper(item,description=key)

class FunctionCW(Sourced):  # class FunctionCW(Sourced):
    def __init__(self, f, **applications: 'dict'):  #     def __init__(self,f,**applications:dict):
        """
        each value of application must be CW
        """  #         """
        self.f = f  #         self.f = f
        self.signature = signature(self.f)  #         self.signature=signature(self.f)
        self.applications = {k: ensure_iwa(k, a) for k, a in applications.items()}  #         self.applications = {k:ensure_iwa(k,a) for k,a in applications.items()}



    def __call__(self, *args, **kwargs):  #     def __call__(self,*args,**kwargs):
        non_applied = [k for k in self.signature.parameters.keys() if k not in self.applications]  #         non_applied = [k for k in self.signature.parameters.keys() if k not in self.applications]
        application = self.applications.copy()  #         application = self.applications.copy()
        for nak, a in (zip(non_applied, args)):  #         for nak,a in zip(non_applied,args):
            application[nak] = a  #             application[nak] = a
        for k, a in (kwargs.items()):  #         for k,a in kwargs.items():
            application[k] = a  #             application[k] = a
        return (FunctionCW(self.f, **application))  #         return FunctionCW(self.f,**application)


    @lazy  #     @lazy
    def parents(self):  #     def parents(self):
        return ([v for k, v in self.applications.items()])  #         return [v for k,v in self.applications.items()]


    @lazy  #     @lazy
    def value(self):  #     def value(self):
        keys, obs = zip(*((list)(self.applications.items())))  #         keys,obs = zip(*(self.applications.items() |> list))
        obs = [o.value for o in obs]  #         obs = [o.value for o in obs]
        subject = BehaviorSubject(None)  #         subject = BehaviorSubject(None)
        Observable.combine_latest(obs, lambda *t: (self.f)(**((dict)(zip(keys, t))))).subscribe(subject.on_next)  #         Observable.combine_latest(obs,(*t)-> (zip(keys,t)|>dict) |**> self.f).subscribe(subject.on_next)
        return (subject)  #         return subject


_coconut_call_set_names(FunctionCW)  #         return subject
