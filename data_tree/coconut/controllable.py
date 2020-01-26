#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xee62c790

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

class ControllableWidget:  #mutable subject with controller  # class ControllableWidget: #mutable subject with controller
    @property  #     @property
    def controllers(self):  #     def controllers(self):
        return []  #         return []

    @property  #     @property
    def value(self) -> 'BehaviorSubject':  #     def value(self)->BehaviorSubject:
        return None  #         return None

    @property  #     @property
    def parents(self):  #     def parents(self):
        return []  #         return []

    @lazy  #     @lazy
    def widget(self):  #     def widget(self):
        out = widgets.Output()  #         out = widgets.Output()
        def _on_change(item):  #         def _on_change(item):
# I want the item to be cleared before item comes.
            out.clear_output(wait=False)  #             out.clear_output(wait=False)
            with out:  #             with out:
                (display)((infer_widget)(item))  #                 item |> infer_widget |> display
        self.value.subscribe(_on_change, _on_change, _on_change)  #         self.value.subscribe(
        box = widgets.VBox([*self.controllers, out])  #         box = widgets.VBox([*self.controllers,out])
#box.layout = widgets.Layout(
#            #grid_template_columns="auto auto auto",
#            border="solid 2px"
#        )
        box.layout.border = "solid 2px"  #         box.layout.border="solid 2px"
        return box  #         return box

    def _ipython_display_(self):  #     def _ipython_display_(self):
        display(self.widget)  #         display(self.widget)

    def map(self, f):  #     def map(self,f):
        return MappedCW(f, self)  #         return MappedCW(f,self)

    def star_map(self, f):  #     def star_map(self,f):
        return MappedCW(f, self, expand=True)  #         return MappedCW(f,self,expand=True)

    def zip(self, *others):  #     def zip(self,*others):
        return ZippedCW(self, *others)  #         return ZippedCW(self,*others)

    @lazy  #     @lazy
    def roots(self):  #     def roots(self):
        result = []  #         result = []
        for p in self.parents:  #         for p in self.parents:
            if not p.parents:  #             if not p.parents:
                result.append(p)  #                 result.append(p)
            else:  #             else:
                result += p.roots  #                 result += p.roots
        return (list)({id(r): r for r in result}.values())  #         return {id(r):r for r in result}.values() |> list

    @lazy  #     @lazy
    def traverse_parents(self):  #     def traverse_parents(self):
        result = []  #         result = []
        for p in self.parents:  #         for p in self.parents:
            result.append(p)  #             result.append(p)
            result += p.traverse_parents  #             result += p.traverse_parents
        return (list)({id(r): r for r in result}.values())  #         return {id(r):r for r in result}.values() |> list

    def get(self):  #     def get(self):
        return self.value.value  #         return self.value.value


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
        return self._value  #         return self._value

    @property  #     @property
    def controllers(self):  #     def controllers(self):
        return [self._widget]  #         return [self._widget]

unique_objects = lambda items: (tuple)({id(i): i for i in items}.values())  # unique_objects = items ->{id(i):i for i in items}.values() |> tuple
class Sourced(ControllableWidget):  # class Sourced(ControllableWidget):

    @property  #     @property
    def parents(self):  #     def parents(self):
        raise RuntimeError("Not implemented")  #         raise RuntimeError("Not implemented")

    @property  #     @property
    def controllers(self):  #     def controllers(self):
        return (unique_objects)((chain)(*map(_coconut.operator.attrgetter("controllers"), self.parents)))  #         return self.parents |> map$(.controllers) |*> chain |> unique_objects
    @lazy  #     @lazy
    def widget(self):  #     def widget(self):
        out = widgets.Output()  #         out = widgets.Output()
        def data_output(data):  #         def data_output(data):
            out.clear_output(wait=False)  #             out.clear_output(wait=False)
            with out:  #             with out:
                (display)((infer_widget)(data))  #                 data |> infer_widget |> display
        roots = [r.value for r in self.roots]  #         roots = [r.value for r in self.roots]
#Observable.combine_latest(roots,(*i)->tuple(i)).subscribe(_->out.clear_output(wait=False))

        def _on_change(item):  #         def _on_change(item):
            if (isinstance)(item, Future):  #             if item `isinstance` Future:
                data_output("waiting for future")  #                 data_output("waiting for future")
                item.add_done_callback(lambda f: data_output(f.result()))  #                 item.add_done_callback(f->data_output(f.result()))
            else:  #             else:
                data_output(item)  #                 data_output(item)

        self.value.subscribe(_on_change, _on_change, _on_change)  #         self.value.subscribe(
#out.layout.border="solid 2px"
        box = widgets.HBox([out])  #         box = widgets.HBox([out])
        box.layout.border = "solid 2px"  #         box.layout.border="solid 2px"
        return box  #         return box

    def _ipython_display_(self):  #     def _ipython_display_(self):
#parent_widgets = [p.widget for p in self.traverse_parents]
#display(widgets.VBox((parent_widgets |> reversed |> list) + [self.widget]))
        display(widgets.VBox([*self.controllers, self.widget]))  #         display(widgets.VBox([*self.controllers,self.widget]))

class MappedCW(Sourced):  # class MappedCW(Sourced):
    def __init__(self, f, src, expand=False):  #     def __init__(self,f,src,expand=False):
        self.src = src  #         self.src = src
        self.f = f  #         self.f = f
        self.expand = expand  #         self.expand=expand

    @property  #     @property
    def parents(self):  #     def parents(self):
        return [self.src]  #         return [self.src]

    @lazy  #     @lazy
    def value(self):  #     def value(self):
        subject = BehaviorSubject(None)  #         subject = BehaviorSubject(None)
        if self.expand:  #         if self.expand:
            self.src.value.map(lambda args: self.f(*args)).subscribe(subject.on_next)  #             self.src.value.map(args->self.f(*args)).subscribe(subject.on_next)
        else:  #         else:
            self.src.value.map(self.f).subscribe(subject.on_next)  #             self.src.value.map(self.f).subscribe(subject.on_next)
        return subject  #         return subject


class ZippedCW(Sourced):  # class ZippedCW(Sourced):
    def __init__(self, *srcs):  #     def __init__(self,*srcs):
        self.srcs = srcs  #         self.srcs = srcs

    @property  #     @property
    def parents(self):  #     def parents(self):
        return list(self.srcs)  #         return list(self.srcs)

    @property  #     @property
    def controller(self):  #     def controller(self):
        return widgets.VBox([w.controller for w in self.srcs])  #         return widgets.VBox([w.controller for w in self.srcs])

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
        return subject  #         return subject

def infer_ipywidget(item, **kwargs):  # def infer_ipywidget(item,**kwargs):
    _coconut_match_to = item  #     case item:
    _coconut_case_check_0 = False  #     case item:
    if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut.isinstance(_coconut_match_to[0], int)) and (_coconut.isinstance(_coconut_match_to[1], int)) and (_coconut.isinstance(_coconut_match_to[2], int)):  #     case item:
        start = _coconut_match_to[0]  #     case item:
        end = _coconut_match_to[1]  #     case item:
        step = _coconut_match_to[2]  #     case item:
        _coconut_case_check_0 = True  #     case item:
    if _coconut_case_check_0:  #     case item:
        _kwargs = dict(value=(start + end) // 2, max=end, min=start, step=step)  #             _kwargs = dict(
        return widgets.IntSlider(**{**_kwargs, **kwargs})  #             return widgets.IntSlider(**{**_kwargs,**kwargs})
    if not _coconut_case_check_0:  #         match (start is float,end,step):
        if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3) and (_coconut.isinstance(_coconut_match_to[0], float)):  #         match (start is float,end,step):
            start = _coconut_match_to[0]  #         match (start is float,end,step):
            end = _coconut_match_to[1]  #         match (start is float,end,step):
            step = _coconut_match_to[2]  #         match (start is float,end,step):
            _coconut_case_check_0 = True  #         match (start is float,end,step):
        if _coconut_case_check_0:  #         match (start is float,end,step):
            _kwargs = dict(value=(start + end) / 2, max=end, min=start, step=step)  #             _kwargs = dict(
            return widgets.FloatSlider(**{**_kwargs, **kwargs})  #             return widgets.FloatSlider(**{**_kwargs,**kwargs})
    if not _coconut_case_check_0:  #         match _ is str:
        if _coconut.isinstance(_coconut_match_to, str):  #         match _ is str:
            _coconut_case_check_0 = True  #         match _ is str:
        if _coconut_case_check_0:  #         match _ is str:
            return widgets.Text(value=item, **kwargs)  #             return widgets.Text(value=item,**kwargs)
    if not _coconut_case_check_0:  #         match _ is list:
        if _coconut.isinstance(_coconut_match_to, list):  #         match _ is list:
            _coconut_case_check_0 = True  #         match _ is list:
        if _coconut_case_check_0:  #         match _ is list:
            return widgets.Dropdown(options=item, index=0, value=item[0], **kwargs)  #             return widgets.Dropdown(options=item,index=0,value=item[0],**kwargs)
    if not _coconut_case_check_0:  #     else:
        raise RuntimeError("cannot convert {_coconut_format_0} to widget".format(_coconut_format_0=(item)))  #         raise RuntimeError(f"cannot convert {item} to widget")

iwa = IpyWidgetAdapter  # iwa = IpyWidgetAdapter
iwa_helper = _coconut_base_compose(infer_ipywidget, (iwa, 0))  # iwa_helper = infer_ipywidget ..> iwa
iwidget = iwa_helper  # iwidget = iwa_helper
def ensure_iwa(key, item):  # def ensure_iwa(key,item):
    _coconut_match_to = item  #     case item:
    _coconut_case_check_1 = False  #     case item:
    if _coconut.isinstance(_coconut_match_to, ControllableWidget):  #     case item:
        _coconut_case_check_1 = True  #     case item:
    if _coconut_case_check_1:  #     case item:
        return item  #             return item
    if not _coconut_case_check_1:  #     else:
        return iwa_helper(item, description=key)  #         return iwa_helper(item,description=key)
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
        for nak, a in zip(non_applied, args):  #         for nak,a in zip(non_applied,args):
            application[nak] = a  #             application[nak] = a
        for k, a in kwargs.items():  #         for k,a in kwargs.items():
            application[k] = a  #             application[k] = a
        return FunctionCW(self.f, **application)  #         return FunctionCW(self.f,**application)

    @lazy  #     @lazy
    def parents(self):  #     def parents(self):
        return [v for k, v in self.applications.items()]  #         return [v for k,v in self.applications.items()]

    @lazy  #     @lazy
    def value(self):  #     def value(self):
        keys, obs = zip(*((list)(self.applications.items())))  #         keys,obs = zip(*(self.applications.items() |> list))
        obs = [o.value for o in obs]  #         obs = [o.value for o in obs]
        subject = BehaviorSubject(None)  #         subject = BehaviorSubject(None)
        Observable.combine_latest(obs, lambda *t: (self.f)(**((dict)(zip(keys, t))))).subscribe(subject.on_next)  #         Observable.combine_latest(obs,(*t)-> (zip(keys,t)|>dict) |**> self.f).subscribe(subject.on_next)
        return subject  #         return subject
