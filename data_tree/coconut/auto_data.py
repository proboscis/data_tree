#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xc9713a7c

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

from typing import Mapping  # from typing import Mapping
from data_tree.coconut.astar import AStarSolver  # from data_tree.coconut.astar import AStarSolver
from IPython.display import display  # from IPython.display import display
import dill  # to make the lambda functions picklable  # import dill # to make the lambda functions picklable
from loguru import logger  # from loguru import logger
"""
What I want to achieve is to generate a class.
I want to generate a AutoImage class form AutoData
well the solution is to just use partially applied constructor. that's it.
"""  # """
def identity(a):  # def identity(a):
    return a  #     return a

class _CastLambda:  # class _CastLambda:
    def __init__(self, rule, name, swap, cost=1):  #     def __init__(self,rule,name,swap,cost=1):
        self.rule = rule  #         self.rule = rule
        self.name = name  #         self.name = name
        self.swap = swap  #         self.swap = swap
        self.cost = cost  #         self.cost = cost
    def __call__(self, state):  #     def __call__(self,state):
        new_states = self.rule(state)  #         new_states = self.rule(state)
        if new_states is not None:  #         if new_states is not None:
            if self.name is None:  #             if self.name is None:
                cast_name = "{_coconut_format_0}".format(_coconut_format_0=(self.rule.__name__))  #                 cast_name = f"{self.rule.__name__}"
            else:  #             else:
                cast_name = self.name  #                 cast_name=self.name
            if self.swap:  #             if self.swap:
                return [(identity, new_state, cast_name, self.cost) for new_state in new_states]  #                 return [(identity,new_state,cast_name,self.cost) for new_state in new_states]
            else:  #             else:
                return [(identity, new_state, self.cost, cast_name) for new_state in new_states]  #                 return [(identity,new_state,self.cost,cast_name) for new_state in new_states]
        else:  #         else:
            return None  #             return None
class _ConversionLambda:  # class _ConversionLambda:
    def __init__(self, rule, cost=1):  #     def __init__(self,rule,cost=1):
        self.rule = rule  #         self.rule = rule
        self.cost = cost  #         self.cost = cost
    def __call__(self, state):  #     def __call__(self,state):
        edges = self.rule(state)  #         edges = self.rule(state)
        if edges is None:  #         if edges is None:
            return []  #             return []
        result = []  #         result = []
        for edge in edges:  #         for edge in edges:
            _coconut_match_to = edge  #             case edge:
            _coconut_case_check_0 = False  #             case edge:
            if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2):  #             case edge:
                converter = _coconut_match_to[0]  #             case edge:
                new_state = _coconut_match_to[1]  #             case edge:
                _coconut_case_check_0 = True  #             case edge:
            if _coconut_case_check_0:  #             case edge:
                result.append((converter, new_state, self.cost, converter.__name__))  #                     result.append((converter,new_state,self.cost,converter.__name__))
            if not _coconut_case_check_0:  #                 match (converter,new_state,name):
                if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3):  #                 match (converter,new_state,name):
                    converter = _coconut_match_to[0]  #                 match (converter,new_state,name):
                    new_state = _coconut_match_to[1]  #                 match (converter,new_state,name):
                    name = _coconut_match_to[2]  #                 match (converter,new_state,name):
                    _coconut_case_check_0 = True  #                 match (converter,new_state,name):
                if _coconut_case_check_0:  #                 match (converter,new_state,name):
                    result.append((converter, new_state, self.cost, name))  #                     result.append((converter,new_state,self.cost,name))
            if not _coconut_case_check_0:  #                 match (converter,new_state,name,score):
                if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4):  #                 match (converter,new_state,name,score):
                    converter = _coconut_match_to[0]  #                 match (converter,new_state,name,score):
                    new_state = _coconut_match_to[1]  #                 match (converter,new_state,name,score):
                    name = _coconut_match_to[2]  #                 match (converter,new_state,name,score):
                    score = _coconut_match_to[3]  #                 match (converter,new_state,name,score):
                    _coconut_case_check_0 = True  #                 match (converter,new_state,name,score):
                if _coconut_case_check_0:  #                 match (converter,new_state,name,score):
                    result.append((converter, new_state, score, name))  #                     result.append((converter,new_state,score,name))
            if not _coconut_case_check_0:  #                 match _:
                _coconut_case_check_0 = True  #                 match _:
                if _coconut_case_check_0:  #                 match _:
                    raise RuntimeError("rule:{_coconut_format_0} returned invalid edge:{_coconut_format_1}.".format(_coconut_format_0=(self.rule), _coconut_format_1=(edge)))  #                     raise RuntimeError(f"rule:{self.rule} returned invalid edge:{edge}.")
        return result  #         return result
class _SmartConversionLambda:  # class _SmartConversionLambda:
    def __init__(self, rule, cost=1):  #     def __init__(self,rule,cost=1):
        self.rule = rule  #         self.rule = rule
        self.cost = cost  #         self.cost = cost
    def __call__(self, state, end):  #     def __call__(self,state,end):
        edges = self.rule(state, end)  #         edges = self.rule(state,end)
        if edges is None:  #         if edges is None:
            return []  #             return []
        result = []  #         result = []
        for edge in edges:  #         for edge in edges:
            _coconut_match_to = edge  #             case edge:
            _coconut_case_check_1 = False  #             case edge:
            if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2):  #             case edge:
                converter = _coconut_match_to[0]  #             case edge:
                new_state = _coconut_match_to[1]  #             case edge:
                _coconut_case_check_1 = True  #             case edge:
            if _coconut_case_check_1:  #             case edge:
                result.append((converter, new_state, self.cost, converter.__name__))  #                     result.append((converter,new_state,self.cost,converter.__name__))
            if not _coconut_case_check_1:  #                 match (converter,new_state,name):
                if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3):  #                 match (converter,new_state,name):
                    converter = _coconut_match_to[0]  #                 match (converter,new_state,name):
                    new_state = _coconut_match_to[1]  #                 match (converter,new_state,name):
                    name = _coconut_match_to[2]  #                 match (converter,new_state,name):
                    _coconut_case_check_1 = True  #                 match (converter,new_state,name):
                if _coconut_case_check_1:  #                 match (converter,new_state,name):
                    result.append((converter, new_state, self.cost, name))  #                     result.append((converter,new_state,self.cost,name))
            if not _coconut_case_check_1:  #                 match (converter,new_state,name,score):
                if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 4):  #                 match (converter,new_state,name,score):
                    converter = _coconut_match_to[0]  #                 match (converter,new_state,name,score):
                    new_state = _coconut_match_to[1]  #                 match (converter,new_state,name,score):
                    name = _coconut_match_to[2]  #                 match (converter,new_state,name,score):
                    score = _coconut_match_to[3]  #                 match (converter,new_state,name,score):
                    _coconut_case_check_1 = True  #                 match (converter,new_state,name,score):
                if _coconut_case_check_1:  #                 match (converter,new_state,name,score):
                    result.append((converter, new_state, score, name))  #                     result.append((converter,new_state,score,name))
            if not _coconut_case_check_1:  #                 match _:
                _coconut_case_check_1 = True  #                 match _:
                if _coconut_case_check_1:  #                 match _:
                    raise RuntimeError("rule:{_coconut_format_0} returned invalid edge:{_coconut_format_1}.".format(_coconut_format_0=(self.rule), _coconut_format_1=(edge)))  #                     raise RuntimeError(f"rule:{self.rule} returned invalid edge:{edge}.")
        return result  #         return result
class AutoSolver:  # class AutoSolver:
    """
    TODO stop using local lambda in order to make this class picklable
    Factory for an AutoData class
    """  #     """
    def __init__(self, rules, smart_rules, heuristics=lambda x, y: 0, edge_cutter=lambda x, y, end: False):  #     def __init__(self,rules,smart_rules,heuristics=lambda x,y:0,edge_cutter=lambda x,y,end:False):
        self.initial_rules = rules  #         self.initial_rules = rules
        self.smart_rules = smart_rules  #         self.smart_rules = smart_rules
        self.solver = AStarSolver(rules=self.initial_rules.copy(), smart_rules=self.smart_rules.copy(), heuristics=heuristics, edge_cutter=edge_cutter)  #         self.solver = AStarSolver(

    @staticmethod  #     @staticmethod
    def create_cast_rule(rule, name=None, _swap=False, cost=1):  #     def create_cast_rule(rule,name=None,_swap=False,cost=1):
        """
        rule: State->List[State] # should return list of possible casts without data conversion.
        """  #         """
        return _CastLambda(rule, name, _swap, cost=cost)  #         return _CastLambda(rule,name,_swap,cost=cost)

    def add_cast(self, rule, name=None):  #     def add_cast(self,rule,name=None):
        """
        rule: State->List[State] # should return list of possible casts without data conversion.
        """  #         """

        self.add_conversion(AutoSolver.create_cast_rule(rule, name=name, _swap=True))  #         self.add_conversion(AutoSolver.create_cast_rule(rule,name=name,_swap=True))

    def add_alias(self, a, b):  #     def add_alias(self,a,b):
        self.add_cast(lambda state: [b] if state == a else None, name="alias: {_coconut_format_0}->{_coconut_format_1}".format(_coconut_format_0=(a), _coconut_format_1=(b)))  #         self.add_cast(state->[b] if state == a else None,name=f"alias: {a}->{b}")
        self.add_cast(lambda state: [a] if state == b else None, name="alias: {_coconut_format_0}->{_coconut_format_1}".format(_coconut_format_0=(b), _coconut_format_1=(a)))  #         self.add_cast(state->[a] if state == b else None,name=f"alias: {b}->{a}")

    @staticmethod  #     @staticmethod
    def create_alias_rule(a, b):  #     def create_alias_rule(a,b):
        def caster(state):  #         def caster(state):
            if state == a:  #            if state == a:
                return [b]  #                return [b]
            elif state == b:  #            elif state == b:
                return [a]  #                return [a]
        return AutoSolver.create_cast_rule(caster, "alias:{_coconut_format_0}=={_coconut_format_1}".format(_coconut_format_0=(a), _coconut_format_1=(b)))  #         return AutoSolver.create_cast_rule(caster,f"alias:{a}=={b}")

    @staticmethod  #     @staticmethod
    def create_conversion_rule(rule):  #     def create_conversion_rule(rule):
        return _ConversionLambda(rule)  #         return _ConversionLambda(rule)
    @staticmethod  #     @staticmethod
    def create_smart_conversion_rule(rule):  #     def create_smart_conversion_rule(rule):
        return _SmartConversionLambda(rule)  #         return _SmartConversionLambda(rule)

    def add_conversion(self, rule):  #     def add_conversion(self,rule):
        """
        rule: State->List[(converter,new_state,name(optional),cost(optional))]
        """  #         """
        self.solver.add_rule(AutoSolver.create_conversion_rule(rule))  #         self.solver.add_rule(AutoSolver.create_conversion_rule(rule))

    def reset_solver(self,):  #     def reset_solver(self,):
        self.solver = AStarSolver(rules=self.initial_rules.copy(), smart_rules=self.smart_rules.copy(), heuristics=heuristics, edge_cutter=edge_cutter)  #         self.solver = AStarSolver(

    def debug_conversion(self, a, b, samples):  #     def debug_conversion(self,a,b,samples):
        x = samples  #         x = samples
        edges = self.solver.search_direct(a, b).edges  #         edges = self.solver.search_direct(a,b).edges
        for edge in edges:  #         for edge in edges:
            print(edge)  #             print(edge)
            print(edge.f)  #             print(edge.f)
            x = edge.f(x)  #             x = edge.f(x)
            print("converted to type:{_coconut_format_0}".format(_coconut_format_0=(type(x))))  #             print(f"converted to type:{type(x)}")
            if (isinstance)(x, np.ndarray):  #             if x `isinstance` np.ndarray:
                print(x.shape)  #                 print(x.shape)
            print("converted:{_coconut_format_0}".format(_coconut_format_0=(x)))  #             print(f"converted:{x}")
        return x  #         return x

    def new_auto_data(self, value, format):  #     def new_auto_data(self,value,format):
        return AutoData(value, format, self)  #         return AutoData(value,format,self)





class TagMatcher:  # class TagMatcher:
    def __init__(self, **kwargs):  #     def __init__(self,**kwargs):
        self.kwargs = kwargs  #         self.kwargs = kwargs

    def __call__(self, state):  #     def __call__(self,state):
        if isinstance(state, Mapping):  #         if isinstance(state,Mapping):
            for k, v in self.kwargs.items():  #             for k,v in self.kwargs.items():
                if not k in state or not state[k] == v:  #                 if not k in state or not state[k] == v:
                    return False  #                     return False
            return True  #every item matched.  #             return True #every item matched.
        else:  #         else:
            return False  #             return False
    @property  #     @property
    def __name__(self):  #     def __name__(self):
        return "TagMatcher|{_coconut_format_0}".format(_coconut_format_0=(self.kwargs))  #         return f"TagMatcher|{self.kwargs}"

    def __str__(self):  #     def __str__(self):
        return self.__name__  #         return self.__name__


@memoize(1024)  # @memoize(1024)
def tag_matcher(**kwargs):  # def tag_matcher(**kwargs):
    return TagMatcher(**kwargs)  #     return TagMatcher(**kwargs)

SOLVERS = dict()  # SOLVERS = dict()

class AutoData:  # class AutoData:
    """
    Interface class for a user
    """  #     """
    def to_debug(self, format):  #     def to_debug(self,format):
        format = parse_def(format)  #         format = parse_def(format)
        return AutoData.debug_conversion(self.format, format, self.value)  #         return AutoData.debug_conversion(self.format,format,self.value)

    def __init__(self, value, format, solver):  #     def __init__(self,value,format,solver):
        self.value = value  #         self.value = value
        self.format = format  #         self.format = format
        self.solver_id = id(solver)  # do not hold solver, but hold solver's id. in order to make this picklable.  #         self.solver_id = id(solver) # do not hold solver, but hold solver's id. in order to make this picklable.
        if self.solver_id not in SOLVERS:  #         if self.solver_id not in SOLVERS:
            SOLVERS[self.solver_id] = solver  #             SOLVERS[self.solver_id] = solver

    @property  #     @property
    def solver(self):  #     def solver(self):
        return SOLVERS[self.solver_id]  #         return SOLVERS[self.solver_id]

    def converter(self, format=None, **kwargs):  #     def converter(self,format=None,**kwargs):
        if format is not None:  #         if format is not None:
            return self.solver.solver.search_direct(self.format, format)  #             return self.solver.solver.search_direct(self.format,format)
        else:  #         else:
            return self.solver.solver.search(self.format, tag_matcher(**kwargs))  #             return self.solver.solver.search(self.format,tag_matcher(**kwargs))

    def convert(self, format=None, **kwargs):  #     def convert(self,format=None,**kwargs):
        conversion = self.converter(format, **kwargs)  #         conversion = self.converter(format,**kwargs)
        if conversion.edges:  #         if conversion.edges:
            return AutoData(conversion(self.value), conversion.edges[-1].dst, self.solver)  #             return AutoData(conversion(self.value),conversion.edges[-1].dst,self.solver)
        else:  #         else:
            return self  #             return self



    def to(self, format=None, **kwargs):  # I want 'to' to accept format string too  #     def to(self,format=None,**kwargs): # I want 'to' to accept format string too
# if format is given, use direct matching.
# else use tag matching
# format can be of any type, but you need to have a conversion rule to tag_dict, otherwise you won't get any result
# so, ask user to provide any state and state->tag_dict rule.
# That's it.
        converted = self.convert(format=format, **kwargs)  #         converted = self.convert(format=format,**kwargs)
        return converted.value  #         return converted.value

    def map(self, f, new_format=None):  #     def map(self,f,new_format=None):
        if new_format is not None:  #         if new_format is not None:
            format = new_format  #             format = new_format
        else:  #         else:
            format = self.format  #             format = self.format
        return AutoData(f(self.value), format, self.solver)  #         return AutoData(f(self.value),format,self.solver)

    def map_in(self, start_format, f, new_format=None):  #     def map_in(self,start_format,f,new_format=None):
        return AutoData(f(self.to(start_format)), (lambda _coconut_none_coalesce_item: self.format if _coconut_none_coalesce_item is None else _coconut_none_coalesce_item)(new_format), self.solver)  #         return AutoData(f(self.to(start_format)),new_format??self.format,self.solver)

    def neighbors(self):  #     def neighbors(self):
        return self.solver.solver.neighbors(self.format)  #         return self.solver.solver.neighbors(self.format)

    def to_widget(self):  #     def to_widget(self):
        return self.to("widget")  #         return self.to("widget")

    def _repr_html_(self):  #     def _repr_html_(self):
        (display)(self.format)  #         self.format |> display
        (display)(self.to("widget"))  #         self.to("widget") |> display
    def __repr__(self):  #     def __repr__(self):
        return "<AutoData {_coconut_format_0}>".format(_coconut_format_0=(self.format))  #         return f"<AutoData {self.format}>"

    def _repr_png_(self):  #     def _repr_png_(self):
        try:  #         try:
            return self.to(type="image")._repr_png_()  #             return self.to(type="image")._repr_png_()
        except Exception as e:  #         except Exception as e:
            logger.warning("cannot convert data to an image:{_coconut_format_0}".format(_coconut_format_0=(self.format)))  #             logger.warning(f"cannot convert data to an image:{self.format}")
            return None  #             return None

    def cast(self, format):  #     def cast(self,format):
        return AutoData(self.value, format, self.solver)  #         return AutoData(self.value,format,self.solver)

    def show(self):  #     def show(self):
        from matplotlib.pyplot import imshow  #         from matplotlib.pyplot import imshow,show
        from matplotlib.pyplot import show  #         from matplotlib.pyplot import imshow,show
        imshow(self.to("numpy_rgb"))  #         imshow(self.to("numpy_rgb"))
        show()  #         show()


#    def __getstate__(self):


try:  # def AutoData.call(self,name,*args,**kwargs):
    _coconut_dotted_func_name_store_0 = call  # def AutoData.call(self,name,*args,**kwargs):
except _coconut.NameError:  # def AutoData.call(self,name,*args,**kwargs):
    _coconut_dotted_func_name_store_0 = None  # def AutoData.call(self,name,*args,**kwargs):
def call(self, name, *args, **kwargs):  # def AutoData.call(self,name,*args,**kwargs):
    return self.to(name)(*args, **kwargs)  #     return self.to(name)(*args,**kwargs)

AutoData.call = call  #     return self.to(name)(*args,**kwargs)
call = _coconut_dotted_func_name_store_0  #     return self.to(name)(*args,**kwargs)
