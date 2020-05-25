#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xb843031f

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
"""
What I want to achieve is to generate a class.
I want to generate a AutoImage class form AutoData
well the solution is to just use partially applied constructor. that's it.
"""  # """
def identity(a):  # def identity(a):
    return a  #     return a

class _CastLambda:  # class _CastLambda:
    def __init__(self, rule, name, swap):  #     def __init__(self,rule,name,swap):
        self.rule = rule  #         self.rule = rule
        self.name = name  #         self.name = name
        self.swap = swap  #         self.swap = swap
    def __call__(self, state):  #     def __call__(self,state):
        new_states = self.rule(state)  #         new_states = self.rule(state)
        if new_states is not None:  #         if new_states is not None:
            if self.name is None:  #             if self.name is None:
                cast_name = "{_coconut_format_0}".format(_coconut_format_0=(rule.__name__))  #                 cast_name = f"{rule.__name__}"
            else:  #             else:
                cast_name = self.name  #                 cast_name=self.name
            if self.swap:  #             if self.swap:
                return [(identity, new_state, cast_name, 1) for new_state in new_states]  #                 return [(identity,new_state,cast_name,1) for new_state in new_states]
            else:  #             else:
                return [(identity, new_state, 1, cast_name) for new_state in new_states]  #                 return [(identity,new_state,1,cast_name) for new_state in new_states]
        else:  #         else:
            return None  #             return None
class _ConversionLambda:  # class _ConversionLambda:
    def __init__(self, rule):  #     def __init__(self,rule):
        self.rule = rule  #         self.rule = rule
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
                result.append((converter, new_state, 1, converter.__name__))  #                     result.append((converter,new_state,1,converter.__name__))
            if not _coconut_case_check_0:  #                 match (converter,new_state,name):
                if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 3):  #                 match (converter,new_state,name):
                    converter = _coconut_match_to[0]  #                 match (converter,new_state,name):
                    new_state = _coconut_match_to[1]  #                 match (converter,new_state,name):
                    name = _coconut_match_to[2]  #                 match (converter,new_state,name):
                    _coconut_case_check_0 = True  #                 match (converter,new_state,name):
                if _coconut_case_check_0:  #                 match (converter,new_state,name):
                    result.append((converter, new_state, 1, name))  #                     result.append((converter,new_state,1,name))
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
                    raise RuntimeError("rule:{_coconut_format_0} returned invalid edge:{_coconut_format_1}.".format(_coconut_format_0=(rule), _coconut_format_1=(edge)))  #                     raise RuntimeError(f"rule:{rule} returned invalid edge:{edge}.")
        return result  #         return result

class AutoSolver:  # class AutoSolver:
    """
    TODO stop using local lambda in order to make this class picklable
    Factory for an AutoData class
    """  #     """
    def __init__(self, rules):  #     def __init__(self,rules):
        self.initial_rules = rules  #         self.initial_rules = rules
        self.solver = AStarSolver(rules=self.initial_rules.copy())  #         self.solver = AStarSolver(rules = self.initial_rules.copy())

    @staticmethod  #     @staticmethod
    def create_cast_rule(rule, name=None, _swap=False):  #     def create_cast_rule(rule,name=None,_swap=False):
        """
        rule: State->List[State] # should return list of possible casts without data conversion.
        """  #         """
        return _CastLambda(rule, name, _swap)  #         return _CastLambda(rule,name,_swap)

    def add_cast(self, rule, name=None):  #     def add_cast(self,rule,name=None):
        """
        rule: State->List[State] # should return list of possible casts without data conversion.
        """  #         """

        self.add_conversion(AutoSolver.create_cast_rule(rule, name=name, _swap=True))  #         self.add_conversion(AutoSolver.create_cast_rule(rule,name=name,_swap=True))

    @staticmethod  #     @staticmethod
    def create_conversion_rule(rule):  #     def create_conversion_rule(rule):
        return _ConversionLambda(rule)  #         return _ConversionLambda(rule)

    def add_conversion(self, rule):  #     def add_conversion(self,rule):
        """
        rule: State->List[(converter,new_state,name(optional),cost(optional))]
        """  #         """
        self.solver.add_rule(AutoSolver.create_conversion_rule(rule))  #         self.solver.add_rule(AutoSolver.create_conversion_rule(rule))

    def reset_solver(self,):  #     def reset_solver(self,):
        self.solver = AStarSolver(rules=self.initial_rules.copy())  #         self.solver = AStarSolver(rules = self.initial_rules.copy())

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
        return AutoData(value, format, self.solver)  #         return AutoData(value,format,self.solver)





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
            return self.solver.search_direct(self.format, format)  #             return self.solver.search_direct(self.format,format)
        else:  #         else:
            return self.solver.search(self.format, tag_matcher(**kwargs))  #             return self.solver.search(self.format,tag_matcher(**kwargs))

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

    def map(self, f):  #     def map(self,f):
        return AutoData(f(self.value), self.format, self.solver)  #         return AutoData(f(self.value),self.format,self.solver)

    def neighbors(self):  #     def neighbors(self):
        return self.solver.neighbors(self.format)  #         return self.solver.neighbors(self.format)

    def to_widget(self):  #     def to_widget(self):
        return self.to("widget")  #         return self.to("widget")

    def _repr_html_(self):  #     def _repr_html_(self):
        (display)(self.format)  #         self.format |> display
        (display)(self.to("widget"))  #         self.to("widget") |> display

#    def __getstate__(self):
