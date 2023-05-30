#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xef144ef6

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

from ply import lex  # from ply import lex
from ply import yacc  # import ply.yacc as yacc

tokens = ('LPAREN', 'RPAREN', 'LSB', 'RSB', 'COMMA', 'NAME', 'STR')  #'  # ","  # right square bracket ']'  # left square bracket '['  # tokens = (

t_ignore = ' \t'  # t_ignore = ' \t'
t_STR = r"\'(\w*)\'"  # t_STR = r"\'(\w*)\'"
t_NAME = r'([a-zA-Z_])\w*'  # t_NAME = r'([a-zA-Z_])\w*'
t_LPAREN = r'\('  # t_LPAREN = r'\('
t_RPAREN = r'\)'  # t_RPAREN = r'\)'
t_LSB = r'\['  # t_LSB = r'\['
t_RSB = r'\]'  # t_RSB = r'\]'
t_COMMA = r'\,'  # t_COMMA = r'\,'

def t_error(t):  # def t_error(t):
    print("Invalid Token:", t.value[0])  #     print("Invalid Token:", t.value[0])
    t.lexer.skip(1)  #     t.lexer.skip(1)


lexer = lex.lex()  # lexer = lex.lex()

precedence = ()  # precedence = (

class ATuple(_coconut.collections.namedtuple("ATuple", ('items',))):  # data ATuple(*items)
    __slots__ = ()  # data ATuple(*items)
    _coconut_is_data = True  # data ATuple(*items)
    __match_args__ = ()  # data ATuple(*items)
    def __add__(self, other): return _coconut.NotImplemented  # data ATuple(*items)
    def __mul__(self, other): return _coconut.NotImplemented  # data ATuple(*items)
    def __rmul__(self, other): return _coconut.NotImplemented  # data ATuple(*items)
    __ne__ = _coconut.object.__ne__  # data ATuple(*items)
    def __eq__(self, other):  # data ATuple(*items)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data ATuple(*items)
    def __hash__(self):  # data ATuple(*items)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data ATuple(*items)
    def __new__(_coconut_cls, *items):  # data ATuple(*items)
        return _coconut.tuple.__new__(_coconut_cls, items)  # data ATuple(*items)
    @_coconut.classmethod  # data ATuple(*items)
    def _make(cls, iterable, *, new=_coconut.tuple.__new__, len=None):  # data ATuple(*items)
        return new(cls, iterable)  # data ATuple(*items)
    def _asdict(self):  # data ATuple(*items)
        return _coconut.OrderedDict([("items", self[:])])  # data ATuple(*items)
    def __repr__(self):  # data ATuple(*items)
        return "ATuple(*items=%r)" % (self[:],)  # data ATuple(*items)
    def _replace(_self, **kwds):  # data ATuple(*items)
        result = _self._make(kwds.pop("items", _self))  # data ATuple(*items)
        if kwds:  # data ATuple(*items)
            raise _coconut.ValueError("Got unexpected field names: " + _coconut.repr(kwds.keys()))  # data ATuple(*items)
        return result  # data ATuple(*items)
    @_coconut.property  # data ATuple(*items)
    def items(self):  # data ATuple(*items)
        return self[:]  # data ATuple(*items)

_coconut_call_set_names(ATuple)  # data AList(*items)
class AList(_coconut.collections.namedtuple("AList", ('items',))):  # data AList(*items)
    __slots__ = ()  # data AList(*items)
    _coconut_is_data = True  # data AList(*items)
    __match_args__ = ()  # data AList(*items)
    def __add__(self, other): return _coconut.NotImplemented  # data AList(*items)
    def __mul__(self, other): return _coconut.NotImplemented  # data AList(*items)
    def __rmul__(self, other): return _coconut.NotImplemented  # data AList(*items)
    __ne__ = _coconut.object.__ne__  # data AList(*items)
    def __eq__(self, other):  # data AList(*items)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data AList(*items)
    def __hash__(self):  # data AList(*items)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data AList(*items)
    def __new__(_coconut_cls, *items):  # data AList(*items)
        return _coconut.tuple.__new__(_coconut_cls, items)  # data AList(*items)
    @_coconut.classmethod  # data AList(*items)
    def _make(cls, iterable, *, new=_coconut.tuple.__new__, len=None):  # data AList(*items)
        return new(cls, iterable)  # data AList(*items)
    def _asdict(self):  # data AList(*items)
        return _coconut.OrderedDict([("items", self[:])])  # data AList(*items)
    def __repr__(self):  # data AList(*items)
        return "AList(*items=%r)" % (self[:],)  # data AList(*items)
    def _replace(_self, **kwds):  # data AList(*items)
        result = _self._make(kwds.pop("items", _self))  # data AList(*items)
        if kwds:  # data AList(*items)
            raise _coconut.ValueError("Got unexpected field names: " + _coconut.repr(kwds.keys()))  # data AList(*items)
        return result  # data AList(*items)
    @_coconut.property  # data AList(*items)
    def items(self):  # data AList(*items)
        return self[:]  # data AList(*items)


_coconut_call_set_names(AList)  # def p_str(p):
def p_str(p):  # def p_str(p):
    """expr : STR"""  #     """expr : STR"""
    p[0] = p[1]  #     p[0] = p[1]

def p_tuple(p):  # def p_tuple(p):
    """expr : LPAREN exprs RPAREN"""  #     """expr : LPAREN exprs RPAREN"""
    p[0] = ATuple(*p[2])  #     p[0] = ATuple(*p[2])


def p_list(p):  # def p_list(p):
    """expr : LSB exprs RSB"""  #     """expr : LSB exprs RSB"""
    p[0] = AList(*p[2])  #     p[0] = AList(*p[2])



def p_exprs(p):  # def p_exprs(p):
    """exprs : expr COMMA exprs
             | expr COMMA
             | expr
    """  #     """
    if len(p) == 2:  #     if len(p) == 2:
        p[0] = [p[1],]  #         p[0] = [p[1]]
    elif len(p) == 3:  #     elif len(p) == 3:
        p[0] = [p[1],]  #         p[0] = [p[1]]
    else:  #     else:
        p[0] = [p[1],] + p[3]  #         p[0] = [p[1]] + p[3]


def p_name(p):  # def p_name(p):
    "expr : NAME"  #     "expr : NAME"
    p[0] = p[1]  #     p[0] = p[1]


def p_error(p):  # def p_error(p):
    print("Syntax error in input!")  #     print("Syntax error in input!")


parser = yacc.yacc()  # parser = yacc.yacc()

res = parser.parse("('this is text')")  # the input  # res = parser.parse("('this is text')")  # the input

_coconut_case_match_to_0 = res  # case res:
_coconut_case_match_check_0 = False  # case res:
if (_coconut.isinstance(_coconut_case_match_to_0, _coconut.abc.Sequence)) and (_coconut.len(_coconut_case_match_to_0) == 2) and (_coconut_case_match_to_0[0] == "a") and (_coconut_case_match_to_0[1] == "b"):  # case res:
    _coconut_case_match_check_0 = True  # case res:
if _coconut_case_match_check_0:  # case res:
    print("hmm, is it tuple?")  #         print("hmm, is it tuple?")
print(res)  # print(res)
