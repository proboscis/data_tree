#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x368f8fae

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

from ply import lex  # from ply import lex
from ply import yacc  # import ply.yacc as yacc

tokens = ('LPAREN', 'RPAREN', 'LSB', 'RSB', 'COMMA', 'NAME', 'STR',)  # tokens = (

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

class ATuple(_coconut.collections.namedtuple("ATuple", "items")):  # data ATuple(*items)
    __slots__ = ()  # data ATuple(*items)
    __ne__ = _coconut.object.__ne__  # data ATuple(*items)
    def __eq__(self, other):  # data ATuple(*items)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data ATuple(*items)
    def __hash__(self):  # data ATuple(*items)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data ATuple(*items)
    def __new__(_cls, *items):  # data ATuple(*items)
        return _coconut.tuple.__new__(_cls, items)  # data ATuple(*items)
    @_coconut.classmethod  # data ATuple(*items)
    def _make(cls, iterable, *, new=_coconut.tuple.__new__, len=None):  # data ATuple(*items)
        return new(cls, iterable)  # data ATuple(*items)
    def _asdict(self):  # data ATuple(*items)
        return _coconut.OrderedDict([("items", self[:])])  # data ATuple(*items)
    def __repr__(self):  # data ATuple(*items)
        return "ATuple(*items=%r)" % (self[:],)  # data ATuple(*items)
    def _replace(_self, **kwds):  # data ATuple(*items)
        result = self._make(kwds.pop("items", _self))  # data ATuple(*items)
        if kwds:  # data ATuple(*items)
            raise _coconut.ValueError("Got unexpected field names: " + _coconut.repr(kwds.keys()))  # data ATuple(*items)
        return result  # data ATuple(*items)
    @_coconut.property  # data ATuple(*items)
    def items(self):  # data ATuple(*items)
        return self[:]  # data ATuple(*items)

class AList(_coconut.collections.namedtuple("AList", "items")):  # data AList(*items)
    __slots__ = ()  # data AList(*items)
    __ne__ = _coconut.object.__ne__  # data AList(*items)
    def __eq__(self, other):  # data AList(*items)
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data AList(*items)
    def __hash__(self):  # data AList(*items)
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data AList(*items)
    def __new__(_cls, *items):  # data AList(*items)
        return _coconut.tuple.__new__(_cls, items)  # data AList(*items)
    @_coconut.classmethod  # data AList(*items)
    def _make(cls, iterable, *, new=_coconut.tuple.__new__, len=None):  # data AList(*items)
        return new(cls, iterable)  # data AList(*items)
    def _asdict(self):  # data AList(*items)
        return _coconut.OrderedDict([("items", self[:])])  # data AList(*items)
    def __repr__(self):  # data AList(*items)
        return "AList(*items=%r)" % (self[:],)  # data AList(*items)
    def _replace(_self, **kwds):  # data AList(*items)
        result = self._make(kwds.pop("items", _self))  # data AList(*items)
        if kwds:  # data AList(*items)
            raise _coconut.ValueError("Got unexpected field names: " + _coconut.repr(kwds.keys()))  # data AList(*items)
        return result  # data AList(*items)
    @_coconut.property  # data AList(*items)
    def items(self):  # data AList(*items)
        return self[:]  # data AList(*items)


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
        p[0] = [p[1]]  #         p[0] = [p[1]]
    elif len(p) == 3:  #     elif len(p) == 3:
        p[0] = [p[1]]  #         p[0] = [p[1]]
    else:  #     else:
        p[0] = [p[1]] + p[3]  #         p[0] = [p[1]] + p[3]

def p_name(p):  # def p_name(p):
    "expr : NAME"  #     "expr : NAME"
    p[0] = p[1]  #     p[0] = p[1]

def p_error(p):  # def p_error(p):
    print("Syntax error in input!")  #     print("Syntax error in input!")

parser = yacc.yacc()  # parser = yacc.yacc()

res = parser.parse("('this is text')")  # the input  # res = parser.parse("('this is text')")  # the input

_coconut_match_to = res  # case res:
_coconut_case_check_0 = False  # case res:
if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2) and (_coconut_match_to[0] == "a") and (_coconut_match_to[1] == "b"):  # case res:
    _coconut_case_check_0 = True  # case res:
if _coconut_case_check_0:  # case res:
    print("hmm, is it tuple?")  #         print("hmm, is it tuple?")
print(res)  # print(res)
