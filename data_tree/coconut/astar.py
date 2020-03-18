#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xdea1edc0

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

import heapq  # import heapq
from loguru import logger  # from loguru import logger
from lru import LRU  # from lru import LRU
class Edge(_coconut.collections.namedtuple("Edge", "src dst f cost name")):  # data Edge(src,dst,f,cost,name="unnamed")
    __slots__ = ()  # data Edge(src,dst,f,cost,name="unnamed")
    __ne__ = _coconut.object.__ne__  # data Edge(src,dst,f,cost,name="unnamed")
    def __eq__(self, other):  # data Edge(src,dst,f,cost,name="unnamed")
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Edge(src,dst,f,cost,name="unnamed")
    def __hash__(self):  # data Edge(src,dst,f,cost,name="unnamed")
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Edge(src,dst,f,cost,name="unnamed")
    def __new__(_cls, src, dst, f, cost, name="unnamed"):  # data Edge(src,dst,f,cost,name="unnamed")
        return _coconut.tuple.__new__(_cls, (src, dst, f, cost, name))  # data Edge(src,dst,f,cost,name="unnamed")

class NoRouteException(Exception): pass  # class NoRouteException(Exception)
class Conversion(_coconut.collections.namedtuple("Conversion", "f edges")):  # data Conversion(f,edges):
    __slots__ = ()  # data Conversion(f,edges):
    __ne__ = _coconut.object.__ne__  # data Conversion(f,edges):
    def __eq__(self, other):  # data Conversion(f,edges):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Conversion(f,edges):
    def __hash__(self):  # data Conversion(f,edges):
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Conversion(f,edges):
    def __call__(self, x):  #     def __call__(self,x):
        return self.f(x)  #         return self.f(x)
def new_conversion(edges):  # def new_conversion(edges):
    return Conversion(reduce(_coconut_forward_compose, [e.f for e in edges]), edges)  #     return Conversion(reduce((..>),[e.f for e in edges]),edges)
class _HeapContainer:  # class _HeapContainer:
    def __init__(self, score, data):  #     def __init__(self,score,data):
        self.score = score  #         self.score = score
        self.data = data  #         self.data = data
    def __lt__(self, other):  #     def __lt__(self,other):
        return self.score < other.score  #         return self.score < other.score
def _astar(start, matcher, neighbors, heuristics, max_depth=100):  # def _astar(
    """
    neighbors: node->[(mapper,next_node,cost,name)]
    """  #     """
    to_visit = []  #     to_visit = []
    scores = dict()  #     scores = dict()
    scores[start] = heuristics(start)  #     scores[start] = heuristics(start)
    heapq.heappush(to_visit, _HeapContainer(scores[start], (start, [])))  #     heapq.heappush(to_visit,

    while to_visit:  #     while to_visit:
        hc = heapq.heappop(to_visit)  #         hc = heapq.heappop(to_visit)
        score = hc.score  #         score = hc.score
        (pos, trace) = hc.data  #         (pos,trace) = hc.data
#print(f"visit:{pos}")
#print(f"{((trace[-1].a,trace[-1].name) if trace else 'no trace')}")
#print(f"visit:{trace[-1] if trace else 'no trace'}")
        if len(trace) >= max_depth:  # terminate search on max_depth  #         if len(trace) >= max_depth: # terminate search on max_depth
            continue  #             continue
        if matcher(pos):  # reached a goal  #         if matcher(pos): # reached a goal
            return trace  #             return trace
        for mapper, next_node, cost, name in neighbors(pos):  #         for mapper,next_node,cost,name in neighbors(pos):
            new_trace = trace + [Edge(pos, next_node, mapper, cost, name)]  #             new_trace = trace + [Edge(pos,next_node,mapper,cost,name)]
            new_score = scores[pos] + cost + heuristics(next_node)  #             new_score = scores[pos] + cost + heuristics(next_node)
            if next_node in scores and scores[next_node] <= new_score:  #             if next_node in scores and scores[next_node] <= new_score:
                continue  #                 continue
            else:  #             else:
                scores[next_node] = new_score  #                 scores[next_node] = new_score
                heapq.heappush(to_visit, _HeapContainer(new_score, (next_node, new_trace)))  #                 heapq.heappush(to_visit,_HeapContainer(new_score,(next_node,new_trace)))

    raise NoRouteException("no route found from {_coconut_format_0} matching {_coconut_format_1}".format(_coconut_format_0=(start), _coconut_format_1=(matcher)))  #     raise NoRouteException(f"no route found from {start} matching {matcher}")

def _astar_direct(start, end, neighbors, heuristics, max_depth=100):  # def _astar_direct(
    """
    neighbors: node->[(mapper,next_node,cost,name)]
    """  #     """
    to_visit = []  #     to_visit = []
    scores = dict()  #     scores = dict()
    scores[start] = heuristics(start)  #     scores[start] = heuristics(start)
    heapq.heappush(to_visit, _HeapContainer(scores[start], (start, [])))  #     heapq.heappush(to_visit,

    while to_visit:  #     while to_visit:
        hc = heapq.heappop(to_visit)  #         hc = heapq.heappop(to_visit)
        score = hc.score  #         score = hc.score
        (pos, trace) = hc.data  #         (pos,trace) = hc.data
#print(f"visit:{pos}")
#print(f"{((trace[-1].a,trace[-1].name) if trace else 'no trace')}")
#print(f"visit:{trace[-1] if trace else 'no trace'}")
        if len(trace) >= max_depth:  # terminate search on max_depth  #         if len(trace) >= max_depth: # terminate search on max_depth
            continue  #             continue
        if pos == end:  # reached a goal  #         if pos == end: # reached a goal
            return trace  #             return trace
        for mapper, next_node, cost, name in neighbors(pos):  #         for mapper,next_node,cost,name in neighbors(pos):
            new_trace = trace + [Edge(pos, next_node, mapper, cost, name)]  #             new_trace = trace + [Edge(pos,next_node,mapper,cost,name)]
            new_score = scores[pos] + cost + heuristics(next_node)  #             new_score = scores[pos] + cost + heuristics(next_node)
            if next_node in scores and scores[next_node] <= new_score:  #             if next_node in scores and scores[next_node] <= new_score:
                continue  #                 continue
            else:  #             else:
                scores[next_node] = new_score  #                 scores[next_node] = new_score
                heapq.heappush(to_visit, _HeapContainer(new_score, (next_node, new_trace)))  #                 heapq.heappush(to_visit,_HeapContainer(new_score,(next_node,new_trace)))

    raise NoRouteException("no route found from {_coconut_format_0} matching {_coconut_format_1}".format(_coconut_format_0=(start), _coconut_format_1=(end)))  #     raise NoRouteException(f"no route found from {start} matching {end}")

astar = _coconut_base_compose(_astar, (new_conversion, 0))  # astar = _astar ..> new_conversion
astar_direct = _coconut_base_compose(_astar_direct, (new_conversion, 0))  # astar_direct = _astar_direct ..> new_conversion
class AStarSolver:  # class AStarSolver:
    def __init__(self, rules=None, max_memo=1024):  #     def __init__(self,rules=None,max_memo=1024):
        self.max_memo = max_memo  #         self.max_memo = max_memo
        self.rules = rules if rules is not None else []  #         self.rules = rules if rules is not None else []
        self.heuristics = lambda x: 0  #         self.heuristics = x->0
        self.neighbors = (memoize(self.max_memo))((lambda node: self._neighbors(node)))  #         self.neighbors = (node->self._neighbors(node)) |> memoize(self.max_memo)
        self.search_memo = LRU(self.max_memo)  #         self.search_memo = LRU(self.max_memo)
        self.direct_search_memo = LRU(self.max_memo)  #         self.direct_search_memo = LRU(self.max_memo)

    def _neighbors(self, node):  #     def _neighbors(self,node):
        res = []  #         res = []
        for rule in self.rules:  #         for rule in self.rules:
            edges = rule(node)  #             edges = rule(node)
            if edges is not None:  #             if edges is not None:
                res += edges  #                  res += edges
        return res  #         return res

    def invalidate_cache(self):  #     def invalidate_cache(self):
        self.neighbors.cache_clear()  #         self.neighbors.cache_clear()
        self.search_memo.clear()  #         self.search_memo.clear()
        self.direct_search_memo.clear()  #         self.direct_search_memo.clear()

    def add_rule(self, f):  #     def add_rule(self,f):
        self.rules.append(f)  #         self.rules.append(f)
        self.invalidate_cache()  #         self.invalidate_cache()

    def search(self, start, matcher):  #     def search(self,start,matcher):
#problem is that you can't hash matcher
#let's use id of matcher for now.
        q = (start, id(matcher))  #         q = (start,id(matcher))
        if q in self.search_memo:  #         if q in self.search_memo:
            return self.search_memo[q]  #             return self.search_memo[q]
        else:  #         else:
            logger.debug("searching from {_coconut_format_0} for matching {_coconut_format_1}".format(_coconut_format_0=(start), _coconut_format_1=(matcher)))  #             logger.debug(f"searching from {start} for matching {matcher}")
            res = astar(start=start, matcher=matcher, neighbors=self.neighbors, heuristics=self.heuristics)  #             res = astar(
            self.search_memo[q] = res  #             self.search_memo[q] = res
            return res  #             return res

    def search_direct(self, start, end):  #     def search_direct(self,start,end):
        q = (start, end)  #         q = (start,end)
        if q in self.direct_search_memo:  #         if q in self.direct_search_memo:
            return self.direct_search_memo[q]  #             return self.direct_search_memo[q]
        else:  #         else:
            logger.debug("searching from {_coconut_format_0} for {_coconut_format_1}".format(_coconut_format_0=(start), _coconut_format_1=(end)))  #             logger.debug(f"searching from {start} for {end}")
            res = astar_direct(start=start, end=end, neighbors=self.neighbors, heuristics=self.heuristics)  #             res = astar_direct(
            self.direct_search_memo[q] = res  #             self.direct_search_memo[q] = res
            return res  #             return res
