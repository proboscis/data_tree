#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x6aaa864

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

import heapq  # import heapq
from loguru import logger  # from loguru import logger
from pprint import pformat  # from pprint import pformat
from data_tree.coconut.monad import try_monad  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
from data_tree.coconut.monad import Try  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
from data_tree.coconut.monad import Success  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
from data_tree.coconut.monad import Failure  # from data_tree.coconut.monad import try_monad,Try,Success,Failure
import dill  # for pickling inner lambda...  # import dill # for pickling inner lambda...
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
class ConversionError(Exception): pass  # class ConversionError(Exception)
class Conversion(_coconut.collections.namedtuple("Conversion", "edges")):  # data Conversion(edges):
    __slots__ = ()  # data Conversion(edges):
    __ne__ = _coconut.object.__ne__  # data Conversion(edges):
    def __eq__(self, other):  # data Conversion(edges):
        return self.__class__ is other.__class__ and _coconut.tuple.__eq__(self, other)  # data Conversion(edges):
    def __hash__(self):  # data Conversion(edges):
        return _coconut.tuple.__hash__(self) ^ hash(self.__class__)  # data Conversion(edges):
    def __call__(self, x):  #     def __call__(self,x):
        for e in self.edges:  #         for e in self.edges:
            try:  #             try:
                x = e.f(x)  #                 x = e.f(x)
            except Exception as ex:  #             except Exception as ex:
                raise ConversionError("exception in path:{_coconut_format_0} \n paths:{_coconut_format_1}".format(_coconut_format_0=(e.name), _coconut_format_1=([e.name for e in self.edges]))) from ex  #                 raise ConversionError(f"exception in path:{e.name} \n paths:{[e.name for e in self.edges]}") from ex
        return x  #         return x

    def __repr__(self):  #     def __repr__(self):
        if self.edges:  #         if self.edges:
            start = self.edges[0].src  #             start = self.edges[0].src
            end = self.edges[-1].dst  #             end = self.edges[-1].dst
        else:  #         else:
            start = "None"  #             start = "None"
            end = "None"  #             end = "None"
        info = dict(name="Conversion", start=start, end=end, path=[e.name for e in self.edges])  #         info = dict(
        return pformat(info)  #         return pformat(info)
def new_conversion(edges):  # def new_conversion(edges):
    return Conversion(edges)  #     return Conversion(edges)
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
    visited = 0  #     visited = 0
    while to_visit:  #     while to_visit:
        hc = heapq.heappop(to_visit)  #         hc = heapq.heappop(to_visit)
        score = hc.score  #         score = hc.score
        (pos, trace) = hc.data  #         (pos,trace) = hc.data
        visited += 1  #         visited += 1
#print(f"visit:{pos}")
#print(f"{((trace[-1].a,trace[-1].name) if trace else 'no trace')}")
#print(f"visit:{trace[-1] if trace else 'no trace'}")
        if len(trace) >= max_depth:  # terminate search on max_depth  #         if len(trace) >= max_depth: # terminate search on max_depth
            continue  #             continue
        if matcher(pos):  # reached a goal  #         if matcher(pos): # reached a goal
            logger.debug("found after {_coconut_format_0} visits.".format(_coconut_format_0=(visited)))  #             logger.debug(f"found after {visited} visits.")
            logger.debug("search result:\n{_coconut_format_0}".format(_coconut_format_0=(new_conversion(trace))))  #             logger.debug(f"search result:\n{new_conversion(trace)}")
            return trace  #             return trace
        for mapper, next_node, cost, name in neighbors(pos):  #         for mapper,next_node,cost,name in neighbors(pos):
            assert isinstance(cost, int), "cost is not a number. cost:{_coconut_format_0},name:{_coconut_format_1},pos:{_coconut_format_2}".format(_coconut_format_0=(cost), _coconut_format_1=(name), _coconut_format_2=(pos))  #             assert isinstance(cost,int),f"cost is not a number. cost:{cost},name:{name},pos:{pos}"
            new_trace = trace + [Edge(pos, next_node, mapper, cost, name)]  #             new_trace = trace + [Edge(pos,next_node,mapper,cost,name)]
#new_score = scores[pos] + cost + heuristics(next_node)
            try:  #             try:
                new_score = scores[pos] + cost + heuristics(next_node)  #                 new_score = scores[pos] + cost + heuristics(next_node)
            except Exception as e:  #             except Exception as e:
                logger.error("pos:{_coconut_format_0},cost:{_coconut_format_1},next_node:{_coconut_format_2}".format(_coconut_format_0=(pos), _coconut_format_1=(cost), _coconut_format_2=(next_node)))  #                 logger.error(f"pos:{pos},cost:{cost},next_node:{next_node}")
                raise e  #                 raise e
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
    visited = 0  #     visited = 0
    while to_visit:  #     while to_visit:
        hc = heapq.heappop(to_visit)  #         hc = heapq.heappop(to_visit)
        score = hc.score  #         score = hc.score
        (pos, trace) = hc.data  #         (pos,trace) = hc.data
#logger.debug(f"visiting:{pos}")
        visited += 1  #         visited += 1
#print(f"visit:{pos}")
#print(f"{((trace[-1].a,trace[-1].name) if trace else 'no trace')}")
#print(f"visit:{trace[-1] if trace else 'no trace'}")
        if len(trace) >= max_depth:  # terminate search on max_depth  #         if len(trace) >= max_depth: # terminate search on max_depth
            continue  #             continue
        if pos == end:  # reached a goal  #         if pos == end: # reached a goal
            logger.debug("found after {_coconut_format_0} visits.".format(_coconut_format_0=(visited)))  #             logger.debug(f"found after {visited} visits.")
            logger.debug("search result:\n{_coconut_format_0}".format(_coconut_format_0=(new_conversion(trace))))  #             logger.debug(f"search result:\n{new_conversion(trace)}")
            return trace  #             return trace
        for mapper, next_node, cost, name in neighbors(pos):  #         for mapper,next_node,cost,name in neighbors(pos):
            assert isinstance(cost, int), "cost is not a number:{_coconut_format_0}".format(_coconut_format_0=(pos))  #             assert isinstance(cost,int),f"cost is not a number:{pos}"
            new_trace = trace + [Edge(pos, next_node, mapper, cost, name)]  #             new_trace = trace + [Edge(pos,next_node,mapper,cost,name)]
            try:  #             try:
                new_score = scores[pos] + cost + heuristics(next_node)  #                 new_score = scores[pos] + cost + heuristics(next_node)
            except Exception as e:  #             except Exception as e:
                logger.error("pos:{_coconut_format_0},cost:{_coconut_format_1},next_node:{_coconut_format_2}".format(_coconut_format_0=(pos), _coconut_format_1=(cost), _coconut_format_2=(next_node)))  #                 logger.error(f"pos:{pos},cost:{cost},next_node:{next_node}")
                raise e  #                 raise e
            if next_node in scores and scores[next_node] <= new_score:  #             if next_node in scores and scores[next_node] <= new_score:
                continue  #                 continue
            else:  #             else:
                scores[next_node] = new_score  #                 scores[next_node] = new_score
                heapq.heappush(to_visit, _HeapContainer(new_score, (next_node, new_trace)))  #                 heapq.heappush(to_visit,_HeapContainer(new_score,(next_node,new_trace)))

    raise NoRouteException("no route found from {_coconut_format_0} matching {_coconut_format_1}. searched {_coconut_format_2} nodes.".format(_coconut_format_0=(start), _coconut_format_1=(end), _coconut_format_2=(visited)))  #     raise NoRouteException(f"no route found from {start} matching {end}. searched {visited} nodes.")

astar = (try_monad)((_coconut_base_compose(_astar, (new_conversion, 0))))  # astar = (_astar ..> new_conversion) |> try_monad
astar_direct = (try_monad)((_coconut_base_compose(_astar_direct, (new_conversion, 0))))  # astar_direct = (_astar_direct ..> new_conversion) |> try_monad

def _zero_heuristics(x):  # def _zero_heuristics(x):
    return 0  #     return 0

MAX_MEMO = 2**(20)  # MAX_MEMO = 2**(20)

class AStarSolver:  # class AStarSolver:
    """
    to make this picklable, you have to have these caches on global variable.
    use getstate and setstate
    actually, having to pickle this solver every time you pass auto_data is not feasible.
    so, lets have auto_data to hold solver in global variable and never pickle AStarSolver!.
    so forget about lru_cache pickling issues.
    """  #     """
    def __init__(self, rules=None):  #     def __init__(self,rules=None):
        """
        rules: List[Rule]
        Rule: (state)->List[(converter:(data)->data,new_state,cost,conversion_name)]
        """  #         """
        from lru import LRU  #         from lru import LRU
        self.rules = rules if rules is not None else []  #         self.rules = rules if rules is not None else []
        self.heuristics = _zero_heuristics  #         self.heuristics = _zero_heuristics
        self.neighbors_memo = LRU(MAX_MEMO)  # you cannot pickle a lru.LRU object, thus you cannot pickle this class for multiprocessing.  #         self.neighbors_memo = LRU(MAX_MEMO) # you cannot pickle a lru.LRU object, thus you cannot pickle this class for multiprocessing.
        self.search_memo = LRU(MAX_MEMO)  #         self.search_memo = LRU(MAX_MEMO)
        self.direct_search_memo = LRU(MAX_MEMO)  #         self.direct_search_memo = LRU(MAX_MEMO)

    def neighbors(self, node):  #     def neighbors(self,node):
        if node in self.neighbors_memo:  #         if node in self.neighbors_memo:
            return self.neighbors_memo[node]  #             return self.neighbors_memo[node]

        res = []  #         res = []
        for rule in self.rules:  #         for rule in self.rules:
            edges = rule(node)  #             edges = rule(node)
            if edges is not None:  #             if edges is not None:
                res += edges  #                  res += edges
        self.neighbors_memo[node] = res  #         self.neighbors_memo[node] = res
        return res  #         return res

    def invalidate_cache(self):  #     def invalidate_cache(self):
        self.neighbors_memo.clear()  #         self.neighbors_memo.clear()
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
            res = self.search_memo[q]  #             res = self.search_memo[q]
        else:  #         else:
            logger.debug("searching from {_coconut_format_0} for matching {_coconut_format_1}".format(_coconut_format_0=(start), _coconut_format_1=(matcher)))  #             logger.debug(f"searching from {start} for matching {matcher}")
            res = astar(start=start, matcher=matcher, neighbors=self.neighbors, heuristics=self.heuristics)  #             res = astar(
            self.search_memo[q] = res  #             self.search_memo[q] = res

        _coconut_match_to = res  #         case res:
        _coconut_case_check_0 = False  #         case res:
        if (_coconut.isinstance(_coconut_match_to, Success)) and (_coconut.len(_coconut_match_to) == 1):  #         case res:
            res = _coconut_match_to[0]  #         case res:
            _coconut_case_check_0 = True  #         case res:
        if _coconut_case_check_0:  #         case res:
            return res  #                 return res
        if not _coconut_case_check_0:  #             match Failure(e,trc):
            if (_coconut.isinstance(_coconut_match_to, Failure)) and (_coconut.len(_coconut_match_to) == 2):  #             match Failure(e,trc):
                e = _coconut_match_to[0]  #             match Failure(e,trc):
                trc = _coconut_match_to[1]  #             match Failure(e,trc):
                _coconut_case_check_0 = True  #             match Failure(e,trc):
            if _coconut_case_check_0:  #             match Failure(e,trc):
                raise e  #                 raise e

    def search_direct(self, start, end):  #     def search_direct(self,start,end):
        q = (start, end)  #         q = (start,end)
        if q in self.direct_search_memo:  #         if q in self.direct_search_memo:
            res = self.direct_search_memo[q]  #             res = self.direct_search_memo[q]
        else:  #         else:
            logger.debug("searching {_coconut_format_0} to {_coconut_format_1}".format(_coconut_format_0=(start), _coconut_format_1=(end)))  #             logger.debug(f"searching {start} to {end}")
            res = astar_direct(start=start, end=end, neighbors=self.neighbors, heuristics=self.heuristics)  #             res = astar_direct(
            self.direct_search_memo[q] = res  #             self.direct_search_memo[q] = res
        _coconut_match_to = res  #         case res:
        _coconut_case_check_1 = False  #         case res:
        if (_coconut.isinstance(_coconut_match_to, Success)) and (_coconut.len(_coconut_match_to) == 1):  #         case res:
            res = _coconut_match_to[0]  #         case res:
            _coconut_case_check_1 = True  #         case res:
        if _coconut_case_check_1:  #         case res:
            return res  #                 return res
        if not _coconut_case_check_1:  #             match Failure(e,trc):
            if (_coconut.isinstance(_coconut_match_to, Failure)) and (_coconut.len(_coconut_match_to) == 2):  #             match Failure(e,trc):
                e = _coconut_match_to[0]  #             match Failure(e,trc):
                trc = _coconut_match_to[1]  #             match Failure(e,trc):
                _coconut_case_check_1 = True  #             match Failure(e,trc):
            if _coconut_case_check_1:  #             match Failure(e,trc):
                raise e  #                 raise e

    def search_direct_any(self, start, ends):  #     def search_direct_any(self,start,ends):
        for cand in ends:  #         for cand in ends:
            try:  #             try:
                res = self.search_direct(start, cand)  #                 res = self.search_direct(start,cand)
                return res  #                 return res
            except Exception as e:  #             except Exception as e:
                pass  #                 pass
        raise NoRouteException("no route found from {_coconut_format_0} to any of {_coconut_format_1}".format(_coconut_format_0=(start), _coconut_format_1=(ends)))  #         raise NoRouteException(f"no route found from {start} to any of {ends}")