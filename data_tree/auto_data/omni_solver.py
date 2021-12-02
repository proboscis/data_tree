import abc
import heapq
from dataclasses import dataclass, field
from typing import Callable, List, Generic, Any, Optional, TypeVar

from expression import Result, Ok, Error
from returns.result import Success
from snoop import snoop

from coconut.astar import _HeapContainer

T = TypeVar("T")


@dataclass
class RuleResult:
    dependencies: list
    provider: Callable
    cost: int
    name: str


class ParameterizedRule(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, query) -> List[RuleResult]:
        pass


@dataclass
class ParameterizedSolver:
    rules: List[ParameterizedRule]

    def solve(self, query) -> Result:
        # dfs solve
        for r in self.rules:
            rrs = r(query)
            for rr in rrs:
                deps = [self.solve(d) for d in rr.dependencies]
                if all([isinstance(d,Success) for d in deps]):
                    return Success(rr.provider(*[d.value for d in deps]))
        return Error(f"could not solve dependency query:{query}")


@dataclass
class HeapQueue(Generic[T]):
    data: List[_HeapContainer] = field(default_factory=list)

    def push(self, score, data):
        heapq.heappush(self.data, _HeapContainer(score, data))

    def __bool__(self):
        return len(self.data) != 0

    def pop(self) -> T:
        return heapq.heappop(self.data).data


@dataclass
class State:
    queries: list
    to_value: Any

    def __hash__(self):
        return hash(tuple(self.queries))


class NoSolutionError(RuntimeError): pass


@dataclass
class Resolution:
    query: Any
    substitution: Any
    resolution: Optional["Resolution"]


@dataclass
class ParameterizedAstarSolver:
    @snoop
    def solve(self, query, rule: Callable[[Any], List[RuleResult]], heuristics: Callable[[Any], int]) -> State:
        # we need a queue of queries in question
        scores = dict()
        start = State([query], None)
        scores[start] = heuristics(start)
        to_visit: HeapQueue[State] = HeapQueue()
        to_visit.push(scores[start], start)
        while to_visit:
            pos = to_visit.pop()
            if len(pos.queries) == 0:
                return pos
                # we are done if the state's query became empty.
            # now look for next states
            current_cost = scores[pos]
            for q in pos.queries:
                for r in rule(q):
                    # calc next state
                    qs = pos.queries.copy()
                    qs.remove(q)
                    qs += r.dependencies
                    ns = State(qs,
                               Resolution(q, r, pos.to_value))  # stack required information for now to be used later
                    cost = current_cost + r.cost + heuristics(ns)
                    if ns not in scores or scores[ns] > cost:
                        scores[ns] = cost
                        to_visit.push(cost, ns)
        raise NoSolutionError(f"no solution found that satisfy query:{query}")


def eval_solution(state: State):
    """
    retrieves the originating query's result
    :param state:
    :return: anything the query requested for.
    """
    subs = dict()
    res: Resolution = state.to_value

    # get values out of provider using substitutions
    def solve_rr(q, r: RuleResult):
        deps = [subs[d] for d in r.dependencies]
        value = r.provider(*deps)
        subs[q] = value

    while True:
        if isinstance(res.substitution, RuleResult):
            solve_rr(res.query, res.substitution)
        if res.resolution is None:
            return subs[res.query]
        res = res.resolution
    raise RuntimeError(f"could not evaluate the solution for query result.{state}")

@dataclass
class Conversion:
    src:Any
    dst:Any

    def __hash__(self):
        return hash(hash(self.src)*hash(self.dst))
