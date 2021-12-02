import abc
import heapq
import itertools
from dataclasses import dataclass
from typing import Callable, Any, List

from loguru import logger
from tqdm import tqdm

from data_tree.auto_data.omni_solver import HeapQueue
from data_tree import DEFAULT_RULES
from data_tree.auto_data.rules import IAutoRuleBook, RuleEdge
from data_tree.coconut.astar import _HeapContainer, NoRouteException, Edge
from data_tree.di.util import Design


class IRuleWithSolver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, state, solve) -> List[RuleEdge]:
        pass


@dataclass
class RulesUsingSolver:
    rules: List[IRuleWithSolver]


@dataclass
class Neighbors:
    rulebook: IAutoRuleBook
    rulebook_with_solver: RulesUsingSolver

    def __call__(self, state, solve) -> List[RuleEdge]:
        nexts = list(itertools.chain(*[r(state) for r in self.rulebook.rules]))
        nexts += list(itertools.chain(*[r(state, solve) for r in self.rulebook_with_solver.rules]))
        return nexts

# I need the solver to be hashable.
# the previous implementation of AStarSolver is not good since I didnt have DI in mind.
# I feel my code being more reusable and modular.
# TODOS
# make a solver hashable
# implement intra dict conversion


@dataclass
class AstarSolver:
    heuristics: Callable
    neighbors: Callable[[Any, Callable[[Any, Any], List[Edge]]], List[RuleEdge]]
    # (Any,(Any,Any)=>List[Edge])=>List[ASEdge]
    max_depth: int
    silent: bool

    def solve(self, start, end):
        to_visit = HeapQueue()
        scores = dict()
        scores[start] = self.heuristics(start, end)
        to_visit.push(scores[start],(start,[]))
        visited = 0
        if not self.silent:
            bar = tqdm(desc=f"solving with astar_direct from {start} to {end}")
        last_bar_update = visited
        while to_visit:
            if visited - last_bar_update > 1000 and not self.silent:
                bar.update(visited - last_bar_update)
                last_bar_update = visited

            (pos, trace) = to_visit.pop()
            visited += 1
            if len(trace) >= self.max_depth:  # terminate search on max_depth
                continue
            if pos == end:  # reached a goal
                if not self.silent:
                    bar.close()
                return trace
            if visited % 10000 == 0 and not self.silent:
                msg = str(pos)[:50]
                bar.set_description(f"""pos:{msg:<50}""")
            normal_nodes = self.neighbors(pos, self.solve)
            # why did I chose not to provide neighbors?
            for i, ase in enumerate(normal_nodes):
                ase = RuleEdge(*ase)
                assert isinstance(ase.cost, int), f"cost is not a number:{pos}"
                new_trace = trace + [Edge(pos, ase.new_state, ase.converter, ase.cost, ase.name)]
                try:
                    new_score = scores[pos] + ase.cost + self.heuristics(ase.new_state, end)
                except Exception as e:
                    logger.error(f"pos:{pos},cost:{ase.cost},next_node:{ase.new_state}")
                    raise e
                if ase.new_state in scores and scores[ase.new_state] <= new_score:
                    continue
                else:
                    scores[ase.new_state] = new_score
                    to_visit.push(new_score,(ase.new_state,new_trace))

        raise NoRouteException(f"no route found from {start} matching {end}. searched {visited} nodes.")


DEFAULT_ASTAR_DESIGN = Design().bind_instance(
    heuristics=lambda x, y: 0,
    max_depth=100,
    silent=False,
    rulebook=DEFAULT_RULES,
    rulebook_with_solver=RulesUsingSolver([])
).bind_class(
    neighbors=Neighbors
)
# I think the solver is ready
# and I can use dict_rules with a solver implemented.

#create a scala's parameterized dependency solving functionality.
# for use 