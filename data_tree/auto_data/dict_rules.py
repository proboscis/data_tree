# def dict_conversions(state):

"""
intra structure rules require a solver.
so the rulebook contains several kind of rules.
ohh finally I can start implementing auto solver...
"""
from collections import Callable
from copy import copy

from frozendict import frozendict as fdict


def map_fdict_element(key, mapper):
    def _impl(target):
        tmp = mapper(target[key])
        res = copy(target)
        res[key] = tmp
        return res

    return _impl


def key_neighbor_to_fdict_neighbor(key, neighbor: tuple, data: fdict):
    copied = copy(data)
    copied[key] = neighbor[1]
    return (
        # converter
        map_fdict_element(key, neighbor[0]),
        # new state
        copied,
        # cost,
        neighbor[2],
        # name
        f"dict elem conversion {neighbor[3]}"
    )


def intra_dict_rule(state, solver: Callable[[object], Callable]):
    if isinstance(state, fdict):
        pass
        # for each key I get conversions from a solver.
    # now I realized that it is better to find things from the bottom. since it is the required result.
    # and why is that??
    #  *
    # and that requires a whole rewrite of rules since the rules are written in opposite direction.
    # so... for now forget about that
    # and use that for other tasks
    # I see why the search is done from the result
    # the result is always a deterministic point while the input is not.
    # this was not the case when I was only searching for A=>B edge, where A is always a valid instance.
    # or lets say that defining generic output is difficult?
    # I think that goes the same for input.
    # so.. I need a way to define generic output state.
    # actually I think it is possible
    # a dict can be turned into generic parameterized state
    # and then turned into result
    # so for example, a dict to widget converter should ask an input state to be generic dict.


# the reason why I am not using kanren is that they use depth first search.
# I need a breadth first search/ a* search




