import abc
import inspect
from dataclasses import dataclass
from typing import List, Generic, Mapping, Union, Callable, TypeVar, Tuple, Set

import deprecate
from loguru import logger
from makefun import create_function
from snoop import snoop

from data_tree.image_store import T, U

A = TypeVar("A")
B = TypeVar("B")


@dataclass
class FrameInfo:
    filename: str
    line_number: int
    function_name: str
    sources: List[str]
    line_idx_in_sources: int


def get_frame_info(stack_idx):
    return FrameInfo(*inspect.getframeinfo(inspect.stack()[stack_idx][0]))


class Injected(Generic[T], metaclass=abc.ABCMeta):
    """
    what I want to achieve is ..
    to define a function that can accept injected[T]s to return injected[X]
    I only want to define a function that converts it.
    so given a function we want to lift that function to Injected Function
    so, define a function and convert it with mappings
    """

    @staticmethod
    def bind(_target_function_, **kwargs_mapping: Union[str, type, Callable, "Injected"]) -> "Injected":
        return InjectedFunction(
            target_function=_target_function_,
            kwargs_mapping=kwargs_mapping
        )

    def __init__(self, init_stack):
        frame = get_frame_info(init_stack)
        self.fname = f"{frame.filename.split('.')[-2].split('/')[-1]}_L{frame.line_number}".replace(".py", "")

    @abc.abstractmethod
    def dependencies(self) -> Set[str]:
        pass

    def get_signature(self):
        sig = f"""{self.fname}({",".join(self.dependencies())})"""
        logger.warning(sig)
        return sig

    @abc.abstractmethod
    def get_provider(self):
        """
        :return: a provider
        """

    def map(self, f: Callable[[T], U]) -> 'Injected[U]':
        return MappedInjected(4,self, f)


class MappedInjected(Injected):
    src: Injected[T]
    f: Callable[[T], U]

    def __init__(self, stack, src, f):
        super(MappedInjected, self).__init__(stack)
        self.src = src
        self.f = f

    def dependencies(self) -> Set[str]:
        return self.src.dependencies()

    def get_provider(self):
        def impl(**kwargs):
            tmp = self.src.get_provider()(**kwargs)
            return self.f(tmp)

        return create_function(self.get_signature(), func_impl=impl)


def extract_dependency(dep: Union[str, type, Callable, Injected]) -> Set[str]:
    if isinstance(dep, str):
        return {dep}
    elif isinstance(dep, (type, Callable)):
        argspec = inspect.getfullargspec(dep)
        return set(argspec.args) - {'self'}
    elif isinstance(dep, Injected):
        return dep.dependencies()
    else:
        raise RuntimeError(f"dep must be either str/type/Callable/Injected. got {type(dep)}")


def solve_injection(dep: Union[str, type, Callable, Injected], kwargs: dict):
    if isinstance(dep, str):
        return kwargs[dep]
    elif isinstance(dep, (type, Callable)):
        return dep(**{k: kwargs[k] for k in extract_dependency(dep)})
    elif isinstance(dep, Injected):
        return solve_injection(dep.get_provider(), kwargs)
    else:
        raise RuntimeError(f"dep must be one of str/type/Callable/Injected. got {type(dep)}")


def combine_image_store(a, b):
    # do anything
    return a + b


class InjectedFunction(Injected[T]):
    def __init__(self,
                 target_function: Callable,
                 kwargs_mapping: Mapping[str, Union[str, type, Callable, Injected]]
                 ):
        super().__init__(init_stack=4)
        self.target_function = target_function
        self.kwargs_mapping = kwargs_mapping
        org_deps = extract_dependency(self.target_function)
        missings = {d for d in org_deps if d not in self.kwargs_mapping}
        self.missings = missings

    def get_provider(self):
        signature = self.get_signature()

        def impl(**kwargs):
            deps = dict()
            for mdep in self.missings:
                deps[mdep] = solve_injection(mdep, kwargs)
            for k, dep in self.kwargs_mapping.items():
                deps[k] = solve_injection(dep, kwargs)
            return self.target_function(**deps)

        return create_function(func_signature=signature, func_impl=impl)

    def dependencies(self) -> Set[str]:
        res = set()
        for mdep in self.missings:
            res |= extract_dependency(mdep)
        for dep in self.kwargs_mapping.values():
            deps = extract_dependency(dep)
            res |= deps
        return res


@dataclass
class ZippedInjected(Injected[Tuple[A, B]]):
    a: Injected[A]
    b: Injected[B]

    def dependencies(self) -> Set[str]:
        return set(self.a.dependencies() + self.b.dependencies())

    def get_provider(self):
        # TODO implement
        pass
