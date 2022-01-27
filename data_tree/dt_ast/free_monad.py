import abc
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Any, Tuple, Dict
from toolz.functoolz import compose_left

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
S = TypeVar("S")


class Functor(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def map(self, fa: T[A], f: Callable[[A], B]):
        pass


class Ast: pass


@dataclass
class Call:
    f: Any
    args: Tuple[Any]
    kwargs: Dict[str, Any]


class AstFunctor(Functor[Ast]):
    def map(self, fa: Ast[A], f: Callable[[A], B]):
        if isinstance(fa, Call):
            return Call(compose_left(fa.f, f), fa.args, fa.kwargs)
        else:
            raise NotImplementedError()


class Free(Generic[S, A]):
    def map(self, f: Callable[[A], B]) -> "Free[S,B]":
        return self.flat_map(lambda a: Pure(f(a)))

    def flat_map(self, f: Callable[[A], "Free[S,B]"]) -> "Free[S,B]":
        return Gosub(self, f)


@dataclass
class Pure(Free[S, A]):
    value: A


@dataclass
class Gosub(Free[S, A]):
    c: Free[S, C]
    f: Callable[[C], Free[S, B]]


@dataclass
class Suspend(Free[S, A]):
    a: S[A]

def liftf(functor:Functor[S],f:)->Free[S,A]:
    # given a functor we should be able to construct a monad
    # but I dont need it for now.
    pass

