import abc
from abc import ABC, ABCMeta
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Callable, Iterable

from frozendict import frozendict

from pampy import match_dict
from snoop import snoop

from data_tree import logger


class Expr: pass


# def __call__(self, *args, **kwargs) -> "Call":
#    return Call(self, args, kwargs)


@dataclass(frozen=True)
class Call(Expr):
    func: Expr
    args: Tuple[Expr] = field(default_factory=tuple)
    kwargs: Dict[str, Expr] = field(default_factory=dict)
    def __hash__(self):
        return hash(hash(self.func) + hash(self.args) + hash(frozendict(self.kwargs)))



@dataclass(frozen=True)
class Attr(Expr):
    data: Expr
    attr_name: str  # static access so no ast involved


@dataclass(frozen=True)
class Object(Expr):
    """
    Use this to construct an AST and then compile it for any use.
    """
    data: Any  # holds user data
    def __hash__(self):
        return hash(id(self.data))

    def __repr__(self):
        return f"Object({str(self.data)[:20]})".replace("\n","").replace(" ","")


@dataclass(frozen=True)
class TupleExpr(Expr):
    items: Tuple[Expr]


@dataclass(frozen=True)
class DictExpr(Expr):
    data: Dict[str, Expr]


@dataclass(frozen=True)
class Symbol(Expr):
    identifier: str


@dataclass(frozen=True)
class Assign(Expr):
    tgt: Symbol
    src: Expr



class AstExpr(metaclass=ABCMeta):
    # TODO give an id upon call.
    def __call__(self, *args, **kwargs):
        args = [(
            AstExpr.object(arg) if not isinstance(arg, AstExpr) else arg)
            for arg in args]
        kwargs = {k: (AstExpr.object(a) if not isinstance(a, AstExpr) else a) for k, a in kwargs.items()}
        return AstCall(self, args, kwargs)

    def __getattr__(self, item: str):
        if not item.startswith("__") or item != "shape":
            return AstAttr(self, item)
        else:
            logger.warning(f"trying to access {item}")
            raise AttributeError("no such attribute:{item}")

    @abc.abstractmethod
    def to_ast(self) -> Expr:
        pass

    @staticmethod
    def object(data):
        return AstObject(data)

    @staticmethod
    def tuple(items:Iterable["AstExpr"]):
        return AstTuple(tuple(items))

    @staticmethod
    def macro(f):
        return AstMacro(f)

    @staticmethod
    def symbol(name:str):
        return AstSymbol(name)

    # def map(self, f: Callable[[Any], Any]):
    #     raise NotImplementedError()
    #     return AstMap(self, f)
    #
    # def flatmap(self, f: Callable[[Any], "AstExpr"]):
    #     raise NotImplementedError()
    #     return AstFlatMap(self, f)


@dataclass
class AstFlatMap(AstExpr):
    src: AstExpr
    f: Callable


@dataclass
class AstMap(AstExpr):
    src: AstExpr
    f: Callable


@dataclass
class AstObject(AstExpr):
    data: Any

    def to_ast(self) -> Expr:
        return Object(self.data)

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state

    def __str__(self):
        return "AstObject"


@dataclass
class AstCall(AstExpr):
    func: AstExpr
    args: Tuple[AstExpr] = field(default_factory=tuple)
    kwargs: Dict[str, AstExpr] = field(default_factory=dict)

    def to_ast(self) -> Expr:
        return Call(
            func=self.func.to_ast(),
            args=tuple([t.to_ast() for t in self.args]),
            kwargs={k: t.to_ast() for k, t in self.kwargs.items()}
        )

    def __getstate__(self):
        return (self.func, self.args, self.kwargs)

    def __setstate__(self, state):
        self.func, self.args, self.kwargs = state


@dataclass
class AstSymbol(AstExpr):
    identifier: str

    def to_ast(self) -> Expr:
        return Symbol(self.identifier)


@dataclass
class AstAssign(AstExpr):
    tgt: AstSymbol
    src: AstExpr

    def to_ast(self) -> Expr:
        return Assign(self.tgt.to_ast(), self.src.to_ast())


@dataclass
class AstAttr(AstExpr):
    data: AstExpr
    attr: str

    def to_ast(self) -> Expr:
        return Attr(
            data=self.data.to_ast(),
            attr_name=self.attr
        )

    def __getstate__(self):
        return self.data, self.attr

    def __setstate__(self, state):
        self.data, self.attr = state


@dataclass
class AstTuple(AstExpr):
    items: Tuple[AstExpr]

    def to_ast(self) -> Expr:
        return TupleExpr(tuple([expr.to_ast() for expr in self.items]))

    def __getstate__(self):
        return self.items

    def __setstate__(self, state):
        self.items = state


@dataclass
class AstDict(AstExpr):
    dict_data: Dict[str, AstExpr]

    def to_ast(self) -> Expr:
        return DictExpr({k: v.to_ast() for k, v in self.dict_data.items()})

    def __getstate__(self):
        return self.dict_data

    def __setstate__(self, state):
        self.dict_data = state


@dataclass
class AstMacro(AstExpr):
    macro: Callable[[AstExpr], AstExpr]

    def __call__(self, *args, **kwargs)->AstExpr:
        return AstMacroCall(self.macro, AstTuple(args), AstDict(kwargs)).expand()

    def to_ast(self) -> Expr:
        raise RuntimeError("macro needs to be called before converting into ast")

    def __setstate__(self, state):
        self.macro = state

    def __getstate__(self):
        return self.macro


@dataclass
class AstMacroCall(AstExpr):
    _macro: Callable[[AstExpr], AstExpr]
    _args: AstTuple
    _kwargs: AstDict

    def to_ast(self) -> Expr:
        return self.expand().to_ast()

    def expand(self):
        return self._macro(*self._args.items, **self._kwargs.dict_data)

    def __setstate__(self, state):
        self._macro, self._args, self._kwargs = state

    def __getstate__(self):
        return self._macro, self._args, self._kwargs


def match_dataclass(pattern):
    def _impl(tgt):
        if pattern.__class__ == tgt.__class__:
            return match_dict(pattern.__dict__, tgt.__dict__)
        else:
            return False, []

    return _impl


def replace_object_in_ast(ast, pattern, replacer) -> AstExpr:
    from pampy import match, _
    def _replace(o: AstExpr) -> AstExpr:
        return match(o,
                     match_dataclass(AstObject(pattern)), lambda *args: AstObject(replacer(*args)),
                     AstTuple, lambda _tuple: AstTuple(tuple([_replace(t) for t in _tuple.items])),
                     AstAttr, lambda attr: AstAttr(_replace(attr.data), attr.attr),
                     AstCall, lambda call: AstCall(
                _replace(call.func),
                tuple(_replace(item) for item in call.args),
                {k: _replace(v) for k, v in call.kwargs.items()}
            ),
                     AstMacroCall, lambda amc: _replace(amc.expand()),
                     AstObject, lambda ao: ao
                     )

    return _replace(ast)

# we need special map and reduce operator to be used... actually we can convert obj.map to python's map
# which should I choose, I guess I should choose py_map over task.map? not really. func.map is ok I think.
# in that case I have to handle Call(Attr(Object(func),"map"),args) to map(f,args) for python
