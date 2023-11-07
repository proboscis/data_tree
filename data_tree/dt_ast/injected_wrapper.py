from typing import Union

from pinjected.di.graph import IObjectGraph
from pinjected import Design
from data_tree.dt_ast.dt_core import AstExpr, AstCall, AstTuple, AstDict, match_dataclass, AstObject


def injected(*dependencies: str):
    """
    given a function [Dep,T]->U, converts it to Injected[T->U] by making specified deps as injected by Injected.
    :param dependencies:
    :return:
    """

    # converts (dep,T)=>U to Injected[T=>U]
    def convert(f):
        # all_args = extract_dependency(f)
        def injected_impl(**deps):  # this way I need to use create_function
            def func_impl(*args, **kwargs):
                # we need to pass both deps and args and kwargs at this point.
                # we need to remember how the deps are defined in the original f.
                # to correctly pass injected args.
                # for now I just force the user to specify deps as kwargs
                # logger.info(f"calling with {args,kwargs,deps}")
                # logger.info(f"calling :{f}")
                return f(*args, **kwargs, **deps)

            return func_impl

        impl = AstExpr.object(GeneratedInjected(injected_impl, set(dependencies), 2))
        return impl

    return convert


def run_injected(ast: AstExpr, g: Union[Design, IObjectGraph]):
    """
    runs a given ast which contains Injected as Object. with provided graph.
    so you can construct a expression without actually evaluating Injected values,
    and then finally run it with this function.
    or maybe convert that Ast into a Injected instance?
    :param ast:
    :param g:
    :return:
    """
    from pampy import match
    if isinstance(g, Design):
        g = g.to_graph()

    def _call(c: AstCall):
        callee = run(c.func)
        args = run(AstTuple(c.args))
        kwargs = run(AstDict(c.kwargs))
        return callee(*args, **kwargs)

    patterns = [
        AstTuple, lambda asts: tuple([run(a) for a in asts.items]),
        AstDict, lambda ast_dict: {k: run(v) for k, v in ast_dict.dict_data.items()},
        match_dataclass(AstObject(Injected)), lambda i: g.provide(i),
        AstCall, _call,
        AstObject, lambda data: data.data  # finally just pass raw data
    ]

    def run(ast_expr: AstExpr):
        return match(ast_expr,
                     *patterns)

    return run(ast)
