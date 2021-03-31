import abc
import inspect
from copy import copy
from dataclasses import dataclass, field
from types import FunctionType
from typing import Union, Type, TypeVar

import pinject
from IPython import embed
from loguru import logger
from makefun import create_function, wraps
from pinject import BindingSpec
from pinject.object_graph import ObjectGraph
from pprintpp import pformat

prototype = pinject.provides(in_scope=pinject.PROTOTYPE)
# is it possible to create a binding class which also has an ability to dynamically add..?
# yes. actually.
T = TypeVar("T")


class TraversableBinding(BindingSpec):
    def traversed(self):
        return DynamicBindingContainer.gather_spec(self)


def inject_proto(all_except=None, arg_names=None):
    """injects provider with prototype context"""

    def _inner(f):
        return pinject.inject(all_except=all_except, arg_names=arg_names)(prototype(f))

    return _inner


def method_to_function(method):
    """
    converts a class method to a function
    """
    argspec = inspect.getfullargspec(method)
    signature = f"""f_of_{method.__name__}({" ,".join((argspec.args or []) + (argspec.varargs or []))})"""

    def impl(self, *args, **kwargs):
        return method(*args, **kwargs)  # gets multiple values for self

    assert len(argspec.kwonlyargs) == 0, "providing method cannot have any kwonly args"
    return create_function(signature, impl)


def ensure_self_arg(func):
    """
    adds a self arg in to func signature if not present
    :param func:
    :return:
    """
    argspec = inspect.getfullargspec(func)
    if argspec.args and argspec.args[0] == "self":
        return func
    source = inspect.getsource(func)
    module = inspect.getmodule(func)
    lines, ln = inspect.getsourcelines(func)
    # logger.error(source)
    signature = f"""{module.__name__.replace(".", "_")}_ln_{ln}_{func.__name__}({" ,".join(["self"] + (argspec.args or []) + (argspec.varargs or []))})"""

    def impl(self, *args, **kwargs):
        return func(*args, **kwargs)  # gets multiple values for self

    assert len(argspec.kwonlyargs) == 0, "providing method cannot have any kwonly args"
    return create_function(signature, impl)


def none_provider(func):
    argspec = inspect.getfullargspec(func)
    signature = f"""none_provider({" ,".join((argspec.args or []) + (argspec.varargs or []))})"""

    def impl(*args, **kwargs):
        func(*args, **kwargs)  # gets multiple values for self
        return "success of none_provider"

    assert len(argspec.kwonlyargs) == 0, "providing method cannot have any kwonly args"
    return create_function(signature, impl)


class ExtendedObjectGraph:
    """
    an object graph which can also provide instances based on its name not only from class object.
    """

    def __init__(self, src: ObjectGraph):
        self.src = src

    def provide(self, target: Union[str, Type[T]]) -> Union[object, T]:
        if isinstance(target, str):
            code = compile(f"""def __init__(self,{target}):self.res={target}""", "<string>", "exec")
            fn = FunctionType(code.co_consts[0], globals(), "__init__")
            Request = type("Request", (object,), dict(__init__=fn))
            return self.src.provide(Request).res
        elif isinstance(target, type):
            return self.src.provide(target)
        else:
            raise TypeError(f"target must be either class or a string")


@dataclass
class DynamicBindingContainer:
    """
    This is an injection binding class which can be used to compose configures and providers.
    """
    providers: dict = field(default_factory=dict)
    configures: dict = field(default_factory=dict)
    modules: list = field(default_factory=list)
    classes: list = field(default_factory=list)

    def to_binding_spec(self):
        """Generate a BindingSpec class for pinject and returns its instance."""
        configures = self.configures

        def configure(self, bind):
            for c, v in configures.items():
                bind(c, **v)

        DynamicBinding = type("DynamicBinding", (BindingSpec,),
                              {
                                  "configure": configure,
                                  **{f"provide_{k}": v for k, v in self.providers.items()}
                              })
        return DynamicBinding()

    def __add__(self, other: Union["DynamicBindingContainer", BindingSpec]):
        other = self._ensure_dbc(other)
        return self.merged(other)

    def update(self, *others: Union["DynamicBindingContainer", BindingSpec]):
        """adds bindings from others. into this instance. inplace operation"""
        for o in others:
            self.merge_inplace(o)

    def _ensure_dbc(self, other: Union["DynamicBindingContainer", BindingSpec]):
        if isinstance(other, BindingSpec):
            other = DynamicBindingContainer.gather_spec(other)
        return other

    def merged(self, other: "DynamicBindingContainer"):
        """creates another instance with merged bindings. does not modify self"""
        # logger.debug(f"merging:\n\t{self} to \n\t{other}")
        res = DynamicBindingContainer(
            providers=self.providers.copy(),
            configures=self.configures.copy(),
            modules=list(set(self.modules) | set(self.modules)),
            classes=list(set(self.classes) | set(self.classes))
        )
        for k, v in other.providers.items():
            if k in res.configures:
                logger.warning(f"override configure\n\t:{k}->{self.configures[k]} to {v}")
                del res.configures[k]
            if k in res.providers:
                logger.warning(f"override provider\n\t:{k}->{self.providers[k]} to {v}")
            res.providers[k] = v
        for k, v in other.configures.items():
            if k in res.configures:
                logger.warning(f"override configure\n\t:{k}->{self.configures[k]} to {v}")
            if k in res.providers:
                logger.warning(f"override provider\n\t:{k}->{self.providers[k]} to {v}")
                del res.providers[k]
            res.configures[k] = v

        # logger.debug(f"merged:\n\t{res}")
        return res

    def merge_inplace(self, other: "Union[DynamicBindingContainer,BindingSpec]"):
        other = self._ensure_dbc(other)
        merged = self.merged(other)
        self.providers = merged.providers
        self.configures = merged.configures
        self.modules = merged.modules
        self.classes = merged.classes
        return self

    def provide_inplace(self, key, provider):
        """
        :param key:
        :param provider: any provider function like def (self,deps)->result. canbe annotated
        with @pinject.inject like @inject(all_except=..., annotated_with...)
        :return:
        """
        # provider.__name__ = f"provide_{key}"
        _override = DynamicBindingContainer(
            providers={key: provider},
            configures=dict(),
            modules=[]
        )
        return self.merge_inplace(_override)

    def bind_inplace(self, key, **kwargs):
        """
        :param key:
        :param kwargs: to_instance or to_class, and annotated_with
        :return:
        """
        _override = DynamicBindingContainer(
            providers=dict(),
            configures={key: kwargs},
            modules=[]
        )
        return self.merge_inplace(_override)

    def batch_provide_inplace(self, providers: dict):
        x = self
        for k, p in providers.items():
            x = x.provide_inplace(k, p)
        return self

    def batch_bind_inplace(self, configs: dict):
        x = self
        for k, p in configs.items():
            x = x.bind_inplace(k, **p)
        return x

    def to_graph(self, modules=None, classes=None) -> ExtendedObjectGraph:
        modules = self.modules + (modules or [])
        classes = self.classes + (classes or [])
        logger.info(f"to_graph:\n\t{pformat(modules)}\n\t{pformat(classes)}")
        assert classes
        return ExtendedObjectGraph(pinject.new_object_graph(
            modules=modules, binding_specs=[self.to_binding_spec()],
            classes=classes
        ))

    def provided(self, target: str, modules=None, classes=None) -> object:
        """
        :param target: provided name
        :param modules: modules to use for graph construction
        :return:
        """
        return self.to_graph(modules=modules, classes=classes).provide(target)

    @staticmethod
    def gather_spec(b: BindingSpec) -> "DynamicBindingContainer":
        dep_spec = DynamicBindingContainer(dict(), dict(), [], [])
        for dep in b.dependencies() or []:
            spec = DynamicBindingContainer.gather_spec(dep)
            # logger.debug(f"dependency:{spec}")
            dep_spec = dep_spec.merged(spec)
        spec = DynamicBindingContainer(dict(), dict(), [], [])

        def binder(key, **kwargs):
            spec.configures[key] = kwargs

        if hasattr(b, "configure"):
            try:
                b.configure(binder)
            except NotImplementedError as nie:
                pass
        for item in dir(b):
            if item.startswith("provide") and callable(getattr(b, item)):
                spec.providers[item.replace("provide_", "")] = getattr(b, item)
        res = dep_spec.merged(spec)
        return res

    @staticmethod
    def traverse_specs(*bindings: BindingSpec):
        spec = DynamicBindingContainer(dict(), dict(), [])
        for b in bindings:
            spec = spec + b
        # logger.debug(f"traversed binding:\n\t{spec}")
        return spec

    def set_module(self, modules):
        self.modules = modules

    def set_classes(self, classes):
        self.classes = classes

    def bind_provider(self, key, f, all_except=None, arg_names=None):
        argspec = inspect.getfullargspec(f)
        if f.__name__ == "<lambda>":
            f.__name__ = "_lambda_"
        # so first you need to convert a method in to function
        if "__name__" not in dir(f):
            f = method_to_function(f)
        # now ensure the function has self arg
        f = ensure_self_arg(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper.__name__ = f"provide_{key}"
        if all_except is not None or arg_names is not None:
            wrapper = inject_proto(all_except=all_except, arg_names=arg_names)(wrapper)
        new_argspec = inspect.getfullargspec(wrapper)
        # logger.debug(f"set_provider:{wrapper} \n\t {new_argspec}")
        if "self" not in new_argspec.args:
            embed()
            raise RuntimeError("self must be insided signature for provider")
        if key in self.configures:
            del self.configures[key]
        self.providers[key] = wrapper
        return self

    def as_provider(self, f):
        self.bind_provider(f.__name__, ensure_self_arg(f))
        return f

    def extend_provider_inplace(self, src_key, new_name, provider):
        """
        :param src_key: key to hook for input
        :param new_name: name to provide old values
        :param provider: def provider(self,deps)...
        example:
        spec.extend_provider_inplace("generator","old_generator",lambda self,old_generator:wrap(old_generator))
        :return: None
        """
        if src_key in self.providers:
            tmp = self.providers[src_key]
            self.bind_provider(src_key, provider)
            self.bind_provider(new_name, tmp)
        if src_key in self.configures:
            tmp = self.configures[src_key]
            del self.configures[src_key]
            self.bind_provider(src_key, provider)
            self.configures[new_name] = tmp

    def _gen_name(self, name):
        used_names = set(self.providers.keys()) | set(self.configures.keys())
        new_name = f"__generated_{name}"
        i = 0
        while new_name in used_names:
            new_name = f"{new_name}_{i}"
        return new_name

    def map_value(self, src_key, f):
        copied = self.__class__(copy(self.providers), copy(self.configures), copy(self.modules), copy(self.classes))
        copied.map_value_inplace(src_key=src_key, f=f)
        return copied

    def map_value_with_provider(self, src_key, dependencies, f):
        copied = self.__class__(copy(self.providers), copy(self.configures), copy(self.modules), copy(self.classes))
        copied.map_value_inplace_with_provider(src_key, dependencies, f)
        return copied

    def map_value_inplace_with_provider(self, src_key, dependencies, f):
        """
        :param src_key: target value to map in pinject graph
        :param dependencies: list of keys to depend on other than src_key
        :param f: must expect single arg which is a list
        :return:
        """

        def extract_argnames(func):
            spec = inspect.getfullargspec(func)
            return spec.args

        # first make f kwarg based function
        f_signature = f"mapper_with_dependencies(src,{','.join(dependencies)})"

        def f_impl(src, **kwargs):
            return f([src, *[kwargs[k] for k in dependencies]])

        new_f = create_function(f_signature, f_impl)

        def convert_provider(src_provider):
            src_provider = method_to_function(src_provider)
            logger.info(f"converting provider to use additional dependencies")
            src_deps = extract_argnames(src_provider)
            src_deps.remove("self")  # assuming that a src provider always have self.(which is true)
            combined = list(set(src_deps) | set(dependencies))
            src_indices = [combined.index(k) for k in src_deps]
            mapped_indices = [combined.index(k) for k in dependencies]
            signature = f"signature_for_provider(self,{','.join(combined)})"
            logger.info(f"mapping provier with dependency, signature:\n\t {signature}")

            def impl_for_provider(self, **args):
                src_args = {k: args[k] for k in src_deps}
                logger.debug(f"src_args:{src_args.keys()}")
                mapped_args = {k: args[k] for k in dependencies}
                logger.debug(f"mapped_args:{mapped_args.keys()}")
                return new_f(src_provider(self, **src_args), **mapped_args)

            provider = create_function(signature, impl_for_provider)
            return provider

        if src_key in self.configures:
            tmp = self.configures[src_key]
            del self.configures[src_key]
            if "to_class" in tmp:
                provider = convert_provider(method_to_function(tmp["to_class"]))
            else:
                signature = f"provide_{src_key}(self,{','.join(dependencies)})"

                def impl_for_instance(self, **kwargs):
                    return new_f(tmp["to_instance"], **kwargs)

                logger.info(f"instance binding mapping with dependencies:\n\t{signature}")
                provider = create_function(signature, impl_for_instance)

        elif src_key in self.providers:
            src_provider = self.providers[src_key]
            provider = convert_provider(src_provider)
        else:
            raise RuntimeError(f"unknown situation in mapping")

        self.bind_provider(src_key, provider)

    def map_value_inplace(self, src_key, f):
        """
        inplace mapping of a provider
        :param src_key:
        :param f:
        :return: self
        """

        def map_wrap(src):
            @wraps(src)
            def wrapper(*args, **kwargs):
                return f(src(*args, **kwargs))

            return wrapper

        # new_name = self._gen_name(src_key)
        if src_key in self.providers:
            tmp = self.providers[src_key]
            if "__name__" not in dir(tmp):
                tmp = method_to_function(tmp)
            wrapped = map_wrap(tmp)
            argspec = inspect.getfullargspec(wrapped)
            assert "self" in argspec.args
            self.bind_provider(src_key, wrapped)
        elif src_key in self.configures:
            # delete binding and add a provider mapped with f
            tmp = self.configures[src_key]
            del self.configures[src_key]
            if "to_class" in tmp:
                provider = method_to_function(tmp["to_class"])
            else:
                signature = f"provide_{src_key}(self)"

                def impl(self):
                    return tmp["to_instance"]

                provider = create_function(signature, impl)
            self.bind_provider(src_key, map_wrap(provider))
        else:
            raise KeyError(f"no provider to map:{src_key}")
        return self

    def __contains__(self, item):
        return item in self.providers or item in self.configures

    def __getitem__(self, item):
        return {**self.providers, **self.configures}[item]

    def __str__(self):
        import tabulate
        provs = tabulate.tabulate(sorted(list(self.providers.items())))
        confs = tabulate.tabulate(sorted(list(self.configures.items())))
        return provs + "\n" + confs

    def __repr__(self):
        return str(self)


class DynamicSpec(BindingSpec):
    def __init__(self):
        logger.error(f"initializing dynamic spec")
        container = DynamicBindingContainer()
        self.dynamic_configure(container)
        self._dependencies = [container.to_binding_spec()]

    @abc.abstractmethod
    def dynamic_configure(self, binder: DynamicBindingContainer):
        pass

    def configure(self, bind):
        pass

    def dependencies(self):
        return self._dependencies
