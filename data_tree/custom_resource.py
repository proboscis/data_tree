import abc
from concurrent.futures import Executor, ThreadPoolExecutor, Future
from queue import Queue
from threading import Thread
from typing import Callable

from lazy import lazy



class Resource(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def prepare(self):
        pass

    @abc.abstractmethod
    def release(self, resource):
        pass

    def use(self, f):
        resource = self.prepare()
        try:
            result = f(resource)
        finally:
            self.release(resource)
        return result

    def map(self, f):
        return MappedResource(self, f)

    def to_context(self):
        return ResContextManager(self)

    def get(self):
        return self.use(lambda data: data)


class ResContextManager:
    def __init__(self, res: Resource):
        self.preparer = res.prepare
        self.releaser = res.release
        self.resource = None

    def __enter__(self):
        self.resource = self.preparer()
        return self.resource

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.releaser(self.resource)


class MappedResource(Resource):
    def __init__(self, src: Resource, f):
        self.src = src
        self.mapper = f

    def prepare(self):
        return self.src.prepare()

    def release(self, resource):
        return self.src.release(resource)

    def use(self, f):
        res = self.prepare()
        try:
            mapped = self.mapper(res)
            result = f(mapped)
        finally:
            self.release(res)
        return result


class ContextResource(Resource):
    def __init__(self, context_creator):
        self.context_creator = context_creator

    def prepare(self):
        return self.context_creator()

    def release(self, resource):
        resource.close()


class LambdaResource(Resource):
    def __init__(self, preparer=None, releaser=None):
        self.preparer = preparer
        self.releaser = releaser

    def prepare(self):
        if self.preparer is not None:
            return self.preparer()

    def release(self, resource):
        if self.releaser is not None:
            return self.releaser(resource)


class DaemonExecutor:
    def __init__(self):
        self.queue = Queue()

        def work():
            while True:
                f, task, args, kwargs = self.queue.get()
                try:
                    res = task(*args, **kwargs)
                    f.set_result(res)
                except Exception as e:
                    f.set_exception(e)

        self.thread = Thread(target=work, daemon=True)

    def submit(self, task, *args, **kwargs) -> Future:
        f = Future()
        self.queue.put_nowait((f, task, args, kwargs))
        return f


class ExecutorBound:

    @property
    @abc.abstractmethod
    def executor(self) -> Executor:
        # THIS causes deadlock if executor has only one worker
        pass

    def map(self, f, executor=None) -> "ExecutorBound":
        if executor is None:
            executor = self.executor
        return MappedEB(self, f, executor)

    def flatmap(self, f, executor=None) -> "ExecutorBound":
        # you need to check if our results resides in the same context,
        if executor == self.executor:
            return f(self.direct).direct
        else:
            return self.map(lambda item: f(item).value, executor)

    def __or__(self, other):  # object => object function
        return self.flatmap(lambda item: ExecutorBound.create(lambda: other(item), self.executor))

    @property
    def value(self):
        return self.future.result()

    @property
    @abc.abstractmethod
    def future(self):
        pass

    @staticmethod
    def create(f, executor: Executor):
        return ProcExecutorBound(f, executor)

    @property
    def direct(self):
        pass


def do_notation(executor):
    def annotate(f: Callable):
        def l():
            g = f()
            fut = next(g)
            while True:
                try:
                    assert isinstance(fut,ExecutorBound),"you must yield ExecutorBound while using do_notation!"
                    fut = g.send(fut.direct)
                except StopIteration:
                    return fut.direct

        return ExecutorBound.create(l, executor)

    return annotate


class ProcExecutorBound(ExecutorBound):
    @property
    def executor(self) -> Executor:
        return self._executor

    def __init__(self, f, executor):
        self._executor = executor
        self.f = f
        self.task = None
        self._future = Future()

    @lazy
    def future(self):
        if self.task is None and not self._future.done():
            self.task: Future = self.executor.submit(self.f)
            self.task.add_done_callback(lambda res: self._future.set_result(res.result()))
        return self._future

    @property
    def direct(self):
        if self.task is None and not self._future.done():
            res = self.f()
            self._future.set_result(res)
            return res
        elif self.task is not None:
            return self.task.result()
        elif self._future.done():
            return self._future.result()


class MappedEB(ExecutorBound):
    @property
    def executor(self) -> Executor:
        return self._executor

    @lazy
    def future(self):
        if self.task is None and not self._future.done():
            self.task: Future = self.executor.submit(lambda: self.mapper(self.src.value))
            self.task.add_done_callback(lambda res: self._future.set_result(res.result()))
        return self._future

    @property
    def direct(self):
        if self.task is None and not self._future.done():
            src_res = self.src.direct
            res = self.mapper(src_res)
            self._future.set_result(res)
            return res
        elif self.task is not None:
            return self.task.result()
        elif self._future.done():
            return self._future.result()

    def __init__(self, src: ExecutorBound, mapper, executor: Executor):
        self.mapper = mapper
        self._executor = executor
        self.src = src
        self._future = Future()
        self.task = None


def fopen(path,mode:str):
    return ContextResource(lambda:open(path,mode=mode))

def shared_resource():
    pass
