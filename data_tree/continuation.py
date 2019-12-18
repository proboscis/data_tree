from abc import abstractmethod


class Continuation:
    def map(self, mapper):
        return MappedContinuation(self, mapper)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class FContinuation(Continuation):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class MappedContinuation(Continuation):
    def __init__(self, src: Continuation, mapping):
        self.src = src
        self.mapping = mapping

    def __call__(self, *args, **kwargs):
        return self.mapping(self.src(*args, **kwargs))
