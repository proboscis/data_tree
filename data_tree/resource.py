import abc


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
        return self.use(lambda data:data)


class ResContextManager:
    def __init__(self, res:Resource):
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

    def release(self,resource):
        if self.releaser is not None:
            return self.releaser(resource)


def shared_resource():
    pass
