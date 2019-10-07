from uuid import uuid4

GLOBAL_OBJECTS = dict()


class GlobalHolder:
    def __init__(self, id: str):
        self.id = id

    def release(self):
        del GLOBAL_OBJECTS[self.id]

    @property
    def value(self):
        return GLOBAL_OBJECTS[self.id]


def get_global_holder(data: object) -> GlobalHolder:
    id = str(uuid4())
    GLOBAL_OBJECTS[id] = data
    return GlobalHolder(id)
