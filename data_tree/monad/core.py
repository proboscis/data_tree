import inspect
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class TryWrappedFunction:
    f: Callable

    def __post_init__(self):
        self.__signature__ = inspect.signature(self.f)

    def __call__(self, *args, **kwargs):
        try:
            res = self.f(*args, **kwargs)
            return Success(res)
        except Exception as e:
            import traceback
            # trc = traceback.format_exc()
            return Failure(e) # why does this fail at all?


@dataclass
class ErrorWithTrace:
    e: Exception
    trc: str

    def __str__(self):
        non_escaped = "\n".join(self.trc.split("\n"))
        return f"""
ErrorWithTrace
{self.e},
{non_escaped} 
        """


@dataclass
class ResultWrapped:
    f: Callable

    def __post_init__(self):
        self.__signature__ = inspect.signature(self.f)

    def __call__(self, *args, **kwargs):
        try:
            res = self.f(*args, **kwargs)
            return Ok(res)
        except Exception as e:
            import traceback
            trc = traceback.format_exc()
            return Error(ErrorWithTrace(e, trc))



def en_result(f) -> Callable[[Any], Result]:
    return ResultWrapped(f)


