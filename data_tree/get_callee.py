def get_callee():
    import inspect
    curf = inspect.currentframe()
    calf = inspect.getouterframes(curf,4)
    return calf[1]