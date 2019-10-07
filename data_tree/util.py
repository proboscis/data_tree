def batch_index_generator(start, end, batch_size):
    for i in range(start, end, batch_size):
        yield i, min(i + batch_size, end)


def ensure_path_exists(fileName):
    import os
    from os import path, makedirs
    parent = os.path.dirname(fileName)
    if not path.exists(parent) and parent:
        try:
            makedirs(parent)
        except FileExistsError as fee:
            pass

