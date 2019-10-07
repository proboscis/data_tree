import pickle
from datetime import datetime
import pandas as pd
from logzero import logger
def load_or_save(path, proc):
    try:
        with open(path, "rb") as f:
            logger.info(f"loading cache at {path}")
            start = datetime.now()
            res = pickle.load(f)
            end = datetime.now()
            dt = end - start
            logger.info(f"loading cache at {path} took {dt.total_seconds():.2f} seconds")
            return res
    except Exception as e:
        logger.info(f"failed to load cache at {path} for {proc.__name__} (cause:{e})")
        res = proc()
        ensure_path_exists(path)
        with open(path, "wb") as f:
            logger.info(f"caching at {path}")
            pickle.dump(res, f)
        return res


def load_or_save_df(path, proc):
    try:
        logger.info(f"loading df cache at {path}")
        start = datetime.now()
        res = pd.read_hdf(path, key="cache")
        end = datetime.now()
        dt = end - start
        logger.info(f"loading df cache at {path} took {dt.total_seconds():.2f} seconds")
        return res
    except Exception as e:
        logger.info(f"failed to load cache at {path} for {proc.__name__} (cause:{e})")
        df: pd.DataFrame = proc()
        logger.info(f"caching at {path}")
        ensure_path_exists(path)
        df.to_hdf(path, key="cache")
        return df

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

