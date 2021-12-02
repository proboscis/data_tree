import cloudpickle
from cytoolz import valmap
from expression.collections import Map
from loguru import logger
from returns.result import safe, Success, Result, Failure
from tabulate import tabulate



def check_picklable(tgt:dict):
    cloud_dumps_try = safe(cloudpickle.dumps)
    if isinstance(cloud_dumps_try(tgt),Failure):
        target_check = Map.of(**valmap(cloud_dumps_try, tgt))
        logger.error("failed pickling:\n" + tabulate(target_check.filter(lambda k, v: isinstance(v,Failure)).to_list()))
        raise RuntimeError("this object is not picklable. check the error messages above")