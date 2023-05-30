#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x5f6d3ad5

# Compiled with Coconut version 1.4.3 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import generator_stop
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get("__coconut__")
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules["__coconut__"]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

from multiprocessing import Queue  # from multiprocessing import Queue,Process,shared_memory,RawArray
from multiprocessing import Process  # from multiprocessing import Queue,Process,shared_memory,RawArray
from multiprocessing import shared_memory  # from multiprocessing import Queue,Process,shared_memory,RawArray
from multiprocessing import RawArray  # from multiprocessing import Queue,Process,shared_memory,RawArray
import multiprocessing  # import multiprocessing
import numpy as np  # import numpy as np
  # from loguru import logger
import pickle  # import pickle
import ctypes  # import ctypes
class SMQueue:  # class SMQueue:
    def __init__(self, queue):  #     def __init__(self,queue):
        self._queue = queue  #         self._queue = queue

    def put(self, item):  #     def put(self,item):
        data = pickle.dumps(item)  #         data = pickle.dumps(item)
        shm = shared_memory.SharedMemory(create=True, size=len(data) + 4)  #         shm = shared_memory.SharedMemory(create=True,size=len(data)+4)
        data_size = len(data).to_bytes(4, "big")  #         data_size = len(data).to_bytes(4,"big")

        shm.buf[:4] = data_size  #         shm.buf[:4] = data_size
        shm.buf[4:4 + len(data)] = data  #         shm.buf[4:4+len(data)] = data
        self._queue.put(shm)  #         self._queue.put(shm)
        shm.close()  #         shm.close()

    def get(self):  #     def get(self):
        shm = self._queue.get()  #         shm = self._queue.get()
        data_size = int.from_bytes(shm.buf[:4], "big")  #         data_size =int.from_bytes(shm.buf[:4],"big")
        data = pickle.loads(shm.buf[4:4 + data_size])  #         data = pickle.loads(shm.buf[4:4+data_size])
        shm.close()  #         shm.close()
        shm.unlink()  #         shm.unlink()
        return data  #         return data

class SMQueue2:  # unusable as a queue. RawArray cannot be put into queue  # class SMQueue2: # unusable as a queue. RawArray cannot be put into queue
    def __init__(self, queue):  #     def __init__(self,queue):
        self._queue = queue  #         self._queue = queue

    def put(self, item):  #     def put(self,item):
        data = pickle.dumps(item)  #         data = pickle.dumps(item)
        shm = RawArray(ctypes.c_byte, data)  #         shm = RawArray(ctypes.c_byte,data)
#!c array should only be shared through inheritance.

#shm = shared_memory.SharedMemory(create=True,size=len(data)+4)
#data_size = len(data).to_bytes(4,"big")
        self._queue.put(shm)  #         self._queue.put(shm)

    def get(self):  #     def get(self):
        shm = self._queue.get()  #         shm = self._queue.get()
        data = pickle.loads(shm)  #         data = pickle.loads(shm)
        return data  #         return data


def inserter(q):  # def inserter(q):
#q = SMQueue(q) # using this queue is 10x faster
    ary = np.arange(10000 * 10000 * 3)  #     ary = np.arange(10000*10000*3)
    for i in range(1000):  #     for i in range(1000):
        q.put(ary)  #         q.put(ary)
        logger.info("put item:{_coconut_format_0}".format(_coconut_format_0=(i)))  #         logger.info(f"put item:{i}")
    logger.info("enqueued everything".format())  #     logger.info(f"enqueued everything")

def worker(q):  # def worker(q):
#q = SMQueue(q)
    processed = 0  #     processed = 0
    while processed < 1000:  #     while processed < 1000:
        item = q.get()  #         item = q.get()
        processed += 1  #         processed += 1
        logger.info("got an item.:{_coconut_format_0}".format(_coconut_format_0=(processed)))  #         logger.info(f"got an item.:{processed}")
    logger.info("finished processing all items".format())  #     logger.info(f"finished processing all items")

def test_mp_queue():  # def test_mp_queue():
#queue = Queue()
#queue = Queue(10) # it just takes soo much time to 'get' large item
    queue = multiprocessing.Manager().Queue()  #works, but cannot even put an object.  #     queue = multiprocessing.Manager().Queue() #works, but cannot even put an object.
    _in = Process(target=inserter, args=(queue,))  #     _in = Process(target=inserter,args=(queue,))
    _work = Process(target=worker, args=(queue,))  #     _work = Process(target=worker,args=(queue,))
    _in.start()  #     _in.start()
    _work.start()  #     _work.start()
    _work.join()  #     _work.join()
    _in.join()  #     _in.join()
