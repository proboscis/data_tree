import multiprocessing
import pickle
import sys
import time
from concurrent.futures import Future
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from queue import Empty
from threading import Thread
from uuid import uuid4
import os

import logzero
from loguru import logger

from six import reraise

from data_tree.util import shared_npy_array_like

GLOBAL_OBJECTS = dict()

from tblib import pickling_support

pickling_support.install()
from tqdm.autonotebook import tqdm


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


SUCCESS = "success"
FAILURE = "failure"


def _mp_wrapper(tgt_function):
    def inner(__result_pipe, *args, **kwargs):
        uid = uuid4()
        try:
            logger.info(f"running target function on remote process:{os.getpid()}")
            result = tgt_function(*args, **kwargs)
            __result_pipe.send((SUCCESS, result))
        except Exception as e:
            logger.warning(f"serializing remote exception:{e}")
            logger.warning(f"params:{args},{kwargs}")
            serialized = pickle.dumps(sys.exc_info())
            __result_pipe.send((FAILURE, serialized))
        logger.debug(f"finished remote process:{os.getpid()}")

    return inner


def mp_run(target, **process_params) -> Future:
    """
    runs target function on another process.
    returns a future which contains result or serialized exception.
    You will never miss remote process exception again.
    :param target:
    :param process_params: (e.g. args,kwargs,daemon,etc..)
    :return:
    """
    result_future = Future()
    parent, child = Pipe()
    process_params = process_params.copy()
    if "args" in process_params:
        args = (child, *process_params["args"])
        del process_params["args"]
    else:
        args = (child,)

    p = Process(target=_mp_wrapper(target), args=args, **process_params)
    p.start()

    def result_handler():
        case, result = parent.recv()
        if case is SUCCESS:
            result_future.set_result(result)
        elif case is FAILURE:
            result_future.set_exception(pickle.loads(result))

    daemon = process_params.get("daemon", False)
    handling_thread = Thread(target=result_handler, daemon=daemon)
    handling_thread.start()
    return result_future


def mt_run(target, **thread_params):
    """
    runs target function on another process.
    returns a future which contains result or serialized exception.
    You will never miss remote process exception again.
    :param target:
    :param process_params: (e.g. args,kwargs,daemon,etc..)
    :return:
    """
    result_future = Future()
    thread_params = thread_params.copy()

    def _wrapper(*args, **kwargs):
        try:
            result_future.set_result(target(*args, **kwargs))
        except Exception as e:
            result_future.set_exception(e)

    t = Thread(target=_wrapper, **thread_params)
    t.start()
    return result_future


class SequentialTaskParallel:
    def __init__(self, worker_generator, num_worker=8, max_pending_result=100):
        self.worker_generator = worker_generator
        self.num_worker = num_worker
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.worker_start_signal = multiprocessing.Event()
        self.termination_signal = multiprocessing.Event()
        self.lock = multiprocessing.Lock()
        self.task_i = 0
        self.results = dict()
        self.finished_pending = 0
        self.max_pending_result = max_pending_result

    def enqueue(self, item):
        with self.lock:
            fut = Future()
            self.results[self.task_i] = fut
            self.task_queue.put((self.task_i, item))
            self.task_i += 1

    def start(self, total=None):
        self.worker_start_signal.set()

        def worker():
            f = self.worker_generator()
            while not self.termination_signal.is_set():
                try:
                    logger.debug(f"worker waiting for start signal")
                    self.worker_start_signal.wait()
                    i, task = self.task_queue.get(timeout=1)
                    result = f(task)
                    self.result_queue.put((i, result))
                except TimeoutError:
                    if self.termination_signal.is_set():
                        break
                except Empty:
                    pass
            logger.info(f"worker finished")

        workers = [multiprocessing.Process(target=worker, daemon=True) for _ in range(self.num_worker)]
        for w in workers:
            w.start()

        def sorter():
            while not self.termination_signal.is_set():
                try:
                    i, result = self.result_queue.get()
                    logger.debug(f"sorter got result {i}")
                    with self.lock:
                        fut = self.results[i]
                        fut.set_result(result)
                        self.finished_pending += 1
                        if self.finished_pending > self.max_pending_result:
                            self.worker_start_signal.clear()
                except TimeoutError:
                    if self.termination_signal.is_set():
                        break
            logger.info(f"sorter finished")

        sorter_thread = Thread(target=sorter, daemon=True)
        sorter_thread.start()

        waiting_index = 0
        bar = tqdm()
        try:
            while not self.termination_signal.is_set():
                if waiting_index not in self.results:
                    time.sleep(1)
                    continue
                fut: Future = self.results[waiting_index]
                # logger.info(f"yielder waiting for result {waiting_index}")
                result = fut.result()
                with self.lock:
                    del self.results[waiting_index]
                    self.finished_pending -= 1
                    if self.finished_pending < self.max_pending_result:
                        self.worker_start_signal.set()
                bar.update(1)
                yield result
                # logger.info(f"yielder yielded:{waiting_index}")
                waiting_index += 1

                if total is not None:
                    if waiting_index == total:
                        logger.info(f"stopping yielder")
                        break
        except KeyboardInterrupt as ke:
            raise ke
        finally:
            self.worker_start_signal.set()
            self.termination_signal.set()
            logger.info("finished yielder")


class SequentialTaskParallel2:
    TERMINATION_SIGNAL = "__END__"

    def __init__(self, worker_generator, num_worker=8, max_pending_result=100):
        self.worker_generator = worker_generator
        self.num_worker = num_worker
        self.manager = multiprocessing.Manager()  # always use manager to avoid wtf errors!
        self.task_queue = self.manager.Queue(max_pending_result)
        self.result_queue = self.manager.Queue()
        self.worker_start_signal = self.manager.Event()
        self.termination_signal = self.manager.Event()
        self.lock = self.manager.Lock()
        self.task_get_lock = self.manager.Lock()
        self.task_i = 0
        self.results = dict()
        self.finished_pending = self.manager.Value(int, 0)
        self.max_pending_result = max_pending_result
        self.uid = str(uuid4())[:6]
        # logger = logzero.setup_logger(f"parallel_task({self.uid})")

    def enqueue(self, item):
        current_task_i = self.task_i
        # logger.debug(f"waiting for task put lock:{current_task_i}")
        with self.lock:
            task_item = (self.task_i, item)
            fut = Future()
            self.results[self.task_i] = fut
            self.task_i += 1

        # with self.task_get_lock:
        # without a lock, you lose items..
        self.task_queue.put(task_item)
        # logger.debug(f"task put with index:{self.task_i}. before:{current_task_i}")

    def enqueue_termination(self):
        self.enqueue(self.TERMINATION_SIGNAL)

    def start(self):
        self.worker_start_signal.set()

        def worker(worker_id):
            f = self.worker_generator()
            # a worker dies silently?
            while not self.termination_signal.is_set():
                try:
                    if self.result_queue.qsize() <= self.max_pending_result and \
                            self.finished_pending.value <= self.max_pending_result:
                        # logger.debug(f"worker-{self.uid}-{worker_id} waiting for task")
                        # self.task_get_lock.acquire(timeout=1) # you may not call get from multiple processes at the same time.
                        i, task = self.task_queue.get(timeout=1)
                        # self.task_get_lock.release()
                        # logger.debug(f"worker-{self.uid}-{worker_id} got task {i}")
                        # logger.info(f"tasks in queue:{self.task_queue.qsize()}")
                        # logger.info(f"results in queue:{self.result_queue.qsize()}")
                        # logger.info(f"results pending:{self.finished_pending.value}")
                        if task == self.TERMINATION_SIGNAL:
                            self.result_queue.put((i, self.TERMINATION_SIGNAL))
                            break
                        try:
                            result = f(task)
                            self.result_queue.put((i, result))
                        except Exception as task_error:
                            logger.error(f"error in {i}th task: {task_error}")
                            logger.error(f"task input:{task}")
                            logger.error(task_error)
                            logger.error(f"terminating parallel task execution due to unhandled error in task")
                            self.termination_signal.set()
                            raise task_error
                    else:
                        # logger.debug(f"worker-{self.uid}-{worker_id} waiting for result room")
                        time.sleep(1)
                except TimeoutError:
                    if self.termination_signal.is_set():
                        logger.debug("terminated worker due to signal")
                        break
                    else:
                        logger.debug("timeout. wait again.")
                        pass
                except Empty:
                    if self.termination_signal.is_set():
                        logger.debug("terminated worker due to signal")
                        break
                    else:
                        logger.debug("task not found. wait again.")
                        time.sleep(1)
                except Exception as e:
                    logger.error(e)
                    raise e
            logger.info(f"worker-{self.uid}-{worker_id} finished")

        workers = [multiprocessing.Process(target=worker, args=(i,), daemon=True) for i in range(self.num_worker)]
        for w in workers:
            w.start()

        def sorter():
            while not self.termination_signal.is_set():
                try:
                    # logger.debug(f"sorter waiting for item. items in result queue:{self.result_queue.qsize()}")
                    i, result = self.result_queue.get()
                    # logger.debug(f"sorter got result {i}")
                    with self.lock:
                        # logger.debug(f"sort obtained a lock to store item in ")
                        fut = self.results[i]
                        self.finished_pending.value += 1
                    fut.set_result(result)
                except TimeoutError:
                    if self.termination_signal.is_set():
                        break

            logger.info(f"sorter finished")

        sorter_thread = Thread(target=sorter, daemon=True)
        sorter_thread.start()

        waiting_index = 0
        bar = tqdm()
        try:
            while not self.termination_signal.is_set():
                if waiting_index not in self.results:
                    time.sleep(1)
                    continue
                fut: Future = self.results[waiting_index]
                # logger.info(f"yielder waiting for result {waiting_index}")
                result = fut.result()
                if result == self.TERMINATION_SIGNAL:
                    logger.info(f"stopping yielder")
                    self.termination_signal.set()
                    break
                with self.lock:
                    del self.results[waiting_index]
                    self.finished_pending.value -= 1
                bar.update(1)
                yield result
                # logger.info(f"yielder yielded:{waiting_index}")
                waiting_index += 1

        except KeyboardInterrupt as ke:
            raise ke
        finally:
            self.worker_start_signal.set()
            self.termination_signal.set()
            logger.info("finished yielder")

    @contextmanager
    def managed_start(self):
        yield self.start()
        logger.info(f"stopping workers")
        self.termination_signal.set()


class SharedBufferPool:
    def __init__(self, sample):
        self.sample = sample
        self.manager = multiprocessing.Manager()
        self.buffers = self.manager.list()
        self.lock = multiprocessing.Lock()

    def get_a_shared_buffer(self):
        with self.lock:
            if self.buffers:
                return self.buffers.pop()
            else:
                shared_npy_array_like(self.sample)

    def return_shared_buffer(self, buffer):
        with self.lock:
            self.buffers.append(buffer)

class SequentialTaskParallel3:
    """
    parallel task processing of numpy-to-numpy task
    """
    TERMINATION_SIGNAL = "__END__"

    def __init__(self, worker_generator,
                 input_sample,
                 output_sample,
                 num_worker=8, max_pending_result=100):
        self.worker_generator = worker_generator
        self.num_worker = num_worker
        self.manager = multiprocessing.Manager()  # always use manager to avoid wtf errors!
        self.task_queue = self.manager.Queue(max_pending_result)
        self.result_queue = self.manager.Queue()
        self.worker_start_signal = self.manager.Event()
        self.termination_signal = self.manager.Event()
        self.lock = self.manager.Lock()
        self.task_get_lock = self.manager.Lock()
        self.task_i = 0
        self.results = dict()
        self.finished_pending = self.manager.Value(int, 0)
        self.max_pending_result = max_pending_result
        self.uid = str(uuid4())[:6]
        self.task_buffer = SharedBufferPool(input_sample)
        self.result_buffer = SharedBufferPool(output_sample)

        # logger = logzero.setup_logger(f"parallel_task({self.uid})")

    def enqueue(self, item):
        if item != self.TERMINATION_SIGNAL:
            buffer = self.task_buffer.get_a_shared_buffer()
            buffer[:] = item
            item = buffer
        current_task_i = self.task_i
        # logger.debug(f"waiting for task put lock:{current_task_i}")
        with self.lock:
            task_item = (self.task_i, item)
            fut = Future()
            self.results[self.task_i] = fut
            self.task_i += 1

        # with self.task_get_lock:
        # without a lock, you lose items..
        self.task_queue.put(task_item)
        # logger.debug(f"task put with index:{self.task_i}. before:{current_task_i}")

    def enqueue_termination(self):
        self.enqueue(self.TERMINATION_SIGNAL)

    def start(self):
        self.worker_start_signal.set()

        def worker(worker_id):
            f = self.worker_generator()
            # a worker dies silently?
            while not self.termination_signal.is_set():
                try:
                    if self.result_queue.qsize() <= self.max_pending_result and \
                            self.finished_pending.value <= self.max_pending_result:
                        # logger.debug(f"worker-{self.uid}-{worker_id} waiting for task")
                        # self.task_get_lock.acquire(timeout=1) # you may not call get from multiple processes at the same time.
                        i, task = self.task_queue.get(timeout=1)
                        # self.task_get_lock.release()
                        # logger.debug(f"worker-{self.uid}-{worker_id} got task {i}")
                        # logger.info(f"tasks in queue:{self.task_queue.qsize()}")
                        # logger.info(f"results in queue:{self.result_queue.qsize()}")
                        # logger.info(f"results pending:{self.finished_pending.value}")
                        if task == self.TERMINATION_SIGNAL:
                            self.result_queue.put((i, self.TERMINATION_SIGNAL))
                            break
                        try:
                            result = f(task)
                            buf = self.result_buffer.get_a_shared_buffer()
                            buf[:] = result
                            self.task_buffer.return_shared_buffer(task)
                            self.result_queue.put((i, buf[:]))
                        except Exception as task_error:
                            logger.error(f"error in {i}th task: {task_error}")
                            logger.error(f"task input:{task}")
                            logger.error(task_error)
                            logger.error(f"terminating parallel task execution due to unhandled error in task")
                            self.termination_signal.set()
                            raise task_error
                    else:
                        # logger.debug(f"worker-{self.uid}-{worker_id} waiting for result room")
                        time.sleep(1)
                except TimeoutError:
                    if self.termination_signal.is_set():
                        logger.debug("terminated worker due to signal")
                        break
                    else:
                        logger.debug("timeout. wait again.")
                        pass
                except Empty:
                    if self.termination_signal.is_set():
                        logger.debug("terminated worker due to signal")
                        break
                    else:
                        logger.debug("task not found. wait again.")
                        time.sleep(1)
                except Exception as e:
                    logger.error(e)
                    raise e
            logger.info(f"worker-{self.uid}-{worker_id} finished")

        workers = [multiprocessing.Process(target=worker, args=(i,), daemon=True) for i in range(self.num_worker)]
        for w in workers:
            w.start()

        def sorter():
            while not self.termination_signal.is_set():
                try:
                    # logger.debug(f"sorter waiting for item. items in result queue:{self.result_queue.qsize()}")
                    i, result = self.result_queue.get()
                    # logger.debug(f"sorter got result {i}")
                    with self.lock:
                        # logger.debug(f"sort obtained a lock to store item in ")
                        fut = self.results[i]
                        self.finished_pending.value += 1
                    fut.set_result(result.copy())
                    self.result_buffer.return_shared_buffer(result)
                except TimeoutError:
                    if self.termination_signal.is_set():
                        break
            logger.info(f"sorter finished")

        sorter_thread = Thread(target=sorter, daemon=True)
        sorter_thread.start()

        waiting_index = 0
        bar = tqdm()
        try:
            while not self.termination_signal.is_set():
                if waiting_index not in self.results:
                    time.sleep(1)
                    continue
                fut: Future = self.results[waiting_index]
                # logger.info(f"yielder waiting for result {waiting_index}")
                result = fut.result()
                if result == self.TERMINATION_SIGNAL:
                    logger.info(f"stopping yielder")
                    self.termination_signal.set()
                    break
                with self.lock:
                    del self.results[waiting_index]
                    self.finished_pending.value -= 1
                bar.update(1)
                yield result
                # logger.info(f"yielder yielded:{waiting_index}")
                waiting_index += 1

        except KeyboardInterrupt as ke:
            raise ke
        finally:
            self.worker_start_signal.set()
            self.termination_signal.set()
            logger.info("finished yielder")

    @contextmanager
    def managed_start(self):
        yield self.start()
        logger.info(f"stopping workers")
        self.termination_signal.set()
