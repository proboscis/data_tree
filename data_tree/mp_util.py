import multiprocessing
import os
import pickle
import sys
import time
from concurrent.futures import Future
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from queue import Empty
from typing import Iterable
from uuid import uuid4
import multiprocessing as mp

from returns.result import safe, Result

from data_tree import logger
from data_tree.util import shared_npy_array_like

# from logzero import logger
# logger = edict(
#    info=print,
#    warning=print,
#    debug=print
# )

GLOBAL_OBJECTS = dict()

from tblib import pickling_support

pickling_support.install()
from tqdm import tqdm


class GlobalHolder:
    def __init__(self, id: str):
        self.id = id

    def release(self):
        del GLOBAL_OBJECTS[self.id]

    @property
    def value(self):
        return GLOBAL_OBJECTS[self.id]


class GlobalProxy:
    def __init__(self, holder: GlobalHolder):
        self._holder = holder

    def __call__(self, *args, **kwargs):
        return self._holder.value(*args, **kwargs)

    def __getitem__(self, item):
        return self._holder.value[item]

    def __getattr__(self, item):
        if item == "_holder":
            raise AttributeError()
        return getattr(self._holder.value, item)


def get_global_holder(data: object) -> GlobalHolder:
    id = str(uuid4())
    GLOBAL_OBJECTS[id] = data
    return GlobalHolder(id)


def get_global_proxy(data: object) -> GlobalProxy:
    return GlobalProxy(get_global_holder(data))


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


def _worker2(self, worker_id, worker_gen_args):
    f = self.worker_generator(*worker_gen_args)
    # a worker dies silently?
    task_count = 0
    self.worker_task_counts[worker_id] = task_count

    while not self.termination_signal.is_set():
        self.worker_states[worker_id] = "loop start"
        try:

            info = dict(
                qsize=self.result_queue.qsize(),
                finished_pending=self.finished_pending.value,
                max_pending_result=self.max_pending_result,

            )
            flag = info["qsize"] <= self.max_pending_result and \
                   info["finished_pending"] <= self.max_pending_result
            info["flag"] = flag
            # how could anyone stop at here??
            self.worker_states[worker_id] = f"check pending result, info:{info}"
            if flag:
                # logger.debug(f"worker-{self.uid}-{worker_id} waiting for task")
                self.worker_states[worker_id] = "waiting task"
                # return # random worker can't even reach here!! although flag is True
                # self.task_get_lock.acquire(timeout=1) # you may not call get from multiple processes at the same time.
                i, task = self.task_queue.get(timeout=1)
                # self.task_get_lock.release()
                self.worker_states[worker_id] = f"got task-{i}"
                # logger.debug(f"worker-{self.uid}-{worker_id} got task {i}")
                # logger.info(f"tasks in queue:{self.task_queue.qsize()}")
                # logger.info(f"results in queue:{self.result_queue.qsize()}")
                # logger.info(f"results pending:{self.finished_pending.value}")
                if task == self.TERMINATION_SIGNAL:
                    self.worker_states[worker_id] = "got termination signal from queue"
                    # logger.info(f"worker got a termination signal. trying to put into result queue")
                    self.result_queue.put((i, self.TERMINATION_SIGNAL))
                    # logger.info(f"termination signal is successfully put to result queue")
                    break
                try:
                    # logger.info(f"worker working on task {i}")
                    self.worker_states[worker_id] = "working on task"
                    result = safe(f)(task)
                    self.worker_states[worker_id] = "putting result to queue"
                    # logger.info(f"worker trying to put result {i} to queue.")
                    self.result_queue.put((i, result))
                    # logger.info(f"worker result {i} is put.")
                    task_count += 1
                    self.worker_task_counts[worker_id] = task_count
                except Exception as task_error:
                    self.worker_states[worker_id] = f"error:{task_error}"
                    logger.error(f"error in {i}th task: {task_error}")
                    logger.error(f"task input:{task}")
                    logger.error(task_error)
                    logger.error(f"terminating parallel task execution due to unhandled error in task")
                    self.termination_signal.set()
                    raise task_error
            else:
                self.worker_states[
                    worker_id] = f"waiting for result room. termination_signal?:{self.termination_signal.is_set()}"
                # logger.debug(f"worker-{self.uid}-{worker_id} waiting for result room")
                time.sleep(1)
        except TimeoutError:
            if self.termination_signal.is_set():
                self.worker_states[worker_id] = f"timeout->termination_signal"
                # logger.debug(f"terminated worker {worker_id} due to signal")
                break
            else:
                logger.debug("timeout. wait again.")
                pass
        except Empty:
            if self.termination_signal.is_set():
                self.worker_states[worker_id] = f"empty->termination_signal"
                # logger.debug(f"terminated worker {worker_id} due to signal")
                break
            else:
                self.worker_states[worker_id] = f"task wait again"
                logger.debug("task not found. wait again.")
                time.sleep(1)
        except Exception as e:
            import traceback
            self.worker_states[worker_id] = f"unexpected error:{e}"
            logger.error(f"worker terminated due to unknown exception ->{e}\n{traceback.format_exc()}")
            raise e
    self.worker_states[worker_id] = "finished"
    # logger.info(f"worker-{self.uid}-{worker_id} finished")


class SequentialTaskParallel2:
    TERMINATION_SIGNAL = "__END__"

    def __init__(self,
                 worker_generator,
                 worker_gen_args,
                 num_worker=8,
                 max_pending_result=100,
                 mp=multiprocessing,
                 use_result_monad=False
                 ):
        self.worker_gen_args = worker_gen_args
        self.worker_generator = worker_generator
        self.num_worker = num_worker
        self.mp = mp
        self.manager = mp.Manager()  # always use manager to avoid wtf errors!
        # you need to master the 'proper' way of using this..

        self.task_queue = self.manager.Queue()
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
        self.sorter_thread = None
        self.workers = None
        self.worker_states = self.manager.dict()

        self.monitor_termination_signal = self.manager.Event()
        self.monitor_thread = None
        from easydict import EasyDict as edict
        self.worker_task_counts = self.manager.dict()
        self.self_object = edict(
            task_queue=self.task_queue,
            result_queue=self.result_queue,
            worker_start_signal=self.worker_start_signal,
            termination_signal=self.termination_signal,
            lock=self.lock,
            task_get_lock=self.task_get_lock,
            finished_pending=self.finished_pending,
            worker_states=self.worker_states,
            worker_generator=self.worker_generator,
            worker_task_counts=self.worker_task_counts,
            max_pending_result=self.max_pending_result,
            uid=self.uid,
            TERMINATION_SIGNAL=self.TERMINATION_SIGNAL,
        )
        self.use_result_monad = use_result_monad

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

    def enqueue_all(self, items):
        for item in items:
            self.enqueue(item)

    def enqueue_termination(self):
        self.enqueue(self.TERMINATION_SIGNAL)

    def start(self, debug=False):
        waiting_index = 0
        self.worker_start_signal.set()

        def monitor():
            import time
            from pprint import pformat
            while not self.monitor_termination_signal.is_set():
                logger.info(f"tasks in queue:{self.task_queue.qsize()}")
                logger.info(f"results in queue:{self.result_queue.qsize()}")
                logger.info(f"finished pending items:{self.finished_pending.value}")
                logger.info(f"max pending items:{self.max_pending_result}")
                logger.info(f"worker_start_signal:{self.worker_start_signal.is_set()}")
                logger.info(f"termination_signal:{self.termination_signal.is_set()}")
                logger.info(f"waiting_result_index::{waiting_index}")
                logger.info(f"worker_states:\n{pformat(dict(self.worker_states.items()))}")
                # logger.info(f"worker_task_counts:\n{pformat(dict(self.worker_task_counts.items()))}")
                # if self.workers is not None:
                #    lives = dict()
                #    for i, w in enumerate(self.workers):
                #        lives[i] = w.is_alive()
                #    logger.info(f"worker_lives:{pformat(lives)}")
                time.sleep(0.1)
            logger.info(f"monitoring thread stopped")

        workers = [self.mp.Process(target=_worker2, args=(self.self_object, i, self.worker_gen_args), daemon=True) for i
                   in
                   range(self.num_worker)]
        for w in workers:
            w.start()
        self.workers = workers

        if debug:
            self.monitor_thread = Thread(target=monitor)
            self.monitor_thread.start()  # holy shit...

        # NEVER EVER ACCESS SHARED RESOURCE BEFORE STARTING A PROCESS! IT WILL CAUSE DEADLOCK
        # if you fork while some one is locking something, forked process will have it locked forever..
        # In this specific case, never start monitor/sorter or anything before worker starts.

        def sorter():
            # sorter is not terminated properly
            while not self.termination_signal.is_set():
                try:
                    # logger.debug(f"sorter waiting for item. items in result queue:{self.result_queue.qsize()}")
                    i, result = self.result_queue.get(timeout=1)
                    # logger.debug(f"sorter got result {i}. waiting for lock.")
                    with self.lock:
                        # logger.debug(f"sort obtained a lock to store item in ")
                        fut = self.results[i]
                        self.finished_pending.value += 1
                    fut.set_result(result)
                except (Empty, TimeoutError):
                    if self.termination_signal.is_set():
                        break
                except Exception as e:
                    import traceback
                    logger.error(f"unexpected error in sorter:{e}\n{traceback.format_exc()}")

            # logger.info(f"sorter finished")

        sorter_thread = Thread(target=sorter, daemon=True)
        sorter_thread.start()
        self.sorter_thread = sorter_thread

        # bar = tqdm()
        try:
            while not self.termination_signal.is_set():
                if waiting_index not in self.results:
                    time.sleep(1)
                    continue
                fut: Future = self.results[waiting_index]
                # logger.info(f"yielder waiting for result {waiting_index}")
                result: Result = fut.result()
                if result == self.TERMINATION_SIGNAL:
                    # logger.info(f"stopping yielder")
                    self.termination_signal.set()
                    break
                with self.lock:
                    del self.results[waiting_index]
                    self.finished_pending.value -= 1
                # bar.update(1)
                if self.use_result_monad:
                    yield result  # yield as is
                else:
                    yield result.unwrap()
                # logger.info(f"yielder yielded:{waiting_index}")
                waiting_index += 1

        except KeyboardInterrupt as ke:
            raise ke
        finally:
            self.worker_start_signal.set()
            self.termination_signal.set()
            # logger.info("finished yielder")

    @contextmanager
    def managed_start(self, debug=False):
        yield self.start(debug=debug)
        self.close()
        # the code is not reaching here.

    def close(self):
        assert self.worker_start_signal.is_set()
        logger.info(f"stopping workers")
        self.termination_signal.set()
        if self.workers is not None:
            for i, w in enumerate(self.workers):
                w.join()
                logger.info(f"joined worker:{i}/{len(self.workers)}")
        logger.info(f"workers are closed.")

        if self.sorter_thread is not None:
            self.sorter_thread.join()
        logger.info(f"sorter is closed.")
        if self.monitor_thread is not None:
            self.monitor_termination_signal.set()
            self.monitor_thread.join()
        logger.info(f"monitoring thread is stopped")


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


class SequentialNumpyTaskParallel:
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


from threading import Thread
# mp.set_start_method("spawn",force=True)



def _mp_worker(f, in_queue, res_dict):
    while True:
        # logger.info(f"worker waiting for result")
        _input = in_queue.get()
        if _input is None:
            return
        _id, _in, _event = _input
        # logger.info(f"worker got a task")

        try:
            res = ("success", f(_in))
        except Exception as e:
            import traceback
            res = ("failure", (e, traceback.format_exc()))
        # logger.info(f"worker set result")
        res_dict[_id] = res
        # logger.info(f"worker set event")
        _event.set()


class MPServer:
    def __init__(self, f):
        self.input_queue = mp.Queue(10)
        self.manager = mp.Manager()
        self.result_dict = self.manager.dict()
        self.f = f

    def query(self, item):
        from uuid import uuid4
        res_id = str(uuid4())
        res_event = self.manager.Event()
        self.input_queue.put((res_id, item, res_event))
        res_event.wait()
        code, data = self.result_dict[res_id]
        del self.result_dict[res_id]
        if code == "success":
            return data
        else:
            raise RuntimeError(f"MPServer remote exception:{data}")

    def run(self):
        import torch.multiprocessing as mp
        self.process = mp.Process(target=_mp_worker, args=(self.f, self.input_queue, self.result_dict))
        self.process.start()

    def stop(self):
        self.input_queue.put(None)


identity = lambda a: a


class MTServer:
    def __init__(self, f, pre_input=identity, post_result=identity):
        import torch.multiprocessing as mp
        self.input_queue = mp.Queue(10)
        self.manager = mp.Manager()
        self.result_dict = self.manager.dict()
        if not isinstance(f, Iterable):
            self.f = [f]
        else:
            self.f = f
        self.pre_input = pre_input
        self.post_result = post_result

    def query(self, item):
        from uuid import uuid4
        res_id = str(uuid4())
        res_event = self.manager.Event()
        _input = self.pre_input(item)
        self.input_queue.put((res_id, _input, res_event))

        res_event.wait()
        code, data = self.result_dict[res_id]
        del self.result_dict[res_id]
        if code == "success":
            return self.post_result(data)
        else:
            raise RuntimeError(f"MTServer remote exception:{data}")

    def run(self):
        for f in self.f:
            self.thread = Thread(target=_mp_worker, args=(f, self.input_queue, self.result_dict))
            self.thread.start()

    def stop(self):
        self.input_queue.put(None)


class Pipeline:
    """
    Problem is an error handling
    """

    def put(self):
        pass

    def get(self):
        pass
