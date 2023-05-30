import time
from threading import Thread

import rx

from rx.scheduler import NewThreadScheduler
from rx.subject import Subject

from data_tree import logger
from data_tree.image_store_pkg.lmdb import subscribe_as_generator


def test_subscribe_as_generator():
    sub = Subject()

    def src():
        for i in range(1000):
            time.sleep(0.01)
            logger.info(f"src {i}")
            # ok this is blocked by the consumer
            sub.on_next(i)
        sub.on_completed()

    st = Thread(target=src)

    def slowly_consume(g):
        logger.info(f"subscribed")
        for item in g:
            time.sleep(1)
            logger.info(f"recieved {item}")

    from rx import operators as ops
    # ops.buffer_with_count()
    # src = rx.from_iterable(src())
    t = subscribe_as_generator(sub, slowly_consume,100)
    st.start()
    t.join()
