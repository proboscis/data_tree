import sys
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Callable, Any, Iterator, cast

import better_exceptions
import zmq

from returns.result import Failure as RFailure

from data_tree import logger
from data_tree.custom_zmq import ZMQRemoteException


@dataclass
class ZMQ_GeneratorServer:
    address: str
    generator_function: Callable[[Any], Iterator[Any]]
    TERMINATION_SIGNAL = "__END__"
    ACK = "__ACK__"

    def serve(self, threaded=True, daemon=False):
        """
            now this server works!
            :return:
            """

        context = zmq.Context()
        self.context = context

        def _serve():
            socket = context.socket(zmq.REP)
            socket.bind(self.address)
            send_queue = Queue()
            pyobj = socket.recv_pyobj()

            def session(queue):
                logger.info(f"starting zmq session")
                header, item = queue.get()
                while header is not self.TERMINATION_SIGNAL and header != -1:
                    socket.send_pyobj((header, item))
                    ack = socket.recv_pyobj()
                    assert ack == self.ACK
                    header, item = queue.get()
                logger.info(f"sending termination signal")
                socket.send_pyobj((header, item))
                logger.info(f"finished zmq session")

            session_thread = Thread(target=session, args=(send_queue,))
            session_thread.start()
            try:
                for i, item in enumerate(self.generator_function(pyobj)):
                    send_queue.put((i, item))
                send_queue.put((self.TERMINATION_SIGNAL, None))
            except Exception as e:
                trc = better_exceptions.format_exception(*sys.exc_info())
                logger.warning(f"error while serving.:{e}")
                logger.warning("\n".join(trc))
                from returns.result import Failure
                send_queue.put((-1, Failure(sys.exc_info())))
            try:
                session_thread.join()
            except Exception as e:
                socket.close()
                raise e

        def serve_forever():
            while True:
                try:
                    _serve()
                except Exception as e:
                    trc = better_exceptions.format_exception(*sys.exc_info())
                    logger.warning(f"error while serve forever...:{e}")
                    logger.warning("\n".join(trc))

        if threaded:
            server_thread = Thread(target=serve_forever, daemon=daemon)
            server_thread.start()
        else:
            serve_forever()

    def stop(self):
        self.socket.close()


class ZMQ_GeneratorClient:
    def __init__(self, address: str):
        self.address = address
        self.context = zmq.Context()

    def send_pyobj(self, data) -> Iterator[Any]:
        """
        :param data: any python object to send to the server
        :return: iterator of returned values.
        """
        logger.debug(f"connecting to {self.address}")
        socket = self.context.socket(zmq.REQ)
        socket.connect(self.address)
        logger.debug(f"sending data to {self.address}")
        socket.send_pyobj(data)
        header, item = socket.recv_pyobj()
        while header != ZMQ_GeneratorServer.TERMINATION_SIGNAL and header != -1:
            socket.send_pyobj(ZMQ_GeneratorServer.ACK)
            yield item
            header, item = socket.recv_pyobj()
        # dont ack on termination signal
        if header == -1:
            item = cast(RFailure, item)
            typ, val, trb = item.failure()
            raise ZMQRemoteException(val, trb)
        socket.close()
