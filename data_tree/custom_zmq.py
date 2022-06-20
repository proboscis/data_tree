import io
from threading import Thread

import zmq
from PIL import Image
from loguru import logger

from data_tree.coconut.monad import try_monad, Failure, Success


class ZMQRemoteException(Exception):
    def __init__(self, cause, trace):
        super(ZMQRemoteException, self).__init__()
        self.cause = cause
        self.trace = trace


class ZMQ_FunctionServer:
    def __init__(self, f, address: str):
        """
        :param f: Image->Image
        :param address:
        """
        self.f = try_monad(f)
        self.address = address

    def serve(self, threaded=True,daemon=False):
        """
            now this server works!
            :return:
            """

        context = zmq.Context()
        self.context = context
        socket = context.socket(zmq.REP)
        socket.bind(self.address)
        self.socket = socket

        def _serve():
            while True:
                pyobj = socket.recv_pyobj()
                res = self.f(pyobj)
                if isinstance(res, Failure):
                    t = ("failure", (str(res.exception), str(res.trace)))
                    logger.error(f"failed to serve:{res.exception},{res.trace}")
                elif isinstance(res, Success):
                    t = ("success", res.result)
                else:
                    raise RuntimeError(f"unexpected result from given f:{self.f}")
                socket.send_pyobj(t)

        if threaded:
            server_thread = Thread(target=_serve,daemon = daemon)
            server_thread.start()
        else:
            _serve()

    def stop(self):
        self.socket.close()


class ZMQ_FileSaveServer(ZMQ_FunctionServer):

    def on_data(self, data):
        path, value = data
        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception as e:
            return "failed to write data to {path}. cause:{e}"
        return "success"

    def __init__(self, address: str):
        super().__init__(self.on_data, address)


class ZMQ_Client:
    def __init__(self, address):
        self.address = address
        self.context = zmq.Context()

    def send_pyobj(self, data):
        logger.debug(f"connecting to f{self.address}")
        socket = self.context.socket(zmq.REQ)
        socket.connect(self.address)
        logger.debug(f"sending data to {self.address}")
        socket.send_pyobj(data)
        res = socket.recv_pyobj()
        if isinstance(res, Failure):
            raise ZMQRemoteException(res.exception, res.trace)
        logger.debug(f"result:{res}")
        socket.close()
        return res



    def _send_image(self, img: Image):
        raise RuntimeError(f"dont use this now.")
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="PNG")
        to_send = byte_arr.getvalue()
        self.send_pyobj(path, to_send)