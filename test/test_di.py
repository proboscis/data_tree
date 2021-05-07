from loguru import logger

from data_tree.di.util import Design
from data_tree.di import util
from data_tree import di
class A:pass
class B:pass

class Tgt:
    def __init__(self,a:A):
        self.a = a

def test_add_binding():

    dbc1 = Design(modules=[util]).bind("a").to_class(A)
    dbc2 = Design(modules=[di]).bind("a").to_instance("hello")
    classes = [A,B,Tgt]
    tgt = dbc1.provided("tgt",classes=classes)
    logger.info(tgt)
    assert isinstance(tgt.a,A)
    assert set((dbc1+dbc2).modules) == {util, di}
    tgt = (dbc1+dbc2).provided("tgt",classes=classes)
    logger.info(tgt)
    assert tgt.a=="hello"
    tgt = (dbc1+dbc2+dbc1).provided("tgt",classes=classes)
    logger.info(tgt)
    assert isinstance(tgt.a,A)


