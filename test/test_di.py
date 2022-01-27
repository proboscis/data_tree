from loguru import logger

import archpainter.experiments.config.root_cfg
from pinject_design.di.util import Design
from pinject_design.di import util
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
    tgt = dbc1.provide("tgt", classes=classes)
    logger.info(tgt)
    assert isinstance(tgt.a,A)
    assert set(archpainter.experiments.config.root_cfg.modules) == {util, di}
    tgt = (dbc1+dbc2).provide("tgt", classes=classes)
    logger.info(tgt)
    assert tgt.a=="hello"
    tgt = (dbc1+dbc2+dbc1).provide("tgt", classes=classes)
    logger.info(tgt)
    assert isinstance(tgt.a,A)


