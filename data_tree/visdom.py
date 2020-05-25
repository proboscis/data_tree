from lazy_object_proxy import Proxy

from data_tree.config import CONFIG
def get_visdom():
    import visdom
    return visdom.Visdom(**CONFIG.visdom)


VISDOM = Proxy(get_visdom)