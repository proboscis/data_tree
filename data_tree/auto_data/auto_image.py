from data_tree.coconut.auto_data import AutoData
from data_tree.coconut.convert import ch_splitter
from loguru import logger
#import matplotlib
#matplotlib.use('Qt5Agg')


class AutoImage(AutoData):
    def __init__(self, value, format, solver):
        super().__init__(value, format, solver)
        # TODO add some assertions

    def histogram(self):
        from matplotlib import pyplot as plt
        plt.figure()
        aim = self.convert(type="numpy", arrange="BHWC")
        ary = aim.value
        chs = ch_splitter(aim.format["ch_rpr"])
        logger.info(f"histogram src shape:{ary.shape}")
        logger.info(f"src channels:{','.join(chs)}")
        for i, ch in enumerate(chs):
            plt.hist(ary[:, :, :, i].flatten(), bins=100, label=ch)
        plt.legend()
        plt.show()

    def color_map(self,show=True):
        from matplotlib import pyplot as plt

        aim = self.convert(type="numpy", arrange="BHWC")
        #ary = aim.value
        chs = ch_splitter(aim.format["ch_rpr"])
        figs = []
        for c in chs:
            figs.append(plt.figure())
            plt.imshow(aim.to((f"numpy,uint8,HW,{c},0_255")))
            plt.colorbar()
        if show:
            plt.show()
        return figs

