from data_tree.coconut.auto_data import AutoData
from data_tree.coconut.convert import ch_splitter
from loguru import logger
#import matplotlib
#matplotlib.use('Qt5Agg')


class AutoImage(AutoData):
    def __init__(self, value, format, solver_getter):
        super().__init__(value, format, solver_getter)
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

    def color_map(self,show=True,value_range="0_1",vmin=None,vmax=None):
        from matplotlib import pyplot as plt

        aim = self.convert(type="numpy", arrange="BHWC")
        #ary = aim.value
        chs = ch_splitter(aim.format["ch_rpr"])
        figs = []
        for i,c in enumerate(chs):
            figs.append(plt.figure())
            tgt = aim.to(f"numpy,float32,HW,{c},{value_range}")
            plt.imshow(tgt,vmin=vmin,vmax=vmax)
            #plt.imshow(aim.to((f"numpy,uint8,HW,{c},0_255")))
            plt.colorbar()
        if show:
            plt.show()
        return figs

    def as_spectrum_img(self):
        return self.convert("spectrum").cast("numpy,float64,HW,L,None")
