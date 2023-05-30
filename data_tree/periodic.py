from dataclasses import dataclass, field
from typing import Optional

import h5py

from data_tree import logger


@dataclass
class PeriodicReopen:
    tgt_path: str
    f: h5py.File = field(default=None)

    def open(self) -> h5py.File:
        if self.f is None:
            self.f = h5py.File(self.tgt_path, 'a')
        return self.f

    def close(self):
        logger.info(f"closing hdf5")
        f = self.open()
        f.flush()
        f.close()
        self.f = None


@dataclass
class PeriodicReopenV2:
    tgt_path:str
    f:Optional[h5py.File] = field(default=None)
    frequency_to_write:int = 10000
    count:int = 0
    def __enter__(self):
        self.open()
        return self
    def __exit__(self, *args):
        self.close()
        pass

    def update(self):
        self.count += 1
        if self.count >= self.frequency_to_write:
            self.close()
            self.open()
        return self.f


    def open(self) -> h5py.File:
        if self.f is None:
            self.f = h5py.File(self.tgt_path, 'a')
            self.count = 0
        return self.f

    def close(self):
        logger.info(f"closing hdf5")
        f = self.open()
        f.flush()
        f.close()
        self.f = None