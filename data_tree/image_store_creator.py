import gc
import io
import os
from dataclasses import dataclass
from typing import Iterable, Tuple, List

import PIL
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from data_tree.image_store import ImageStore, ImageStoreCreator
from data_tree.periodic import PeriodicReopen
from data_tree.util import ensure_path_exists
from data_tree.wandb_util.artifact_identifier import ArtifactMetadata, ArtifactIdentifier
from data_tree.wandb_util.path_artifact import ManagedPathArtifactLogger, ManagedPathArtifactLoader


@dataclass
class Hdf5BytesKvsCreator:
    managed_path_artifact_logger: ManagedPathArtifactLogger

    def create(self, metadata: ArtifactMetadata, key_data_pairs: Iterable[Tuple[str, bytes]], total: int = None):
        dst_path = self.managed_path_artifact_logger.log_artifact_path(
            metadata=metadata
        )
        hdf_path = os.path.join(dst_path, 'data.hdf5')
        ensure_path_exists(hdf_path)
        hf = PeriodicReopen(hdf_path)
        for i, (key, data) in enumerate(
                tqdm(key_data_pairs, desc=f'storing data to hdf5 at {hdf_path}', total=total)):
            assert isinstance(data, bytes)
            data = np.asarray(data)
            f = hf.open()
            f.create_dataset(key, data=data)  # this might consume memories so much
            if i % 10000 == 0:
                hf.close() # this periodic close is required to prevent memory over usage.
                gc.collect()
        hf.close()


@dataclass
class Hdf5ImageStore(ImageStore):
    hdf5_path: str

    def __getitem__(self, item):
        with h5py.File(self.hdf5_path, 'r') as f:
            return Image.open(io.BytesIO(np.array(f[item])))

    def keys(self) -> List[str]:
        with h5py.File(self.hdf5_path, 'r') as f:
            return list(tqdm(f.keys(), desc='reading keys from hdf5..'))


@dataclass
class Hdf5ManagedImageStoreFactory:
    managed_path_artifact_loader: ManagedPathArtifactLoader

    def get(self, identifier: ArtifactIdentifier):
        path = self.managed_path_artifact_loader.get_path_identifier(identifier)
        return Hdf5ImageStore(os.path.join(path, 'data.hdf5'))


@dataclass
class Hdf5ImageStoreCreator(ImageStoreCreator):
    hdf5_bytes_kvs_creator: Hdf5BytesKvsCreator
    hdf5_output_artifact_metadata: ArtifactMetadata

    def create(self, key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]], total=None):
        from data_tree import auto
        def gen():
            for k, img in key_image_pairs:
                png_bytes = auto("image,RGB,RGB")(img).to('png_bytes')
                yield k, png_bytes

        self.hdf5_bytes_kvs_creator.create(self.hdf5_output_artifact_metadata, gen(), total=total)
