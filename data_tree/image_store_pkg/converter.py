from dataclasses import dataclass

from data_tree.image_store import ImageStore
from data_tree.image_store_creator import Hdf5BytesKvsCreator
from data_tree.wandb_util.artifact_identifier import ArtifactMetadata


@dataclass
class ImageStoreToHdf5Converter:
    hdf5_bytes_kvs_creator: Hdf5BytesKvsCreator

    def save(self, src: ImageStore, dst: ArtifactMetadata):
        def gen():
            for key,img in src.mp_bytes_pair_iterator():
                yield key,img
        #gen = ((key, auto("image,RGB,RGB")(img)) for key,img in src.sorted_series())# loading can be done in 7000/s
        self.hdf5_bytes_kvs_creator.create(dst, gen(), total=len(src.keys()))