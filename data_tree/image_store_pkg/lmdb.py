import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Iterable, Tuple, List

import PIL
import lmdb
from PIL import Image
from loguru import logger
from tqdm import tqdm

from archpainter.wandb_util.artifact_logger import ArtifactMetadata, ArtifactIdentifier
from archpainter.wandb_util.path_artifact import ManagedPathArtifactLogger, ManagedPathArtifactLoader
from data_tree import auto, series
from data_tree.format_converter import FormatConverter
from data_tree.image_store import ImageStoreCreator, convert_images_to_bytes_mp, ImageStore, MillionImageFileCreator, \
    MergedImageStore
from data_tree.lmdb_util import LmdbCreator
from data_tree.storage_manager import FileStorageManager
from data_tree.util import ensure_path_exists, Pickled


@dataclass
class LmdbImageStoreCreator(ImageStoreCreator):
    lmdb_creator: LmdbCreator
    lmdb_output_destination: str
    lmdb_image_format: str
    lmdb_size_in_bytes: int

    def create(self, key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]], total=None):
        def gen():
            for k, img in key_image_pairs:
                bytes = auto("image,RGB,RGB")(img).to(f"{self.lmdb_image_format}_bytes")
                yield k, bytes

        self.lmdb_creator.create(gen(), self.lmdb_output_destination, size_in_bytes=self.lmdb_size_in_bytes,
                                 total=total)


@dataclass
class MultiLmdbImageStoreCreator:
    lmdb_creator: LmdbCreator

    def create(self, key_image_pairs: Iterable[Tuple[str, bytes]], dst_path, total=None, max_size_per_lmdb=2 ** 24,
               bar=None):
        # 16GB is the limitation on abci lustre fs
        i = 0

        def get_path():
            nonlocal i
            res = os.path.join(dst_path, f"{i}")
            i += 1
            return res

        bar = tqdm(desc=f"creating multi lmdb storage at {dst_path}", total=total) if bar is None else bar
        while key_image_pairs is not None:
            dst = get_path()
            ensure_path_exists(dst)
            key_image_pairs = self.lmdb_creator.create(
                key_image_pairs,
                dst=dst,
                size_in_bytes=max_size_per_lmdb,
                total=total,
                bar=bar
            )


@dataclass
class MultiLmdbImageStoreArtifactCreator:
    multi_lmdb_image_store_creator: MultiLmdbImageStoreCreator
    managed_path_artifact_logger: ManagedPathArtifactLogger

    def create(self,
               metadata: ArtifactMetadata,
               key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]],
               format_converter: FormatConverter = FormatConverter('png'),
               bar=None,
               max_size_per_lmdb: int = 2 ** 24,
               total=None
               ):
        return self.create_from_bytes(
            metadata=metadata,
            key_bytes_pairs=convert_images_to_bytes_mp(key_image_pairs, format_converter),
            bar=bar,
            max_size_per_lmdb=max_size_per_lmdb,
            total=total
        )

    def create_from_bytes(self,
                          metadata: ArtifactMetadata,
                          key_bytes_pairs: Iterable[Tuple[str, bytes]],
                          bar=None,
                          max_size_per_lmdb: int = 2 ** 24,
                          total=None
                          ):
        dst_path = self.managed_path_artifact_logger.log_artifact_path(metadata)
        self.multi_lmdb_image_store_creator.create(
            key_image_pairs=key_bytes_pairs,
            dst_path=dst_path,
            bar=bar,
            max_size_per_lmdb=max_size_per_lmdb,
            total=total
        )
        return dst_path


@dataclass
class CreateLmdbImageStoreWithArtifactFromIterables:
    managed_path_artifact_logger: ManagedPathArtifactLogger
    lmdb_creator: LmdbCreator

    def create(self,
               key_imgs_pair: Iterable,
               output_artifact_name: str,
               size_in_bytes: int,
               format_converter: FormatConverter,
               total=None
               ):
        with Pool() as pool:
            converted = pool.imap_unordered(format_converter, key_imgs_pair)
            return self.create_from_bytes(
                converted, output_artifact_name, size_in_bytes, total=total
            )

    def create_from_bytes(self,
                          key_bytes_pair: Iterable,
                          output_artifact_name: str,
                          size_in_bytes: int,
                          total: int = None
                          ):
        dst_path = self.managed_path_artifact_logger.log_path(
            conditions=dict(name=output_artifact_name),
            name=output_artifact_name,
            type="lmdb_image_store",
        )

        self.lmdb_creator.create(key_bytes_pair, dst_path, size_in_bytes=size_in_bytes, total=total)
        return dst_path


@dataclass
class CreateLmdbImageStoreWithArtifact:
    managed_path_artifact_logger: ManagedPathArtifactLogger
    create_lmdb_image_store_with_artifact_from_iterables: CreateLmdbImageStoreWithArtifactFromIterables

    def create(self,
               src: ImageStore,
               output_artifact_name: str,
               image_format: str,
               size_in_bytes: int
               ):
        keys = series(src.keys())
        images = keys.map(src.__getitem__)
        return self.create_lmdb_image_store_with_artifact_from_iterables.create(
            key_imgs_pair=keys.zip(images),
            output_artifact_name=output_artifact_name,
            format_converter=FormatConverter(image_format),
            size_in_bytes=size_in_bytes
        )


@dataclass
class LmdbImageStoreCreationWithArtifact(ImageStoreCreator):
    lmdb_output_artifact_name: str
    lmdb_size_in_bytes: int
    create_lmdb_image_store_with_artifact_from_iterables: CreateLmdbImageStoreWithArtifactFromIterables
    lmdb_format_converter: FormatConverter

    def create(self, key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]], total=None):
        return self.create_lmdb_image_store_with_artifact_from_iterables.create(
            key_imgs_pair=key_image_pairs,
            output_artifact_name=self.lmdb_output_artifact_name,
            size_in_bytes=self.lmdb_size_in_bytes,
            format_converter=self.lmdb_format_converter,
            total=total
        )


@dataclass
class ImageLMDBCreator(MillionImageFileCreator):
    lmdb_creator: LmdbCreator

    def create(self, paths, dst: str):
        def keys_and_data():
            for p in tqdm(paths, desc=f"storing images to lmdb file:{dst}"):
                with open(p, "rb") as f:
                    data = f.read()
                    yield os.path.basename(p), data

        self.lmdb_creator.create(keys_and_data(), dst)


@dataclass
class LmdbImageStore(ImageStore):
    target_lmdb_path: str
    sort_image_store_keys: bool

    @property
    def txn(self):
        return self.create_txn()

    @contextmanager
    def create_env(self):
        # map_size was 1e11
        # on lustre file system on abci you cannot use file lock. and you have to disable it for lmdb to work.
        with lmdb.open(self.target_lmdb_path, lock=False, readonly=True) as env:
            yield env

    @contextmanager
    def create_txn(self):
        with self.create_env() as env:
            with env.begin(write=False, buffers=True) as txn:
                yield txn

    def num_keys(self):
        with self.create_txn() as txn:
            return txn.stat()['entries']

    def _cached_keys(self) -> List[str]:
        cache_path = os.path.join(self.target_lmdb_path, 'keys.pkl')
        pkl_keys = Pickled(cache_path, self._keys)
        keys = pkl_keys.value
        if len(keys) != self.num_keys():
            pkl_keys.clear()
        return pkl_keys.value

    def keys(self) -> List[str]:
        return self._cached_keys()

    def _keys(self) -> List[str]:
        _keys = list(self.key_iterator())
        if self.sort_image_store_keys:
            logger.info(f"sorting image store keys...")
            _keys = list(sorted(_keys))
        return _keys

    def key_iterator(self):
        with self.create_txn() as txn:
            with txn.cursor() as cur:
                for t in tqdm(cur.iternext(values=False), desc="retrieving keys from lmdb"):
                    yield t.tobytes().decode()

    def __getitem__(self, item):
        with self.create_txn() as txn:
            buf = txn.get(item.encode())
            assert buf is not None  # why the heck this is None?
            return Image.open(io.BytesIO(buf.tobytes()))


@dataclass
class MultiLmdbImageStoreArtifactGetter:
    managed_path_artifact_loader: ManagedPathArtifactLoader
    sort_image_store_keys: bool

    def get(self, idt: ArtifactIdentifier):
        root = self.managed_path_artifact_loader.get_path_identifier(idt)
        lmdb_paths = os.listdir(root)
        lmdbs = [LmdbImageStore(os.path.join(root, p), sort_image_store_keys=self.sort_image_store_keys) for p in
                 tqdm(lmdb_paths, desc='loading multi lmdb artifact')]
        return MergedImageStore(lmdbs, sort_image_store_keys=self.sort_image_store_keys)


@dataclass
class LmdbImageStoreArtifactFactory:
    """use this to get an ImageStore using artifact. so that we can track its usage."""
    json_artifact_loader: "JsonArtifactLoader"
    storage_manager: FileStorageManager
    sort_image_store_keys: bool

    def get_image_store(self, artifact_name):
        key = self.json_artifact_loader.load(artifact_name, "million_image_metadata")
        tgt_path = self.storage_manager.find(**key)
        return LmdbImageStore(tgt_path, sort_image_store_keys=self.sort_image_store_keys)


@dataclass
class ManagedLmdbImageStoreArtifact:
    managed_path_artifact_loader: ManagedPathArtifactLoader
    sort_image_store_keys: bool

    def get_image_store(self, identifier: str) -> ImageStore:
        path = self.managed_path_artifact_loader.get_path(identifier, "lmdb_image_store")
        return LmdbImageStore(path, sort_image_store_keys=self.sort_image_store_keys)


@dataclass
class LmdbImageStoreSplitter:
    multi_lmdb_image_store_artifact_creator: MultiLmdbImageStoreArtifactCreator

    def split_and_log(self, src: ImageStore, dst_metadata: ArtifactMetadata, max_bytes_per_lmdb=2 ** 24):
        return self.multi_lmdb_image_store_artifact_creator.create_from_bytes(
            metadata=dst_metadata,
            key_bytes_pairs=src.mp_bytes_pair_iterator(),
            total=len(src.keys()),
            max_size_per_lmdb=max_bytes_per_lmdb
        )
