import abc
import os
from dataclasses import dataclass
from multiprocessing import Pool, Semaphore
from multiprocessing.shared_memory import SharedMemory
from threading import Thread
from typing import List, Callable, Tuple, Iterable, TypeVar, NamedTuple, Dict

import PIL
import h5py
import numpy as np
from PIL import Image
from cytoolz import merge

from tqdm import tqdm

# from omni_converter.coconut.auto_data import AutoData
from data_tree._series import Series
from data_tree import series, logger
from data_tree.key_value_store import JustKVS
from data_tree.mp_util import GlobalHolder, get_global_holder, SequentialTaskParallel2
from omni_converter.coconut.auto_data import AutoData

T = TypeVar("T")
U = TypeVar("U")

class ImageStorePair(NamedTuple):
    key: str
    image: PIL.Image.Image


def image_to_shared_bytes_name(img: PIL.Image.Image) -> SharedMemory:
    """
    shared memories are not supposed to be created on child process.
    https://bugs.python.org/issue39959
    so to avoid this bug, I need to call unregister
    :param img:
    :return:
    """
    png = auto("image,RGB,RGB", img.convert("RGB")).to('png_bytes')
    return png
    # shm = SharedMemory(create=True,size=len(png))
    # shm.buf[:] = png
    # shm.close()
    # resource_tracker.unregister(shm.name,"shared_memory")
    # you must unregister to avoid shm being freed after this process dies
    # return shm


def _img_png_bytes_loader(t: Tuple[GlobalHolder, GlobalHolder, str]) -> Tuple[str, bytes]:
    try:
        sem_holder, store_holder, key = t
        sem: Semaphore = sem_holder.value
        sem.acquire()
        store = store_holder.value
        shm = image_to_shared_bytes_name(store[key])
        # you cannot pass shared memory instance to another process. pass only a name and reconstruct it on the other process
        return key, shm
    except Exception as e:
        logger.error(e)
        raise e


class ImageStore(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def keys(self) -> List[str]:
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    def __call__(self, item):
        return self.__getitem__(item)

    def subset(self, new_keys: List[str]) -> "ImageStoreSubset":
        return ImageStoreSubset(self, new_keys)

    def map(self, mapper: Callable[[PIL.Image.Image], PIL.Image.Image]) -> "MappedImageStore":
        return MappedImageStore(self, mapper)

    def sorted_image_store_pairs(self) -> Series[ImageStorePair]:
        """
        sorted series of tuple[key,img]
        :return:
        """
        return series(self.keys()).sorted().map(lambda k: ImageStorePair(k, self[k]))

    def sorted_pairs(self) -> Series[Tuple[str, PIL.Image.Image]]:
        return series(self.keys()).sorted().map(lambda k: (k, self[k]))


    def to_kvs(self) -> "KeyValueStore":
        return JustKVS(
            self.keys,
            self.__getitem__
        )

    def map_keys(self, mapper):
        return KeyMappedImageStore(self, mapper)

    def mp_pair_iterator_bak(self) -> Iterable[Tuple[str, AutoData]]:
        holder = get_global_holder(self)  # must do this before pool creation
        sem = Semaphore(value=36)
        # shm = SharedMemory(size=1,create=True)
        seq_task = SequentialTaskParallel2(
            worker_generator=lambda: _img_png_bytes_loader,
            worker_gen_args=(),
            num_worker=36,
            max_pending_result=10000,
        )
        queries = [(holder, k) for k in self.keys()]

        def enqueue():
            for item in queries:
                seq_task.enqueue(item)
            seq_task.enqueue_termination()

        producer = Thread(target=enqueue)
        producer.start()
        with seq_task.managed_start() as res:
            for key, png in tqdm(res, desc='loading with mp seq task paralell'):
                img = auto("png_bytes", png)
                # shm.unlink()
                yield key, img
        # shm.close()
        producer.join()
        # shm.unlink()

    def mp_bytes_pair_iterator(self) -> Iterable[Tuple[str, bytes]]:
        holder = get_global_holder(self)  # must do this before pool creation
        sem = Semaphore(value=os.cpu_count())
        sem_holder = get_global_holder(sem)
        queries = [(sem_holder, holder, k) for k in self.keys()]
        with Pool() as p:
            for key, png in tqdm(p.imap_unordered(_img_png_bytes_loader, queries), desc='loading with mp'):
                yield key, png
                sem.release()

    def split(self, n: int) -> Series["ImageStore"]:
        return series(self.keys()).split_n(n).map(lambda subset: self.subset(subset))

    def take(self, n: int) -> "ImageStore":
        return self.subset(self.keys()[:n])

    def ram(self):
        return RamImageStore({k:self[k] for k in self.keys()})




@dataclass
class RamImageStore(ImageStore):
    images: Dict[str, PIL.Image.Image]

    def __getitem__(self, item):
        return self.images[item]

    def keys(self) -> List[str]:
        return list(self.images.keys())

    def __repr__(self):
        return f"RamImageStore:(size={len(self.images)})"


@dataclass
class LambdaImageStore(ImageStore):
    src_keys: List[str]
    get_item: Callable[[str], PIL.Image.Image]

    def keys(self) -> List[str]:
        return self.src_keys

    def __getitem__(self, item):
        return self.get_item(item)

    def __repr__(self):
        return f"LambdaImageStore(n_src_keys={len(self.src_keys)},get_item={self.get_item})"


class KeyMappedImageStore(ImageStore):
    def keys(self) -> List[str]:
        return list(self.key_to_org_key.keys())

    def __getitem__(self, item):
        return self.src[self.key_to_org_key[item]]

    def __init__(self, src, mapper: Callable[[str], str]):
        self.key_to_org_key = {mapper(k): k for k in tqdm(src.keys(), desc="mapping keys of an imagestore.")}
        self.mapper = mapper
        self.src = src


class SourcedImageStore(ImageStore):
    @abc.abstractmethod
    def src(self) -> ImageStore:
        pass

    def keys(self) -> List[str]:
        return self.src().keys()

    def __getitem__(self, item):
        return self.src()[item]


@dataclass
class ImageStoreSubset(ImageStore):
    src: ImageStore
    new_keys: List[str]

    def keys(self) -> List[str]:
        return self.new_keys

    def __getitem__(self, item):
        return self.src[item]


@dataclass
class MappedImageStore(ImageStore):
    src: ImageStore
    mapper: Callable[[PIL.Image.Image], PIL.Image.Image]

    def keys(self):
        return self.src.keys()

    def __getitem__(self, item):
        return self.mapper(self.src[item])


@dataclass
class FileBasedImageStore(ImageStore):
    "uses file's base name as key."
    image_paths: "ImagePaths"

    def __post_init__(self):
        paths = self.image_paths.paths()
        self.key_to_path = {os.path.basename(p): p for p in tqdm(paths, desc="converting paths to basename")}
        assert len(self.key_to_path) == len(paths), "file image must not contain duplicates"

    def keys(self) -> List[str]:
        return list(self.key_to_path.keys())

    def __getitem__(self, item):
        return auto("image_path", self.key_to_path[item]).to("image,RGB,RGB")


class MillionImageFileCreator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create(self, paths, dst: str):
        pass


class ImageStoreCreator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create(self, key_image_pairs: Iterable[Tuple[str, PIL.Image.Image]], total=None):
        pass


def convert_images_to_bytes_mp(pairs: Iterable[Tuple[str, PIL.Image.Image]], format_converter: Callable):
    import pathos
    with pathos.pools.ProcessPool() as pool:
        # do not ever use multiprocessing again. use pathos!
        # you will always endup with pickling errors!!!!
        # when using ParallelPool, the mapping function must be hashable...
        assert isinstance(format_converter, Callable), f"format_convert must be Callable:{format_converter}"
        for item in pool.uimap(format_converter, pairs):
            # somehow the data is None.
            assert item is not None
            yield item


@dataclass
class MillionImageFlatHdfCreator(MillionImageFileCreator):
    def create(self, paths, dst: str):
        hf = h5py.File(dst, "w")
        for p in tqdm(paths, desc="storing images to hdf5"):
            with open(p, "rb") as f:
                data = np.asarray(f.read())
            hf.create_dataset(os.path.basename(p), data=data)


@dataclass
class MergedImageStore(ImageStore):
    srcs: List[ImageStore]
    sort_image_store_keys: bool

    def __post_init__(self):
        self.key_to_store = merge(*({k: src for k in src.keys()} for src in self.srcs))

    def keys(self) -> List[str]:
        _keys = list(self.key_to_store.keys())
        if self.sort_image_store_keys:
            _keys = list(sorted(_keys))
        return _keys

    def __getitem__(self, item):
        return self.key_to_store[item][item]


class MockImageStore(ImageStore):

    def __init__(self):
        from data_tree.mocks.images import dummy_images
        self.imgs = {str(i): img for i, img in enumerate(dummy_images().to('[image,RGB,RGB]'))}

    def keys(self) -> List[str]:
        return list(self.imgs.keys())

    def __getitem__(self, item):
        return self.imgs[item]
