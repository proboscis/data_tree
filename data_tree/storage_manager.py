import abc
import os
import re
import socket
from collections import defaultdict
from hashlib import sha1
from typing import List

import cloudpickle
import yaml


from tabulate import tabulate
from tqdm import tqdm

from data_tree import logger
from data_tree.picklable_file_lock import PicklableFileLock
from data_tree.util import scantree, Pickled, scantree_filtered


class DBProvider:

    @abc.abstractmethod
    def find_path(self, conditions):
        pass

    @abc.abstractmethod
    def get_filename(self, basename, **conditions):
        pass

    def _get_filename(self, basename, **conditions):
        name, ext = os.path.splitext(basename)
        if ext == "":
            ext = "."
        s = sha1(str(sorted(list(conditions.items()))).encode("utf-8"))
        tgt_bytes = s.hexdigest()
        hash_str = tgt_bytes[:6]
        return hash_str, f"{name}.-{hash_str}-{ext}"


class FileStorageManager(DBProvider):
    def __init__(self, db_path, tgt_dirs):
        """
        :param db_path: something like ~/.storage.d
        :param tgt_dirs: list of dirs. firs dirs will be prioritized
        """
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(self.db_path, exist_ok=True)
        from loguru import logger
        logger.info(f"scan targets:{tgt_dirs}")
        self.tgt_dirs = [os.path.expanduser(p) for p in tgt_dirs]
        logger.debug(f"expanded scan targets:{self.tgt_dirs}")
        logger.info(f"scan targets expanded:{tgt_dirs}")
        self.scan_lock_path = os.path.join(self.db_path, f"scan_lock.lock")
        self.scan_lock = PicklableFileLock(self.scan_lock_path)

        self.scan_cache = Pickled(os.path.join(self.db_path, f"scan_cache_{socket.gethostname()}.pkl"), self._scan)
        self.info_cache = Pickled(os.path.join(self.db_path, f"info_cache_{socket.gethostname()}.pkl"),
                                  self.gather_info)

    def __getstate__(self):
        res = dict(
            db_path=self.db_path,
            tgt_dirs=self.tgt_dirs,
            scan_cache=self.scan_cache,
            info_cache=self.info_cache,
            scan_lock_path=self.scan_lock_path
        )
        # check_picklable(res)
        return res

    def __setstate__(self, state):
        state["scan_lock"] = PicklableFileLock(state["scan_lock_path"])
        for k, v in state.items():
            setattr(self, k, v)

    def gather_info(self):
        logger.warning(f"waiting for scan_lock")
        with self.scan_lock:
            info = dict()
            for p in scantree(self.db_path):
                if p.name.endswith(".yaml"):
                    _hash = os.path.splitext(os.path.basename(p))[0]
                    with open(p, "r") as f:
                        info[_hash] = yaml.load(f,yaml.SafeLoader)
                    logger.debug(f"loaded info yaml {_hash}:{info[_hash]}")

            # additionaly look for a scanned candidates for __info__.yaml
            for _hash, paths in self.scan_cache.value.items():
                for p in paths:
                    info_file = os.path.join(p, "__info__.yaml")
                    if os.path.exists(info_file):
                        with open(info_file, "r") as f:
                            info[_hash] = yaml.load(f,yaml.SafeLoader)
            # now any directory with .-as132-. can be searched with conditions
            return info

    def _scan(self):
        logger.warning(f"waiting for scan_lock")
        with self.scan_lock:
            prog = re.compile("""^.*\.-(\w{6})-\.$""")
            # How can I distinguish ..?
            candidates = defaultdict(list)

            def ignore_matched_dir(p):
                match = prog.fullmatch(p)
                if match is not None:
                    logger.debug(f"ignoring files inside dir:{p}")
                    return True  # ignore this directory's content
                return False

            def paths():
                for d in tqdm(self.tgt_dirs, desc="searching directories"):
                    try:
                        for p in tqdm(scantree_filtered(d, ignore_matched_dir, yield_dir=True),
                                      desc=f"searching dir:{d}"):
                            # logger.info(f'checking path:{p}')
                            yield p
                    except FileNotFoundError as fnfe:
                        logger.warning(f"{d} is not found and is ignored.")
                    except Exception as e:
                        logger.warning(f"exception in scantree!:{e}")
                        raise e

            for p in paths():
                # logger.debug(p.path)
                match = prog.fullmatch(p.name)
                if match is not None:
                    if not os.path.basename(p.path).startswith("."):
                        if not p.path.endswith(".lock"):
                            candidates[match[1]].append(os.path.abspath(p.path))

            logger.info(tabulate(candidates.items()))
            return candidates

    def find_matching(self, conditions):
        from frozendict import frozendict
        import os
        logger.info(f"calling customized find_matching")
        logger.info(f"self:{self}")
        logger.info(f"{type(self.scan_lock)}")
        conditions =frozendict(conditions)
        with self.scan_lock:
            # for each key and value in caches
            for k, c in self.info_cache.value.items():
                c = frozendict(c)
                matched = c==conditions
                print(f"matched={matched}!")
                if matched and k in self.scan_cache.value:
                    candidates = self.scan_cache.value[k]
                    candidates = [c for c in candidates if os.path.exists(c)]
                    if len(candidates) >= 2:
                        logger.warning(f"multiple candidates found. using {candidates[0]}.")
                        logger.warning(f"candidates:{candidates}")
                    if candidates:
                        return candidates[0]

    # @lru_cache(maxsize=None)
    def find(self, **conditions) -> str:
        """
        :param conditions:
        :return: absolute path matching condition
        """
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe, 2)

        # frame = calframe[1]
        # logger.debug(f'caller name: {frame.filename}, {frame.index}')
        logger.info(f"looking for:{conditions}")
        with self.scan_lock:
            res = self.find_matching(conditions)
            if res is None or not os.path.exists(res):
                logger.warning(f"no matching file found for {conditions}. rescanning...")
                logger.debug(f"current info\n:{tabulate(sorted(self.info_cache.value.items()))}")
                logger.debug(f"current scan\n:{tabulate(sorted(self.scan_cache.value.items()))}")
                self.info_cache.clear()
                self.scan_cache.clear()
                res = self.find_matching(conditions)
            if res is None:
                candidate = self.get_filename("any_name", **conditions)
                raise RuntimeError(
                    f"no matching path for {conditions}. please make sure a file like {candidate} exists.")
            logger.debug(f"found path for {conditions} : {res}")
            return res

    def clear(self):
        logger.warning(f"clearing all cache of storage manager")
        with self.scan_lock:
            self.scan_cache.clear()
            self.info_cache.clear()

    def get_filename(self, basename, **conditions):
        with self.scan_lock:
            hash, filename = self._get_filename(basename, **conditions)
            with open(os.path.join(self.db_path, hash + ".yaml"), "w") as f:
                yaml.dump(conditions, f)
            return filename

    def get_store_path_candidates(self, basename, **conditions) -> List[str]:
        name = self.get_filename(basename, **conditions)
        return [os.path.join(t, name) for t in self.tgt_dirs]


class MockStorageManager():
    pass
