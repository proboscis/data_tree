import abc
from collections import defaultdict
from hashlib import sha1
import base64
import yaml
import os
import re

from data_tree.util import ensure_path_exists, scantree, Pickled
from loguru import logger
from tqdm.autonotebook import tqdm
import socket

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
        logger.info(f"scan targets:{tgt_dirs}")
        self.tgt_dirs = [os.path.expanduser(p) for p in tgt_dirs]
        logger.debug(f"expanded scan targets:{self.tgt_dirs}")
        logger.info(f"scan targets expanded:{tgt_dirs}")

        self.scan_cache = Pickled(os.path.join(self.db_path, f"scan_cache_{socket.gethostname()}.pkl"), self._scan)
        self.info_cache = Pickled(os.path.join(self.db_path, f"info_cache_{socket.gethostname()}.pkl"), self.gather_info)

    def gather_info(self):
        info = dict()
        for p in scantree(self.db_path):
            if p.name.endswith(".yaml"):
                _hash = os.path.splitext(os.path.basename(p))[0]
                with open(p, "r") as f:
                    info[_hash] = yaml.load(f)
        return info

    def _scan(self):
        prog = re.compile(""".*\.-(\w{6})-\..*""")
        # How can I distinguish ..?
        candidates = defaultdict(list)
        def paths():
            for d in tqdm(self.tgt_dirs, desc="searching directories"):
                try:
                    for p in tqdm(scantree(d,yield_dir=True), desc=f"searching dir:{d}"):
                        yield p
                except FileNotFoundError as fnfe:
                    logger.warning(f"{d} is not found and is ignored.")
                except Exception as e:
                    logger.warning(f"exception in scantree!:{e}")
                    raise e

        for p in paths():
            match = prog.fullmatch(p.name)
            if match is not None:
                if not os.path.basename(p.path).startswith("."):
                    candidates[match[1]].append(os.path.abspath(p.path))
        return candidates

    def find(self, **conditions)->str:
        """
        :param conditions:
        :return: absolute path matching condition
        """
        def find_matching():
            for k, c in self.info_cache.value.items():
                matched = True
                for ck, cv in conditions.items():
                    if ck in c and c[ck] != cv:
                        matched = False
                        break
                if matched and k in self.scan_cache.value:
                    candidates = self.scan_cache.value[k]
                    if len(candidates) >= 2:
                        logger.warning(f"multiple candidates found. using {candidates[0]}.")
                        logger.warning(f"candidates:{candidates}")
                    return candidates[0]

        res = find_matching()
        if res is None or not os.path.exists(res):
            logger.warning(f"no matching file found. rescanning...")
            logger.debug(f"current info:{self.info_cache.value}")
            logger.debug(f"current scan:{self.scan_cache.value}")
            self.info_cache.clear()
            self.scan_cache.clear()
            res = find_matching()
        if res is None:
            candidate = self.get_filename("any_name",**conditions)
            raise RuntimeError(f"no matching path for {conditions}. please make sure a file like {candidate} exists.")
        return res

    def get_filename(self, basename, **conditions):
        hash, filename = self._get_filename(basename, **conditions)
        with open(os.path.join(self.db_path, hash + ".yaml"), "w") as f:
            yaml.dump(conditions, f)
        return filename
