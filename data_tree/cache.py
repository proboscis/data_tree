import os
import shutil
from typing import NamedTuple

import yaml

from data_tree import logger
from data_tree.custom_resource import fopen
from data_tree.util import ensure_path_exists, dict_hash


class CacheHandle(NamedTuple):
    file_dir: str
    conditions: dict
    provider: "ConditionedFilePathProvider"

    def clear(self):
        self.provider.remove(self.conditions)


class ConditionedFilePathProvider:
    """
    root/
      - managed/
        - metadata.yaml => stores all conditions
        - guid of condition
          - single file
          - condition.yaml
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.metadata_file_dir = os.path.join(self.root_dir, "managed", "metadata.yaml")
        ensure_path_exists(self.metadata_file_dir)

        def _create_if_none(fp, default):
            if not os.path.exists(fp):
                ensure_path_exists(fp)
                fopen(fp, "w").use(lambda f: yaml.dump(data=default, stream=f))
            return fopen(fp, "r").use(yaml.load)

        self.db_file_read = lambda: _create_if_none(self.metadata_file_dir, [])
        self.db_file_write = fopen(self.metadata_file_dir, "w")
        self.write_yaml = lambda data: self.db_file_write.use(lambda f: yaml.dump(data=data, stream=f))

    def remove(self, conditions):
        tgt_path = self.get_path(conditions)
        _hash = self.condition_hash(conditions)
        metadata = self.db_file_read()
        new_metadata = [item for item in metadata if item["hash"] != _hash]
        logger.warning(f"removing all cache under {tgt_path} from condition:{conditions}")
        shutil.rmtree(tgt_path)
        self.write_yaml(new_metadata)

    def handles(self):
        data = self.db_file_read()
        result = []
        for item in data:
            result.append(
                CacheHandle(file_dir=self.get_path(item["conditions"]), conditions=item["conditions"], provider=self))
        return result

    def condition_hash(self, conds):
        return dict_hash(conds)

    def get_path(self, conds):
        _hash = self.condition_hash(conds)
        return os.path.join(self.root_dir, "managed", _hash)

    def append_metadata(self, conditions):
        _hash = self.condition_hash(conditions)
        if not os.path.exists(self.metadata_file_dir):
            logger.info(f"created initial metadata at :{self.metadata_file_dir}")
            self.db_file_write.use(lambda f: yaml.dump([], f))
        current_metadata = self.db_file_read()
        hashes = [item["hash"] for item in current_metadata]
        if _hash not in hashes:
            current_metadata.append(dict(hash=_hash, conditions=conditions))
        self.write_yaml(current_metadata)

    def get_managed_file_path(self, conds, filename=None):
        if filename is None:
            filename = "undefined"
        file_dir = self.get_path(conds)
        ensure_path_exists(file_dir)
        self.append_metadata(conds)
        condition_file_path = os.path.join(file_dir, "conditions.yaml")
        logger.info(f"got path for conditions:{conds}")
        ensure_path_exists(condition_file_path)
        with open(condition_file_path, "w") as f:
            yaml.dump(conds, f)
        return os.path.join(file_dir, filename)

    def __call__(self, **conditions):
        return self.get_managed_file_path(conditions)

