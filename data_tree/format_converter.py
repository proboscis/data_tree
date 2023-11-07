import os
from dataclasses import dataclass
from typing import Callable



from proboscis_image_rules.rulebook import legacy_auto as auto

from data_tree import logger


@dataclass
class FormatConverter:
    format: str
    auto_function:Callable

    def __call__(self, t):
        key, img = t
        bytes = self.auto_function("image,RGB,RGB",img).to(f"{self.format}_bytes")
        return key, bytes


@dataclass
class PidInstance:
    create_instance: Callable

    def __post_init__(self):
        self.pid_to_instance = dict()

    def get(self):
        pid = os.getpid()
        if pid not in self.pid_to_instance:
            logger.warning(f"instantiating for pid:{pid}")
            # each time the process is changed, new instance is created.
            self.pid_to_instance[pid] = self.create_instance()
        return self.pid_to_instance[pid]

def test_add(x):
    return x+1