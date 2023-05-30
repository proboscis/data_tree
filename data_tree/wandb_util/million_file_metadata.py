from dataclasses import dataclass

from data_tree.util import scantree


@dataclass
class FileStructure:
    pass

def create_file_structure(root:str):
    files = list(scantree(root))

