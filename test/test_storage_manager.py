from data_tree.storage_manager import FileStorageManager
import os
from loguru import logger
def test_storage_manager():
    sm = FileStorageManager("~/.storage.d",["."])
    fn = sm.get_filename("test.hdf5",name="test",kind="something")
    with open(fn,"wb") as f:
        f.write("this is a test file".encode("utf8"))
    logger.info(sm.find(name="test"))