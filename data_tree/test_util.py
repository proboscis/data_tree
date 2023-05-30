from tqdm import tqdm

from data_tree.util import scan_dir_mp


def test_scan_dir_mp():
    for item in tqdm(scan_dir_mp(".")):
        pass
        #print(item)
