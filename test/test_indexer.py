import numpy as np

from data_tree.indexer import IdentityIndexer


def test_identity():
    indexer = IdentityIndexer(100)
    assert indexer[:5] == slice(None, 5, None)
    assert indexer[5:] == slice(5, None, None)
    assert indexer[0:5] == slice(0, 5, None)
    assert indexer[0:-1] == slice(0, -1, None)
    i5 = np.arange(5)
    m5 = np.ones(5, dtype=bool)
    assert (indexer[i5] == i5).all()
    assert (indexer[m5] == m5).all()

def test_slice():
    indexer = IdentityIndexer(100).slice(slice(None, None, None))
    assert indexer[:5] == slice(0, 5, None)
    assert indexer[5:] == slice(5, 100, None)
    assert indexer[0:5] == slice(0, 5, None)
    assert indexer[0:-1] == slice(0, 99, None)
    i5 = np.arange(5)
    m5 = np.ones(100, dtype=bool)
    assert (indexer[i5] == i5).all()
    assert (indexer[m5] == np.arange(100)).all()

def test_offset_slice():
    indexer = IdentityIndexer(100).slice(slice(5, None, None))
    assert indexer[:5] == slice(5, 10, None)
    assert indexer[5:] == slice(10, 100, None)
    assert indexer[0:5] == slice(5, 10, None)
    assert indexer[0:-1] == slice(5, 99, None)
    i5 = np.arange(5)
    m5 = np.ones(95, dtype=bool)
    assert (indexer[i5] == i5 + 5).all()
    assert (indexer[m5] == (np.arange(95) + 5)).all()

def test_complex_slice():
    indexer = IdentityIndexer(100).slice(slice(-10, None, None))
    assert indexer[:5] == slice(90, 95, None)
    assert indexer[5:] == slice(95, 100, None)
    assert indexer[0:5] == slice(90, 95, None)
    assert indexer[0:-1] == slice(90, 99, None)
    i5 = np.arange(5)
    m5 = np.ones(len(indexer), dtype=bool)
    assert len(indexer) == 10
    assert (indexer[i5] == (i5 + 90)).all()
    assert (indexer[m5] == (np.arange(10) + 90)).all()

def test_nested_slice():
    indexer = IdentityIndexer(100).slice(slice(-50, None, None)).slice(slice(5,-1))
    assert indexer[:5] == slice(55, 60, None)
    assert indexer[5:] == slice(60, 99, None)
    assert indexer[0:5] == slice(55, 60, None)
    assert indexer[0:-1] == slice(55, 98, None)
    i5 = np.arange(5)
    m5 = np.ones(len(indexer), dtype=bool)
    assert len(indexer) == 44
    assert (indexer[i5] == (i5 + 55)).all()
    assert (indexer[m5] == (np.arange(44) + 55)).all()
