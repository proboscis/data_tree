from setuptools import setup, Extension, Command, find_packages
from Cython.Build import cythonize
# from setuptools import setup, find_packages
import numpy as np

packages = find_packages(include=["data_tree"])
setup(
    zip_safe=False,
    name="data_tree",
    version="0.1",
    description="data conversion tracing library",
    author="proboscis",
    author_email="nameissoap@gmail.com",
    packages=packages,
    install_requires=[
        'lazy',
        'logzero',
        'tqdm',
        'pandas',
        'numpy',
        'matplotlib',
        "h5py",
        "tqdm",
        "frozendict",
        "tblib",
        "bqplot",
        "pyvis",
        "lazy_object_proxy",
        "retry",
        "pprintpp"
    ],
)
