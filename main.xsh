#!/Users/Masui/.pyenv/versions/anaconda3-5.3.1/bin/xonsh


import click
from click import command,option,argument,group
import socket
from tqdm import tqdm
from loguru import logger

from data_tree import auto
from loguru import logger

@group()
def main():
    """ entry point for everything"""
    pass

@main.command()
def build():
    coconut -k --no-tco --target 35 coconut data_tree/coconut
    coconut -k --no-tco --target 35 coconut_test data_tree/test/coconut


@main.command()
@argument("src")
@argument("dst")
def test_convert(src,dst):
    coconut -k --no-tco --target 35 coconut data_tree/coconut --force
    coconut -k --no-tco --target 35 coconut_test data_tree/test/coconut
    logger.info(auto(eval(src))(None).converter(eval(dst)))

if __name__ == '__main__':
    main()