
#!/usr/bin/env xonsh


import click
from click import command,option,argument,group
import socket
from tqdm import tqdm
from loguru import logger
@group()
def main():
    """ entry point for everything"""
    pass

@main.command()
def build():
    coconut -k --no-tco --target 35 coconut data_tree/coconut
    coconut -k --no-tco --target 35 coconut_test data_tree/test/coconut


