from typing import NamedTuple


class CliElement:
    pass
class Argument(CliElement,NamedTuple):
    pass
class Option(CliElement,NamedTuple):
    pass

# I can now combine these for my own definitions and the convert it to either click or argparse
# I will just follow cli interface