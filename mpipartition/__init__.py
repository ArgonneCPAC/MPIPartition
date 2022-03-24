"""Top-level package for MPIPartition."""

__author__ = """Michael Buehlmann"""
__email__ = "buehlmann.michi@gmail.com"
__version__ = "0.2.2"


from .partition import Partition
from .distribute import distribute
from .overload import overload
from .exchange import exchange
