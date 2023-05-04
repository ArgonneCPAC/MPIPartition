"""Top-level package for MPIPartition."""

__author__ = """Michael Buehlmann"""
__email__ = "buehlmann.michi@gmail.com"
__version__ = "1.0.2"


from .distribute import distribute
from .exchange import exchange
from .overload import overload
from .partition import Partition
