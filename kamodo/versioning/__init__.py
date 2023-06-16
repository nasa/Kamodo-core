# needed for setuptools to find appropriate version
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from ._version import __version__
__version__.package = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def _setuptools_version():  # type: () -> str
    return __version__.public()
