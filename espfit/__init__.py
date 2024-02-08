"""Infrastruture to train espaloma with experimental observables

Notes
-----
Following INFO message is sometimes raised when espfit.utils.graphs.CustomGraph is imported:

[INFO] 2024-01-09 14:17:16 Note: detected 128 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
[INFO] 2024-01-09 14:17:16 Note: NumExpr detected 128 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 8.
[INFO] 2024-01-09 14:17:16 NumExpr defaulting to 8 threads.

Message muted by setting the logging level to WARNING.
"""

import logging

# Disable asyncio and numexpr logging
# https://stackoverflow.com/questions/60503705/how-can-i-avoid-using-selector-epollselector-log-message-in-django
# https://github.com/deeptools/pyGenomeTracks/issues/98
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('numexpr').setLevel(logging.WARNING)
logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


# Add imports here
#from .espfit import *


#from ._version import __version__
