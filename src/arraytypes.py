"""
We want to have flexible arrays, so we do it all in here, and then import from
this module.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
"""

from config import ytcfg
import logging

class ArrayNumTypes:
    def __init__(self):
        pass

# This should only be in the yt_numpy branch!  Got it?

u_numpy = False
u_numarray = False

nT = ArrayNumTypes()

myTypes = [ 'Complex128',  'Bool', 'Int32', 'Complex64', 'UInt16', 'Float32',
            'Int64', 'UInt8',  'Int8', 'Complex32',  'UInt64', 'Float64', 
            'UInt32', 'Float128', 'Int16']

if not ytcfg.has_option("yt","numarray"):
    logging.info("Using NumPy.")
    logging.info("Please report problems to mturk@slac.stanford.edu, including a full traceback.")
    logging.info("(Check for .flat access to arrays -- it should now be .ravel() !)")
    import numpy as na
    import numpy.linalg as la
    import numpy as obj  # Backwards compat
    import numpy.core.records as rec
    import scipy.ndimage as nd # Moved into scipy
    import scipy as sp
    import scipy.weave as weave
    u_numpy = True
    for type in myTypes:
        setattr(nT, type, na.typeNA[type])
else:
    logging.info("Using NumArray.  Many, many things will probably break.  You should use NumPy!")
    import numarray as na
    import numarray.linear_algebra as la
    import numarray.objects as obj
    import numarray.nd_image as nd
    import numarray.records as rec
    u_numarray = True
    for type in myTypes:
        try:
            setattr(nT, type, na.typeDict[type])
        except KeyError:
            pass # We won't log this -- lots of nonsense crap
            #logging.debug("No type %s - this is probably not a problem", type)
