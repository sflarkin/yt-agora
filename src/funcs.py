"""
Useful functions.  If non-original, see function for citation.

@author: U{Matthew Turk<http://www.stanford.edu/~mturk/>}
@organization: U{KIPAC<http://www-group.slac.stanford.edu/KIPAC/>}
@contact: U{mturk@slac.stanford.edu<mailto:mturk@slac.stanford.edu>}
@license:
  Copyright (C) 2007 Matthew Turk.  All Rights Reserved.

  This file is part of yt.

  yt is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import time, types
import progressbar as pb

def iterable(obj):
    """
    Grabbed from Python Cookbook / matploblib.cbook
    """
    try: len(obj)
    except: return False
    return True

def ensure_list(obj):
    if not isinstance(obj, types.ListType):
        return [obj]
    return obj

def time_execution(func):
    """
    Decorator for seeing how long a given function takes, depending on whether
    or not the global 'yt.timefunctions' config parameter is set.
    """
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        mylog.debug('%s took %0.3f s', func.func_name, (t2-t1))
        return res
    from yt import ytcfg
    from yt import lagosLogger as mylog
    if ytcfg.getboolean("yt","timefunctions") == True:
        return wrapper
    else:
        return func

def get_pbar(title, maxval):
    widgets = [ title,
            pb.Percentage(), ' ',
            pb.Bar(marker=pb.RotatingMarker()),
            ' ', pb.ETA(), ' ']
    pbar = pb.ProgressBar(widgets=widgets,
                          maxval=maxval).start()
    return pbar