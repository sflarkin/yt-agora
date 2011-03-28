"""
A read-eval-print-loop that is served up through Bottle and accepts its
commands through HTTP

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: NSF / Columbia
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2011 Matthew Turk.  All Rights Reserved.

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

import json
import os

from .bottle_mods import preroute
from .basic_repl import ProgrammaticREPL

class HTTPREPL(ProgrammaticREPL):

    def __init__(self, locals=None):
        # First we do the standard initialization
        ProgrammaticREPL.__init__(self, locals)
        # Now, since we want to only preroute functions we know about, and
        # since they have different arguments, and most of all because we only
        # want to add them to the routing tables (which are a singleton for the
        # entire interpreter state) we apply all the pre-routing now, rather
        # than through metaclasses or other fancy decorating.
        preroute_table = dict(index = ("/", "GET"),
                              push = ("/push", "POST"),
                              dir = ("/dir", "GET"),
                              doc = ("/doc", "GET"))
        for v, args in preroute_table:
            preroute(args[0], method=args[1])(getattr(self, v))

    def index(self):
        """Return an HTTP-based Read-Eval-Print-Loop terminal."""
        # For now this doesn't work!  We will need to move to a better method
        # for this.
        return open(os.path.join(localDir, "httprepl.html")).read()
        
    def push(self):
        """Push 'line' and return exec results as a bare response."""
        line = request.POST['line']
        result = ProgrammaticREPL.push(self, line)
        new_values = self.locals.pop("new_values", "")
        if result is None:
            # More input lines needed.
            response.status = 204
        return json.dumps( dict(text = result, new_values = new_values ))

    def dir(self):
        """Push 'line' and return result of eval on the final expr."""
        line = request.GET['line']
        result = ProgrammaticREPL.dir(self, line)
        if not result:
            response.status = 204
            return
        return repr(result)

    def doc(self):
        """Push 'line' and return result of getargspec on the final expr."""
        line = request.GET['line']
        result = ProgrammaticREPL.doc(self, line)
        if not result:
            response.status = 204
        return result
