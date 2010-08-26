"""
Logging facility for yt
Will initialize everything, and associate one with each module

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
Homepage: http://yt.enzotools.org/
License:
  Copyright (C) 2007-2009 Matthew Turk.  All Rights Reserved.

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

import logging, os
import logging.handlers as handlers
from yt.config import ytcfg

# This next bit is grabbed from:
# http://stackoverflow.com/questions/384076/how-can-i-make-the-python-logging-output-to-be-colored
def add_coloring_to_emit_ansi(fn):
    # add methods we need to the class
    def new(*args):
        levelno = args[1].levelno
        if(levelno>=50):
            color = '\x1b[31m' # red
        elif(levelno>=40):
            color = '\x1b[31m' # red
        elif(levelno>=30):
            color = '\x1b[33m' # yellow
        elif(levelno>=20):
            color = '\x1b[32m' # green 
        elif(levelno>=10):
            color = '\x1b[35m' # pink
        else:
            color = '\x1b[0m' # normal
        args[1].msg = color + args[1].msg +  '\x1b[0m'  # normal
        #print "after"
        return fn(*args)
    return new

if ytcfg.getboolean("yt","coloredlogs"):
    logging.StreamHandler.emit = add_coloring_to_emit_ansi(logging.StreamHandler.emit)

level = min(max(ytcfg.getint("yt", "loglevel"), 0), 50)
fstring = "%(name)-10s %(levelname)-10s %(asctime)s %(message)s"
logging.basicConfig(
    format=fstring,
    level=level
)

f = logging.Formatter("%(levelname)-10s %(asctime)s %(message)s")

rootLogger = logging.getLogger()

ytLogger = logging.getLogger("yt")
ytLogger.debug("Set log level to %s", level)

fidoLogger = logging.getLogger("yt.fido")
ravenLogger = logging.getLogger("yt.raven")
lagosLogger = logging.getLogger("yt.lagos")
enkiLogger = logging.getLogger("yt.enki")
deliveratorLogger = logging.getLogger("yt.deliverator")
reasonLogger = logging.getLogger("yt.reason")

# Maybe some day we'll make this more configurable...  unfortunately, for now,
# we preserve thread-safety by opening in the current directory.

mb = 10*1024*1024
bc = 10

loggers = []
file_handlers = []

if ytcfg.getboolean("yt","logfile") and os.access(".", os.W_OK):
    log_file_name = ytcfg.get("yt","LogFileName")
    ytFileHandler = handlers.RotatingFileHandler(log_file_name,
                                             maxBytes=mb, backupCount=bc)
    k = logging.Formatter(fstring)
    ytFileHandler.setFormatter(k)
    ytLogger.addHandler(ytFileHandler)
    loggers.append(ytLogger)
    file_handlers.append(ytFileHandler)

def disable_stream_logging():
    # We just remove the root logger's handlers
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            rootLogger.removeHandler(handler)

def disable_file_logging():
    for logger, handler in zip(loggers, file_handlers):
        logger.removeHandler(handler)

if ytcfg.getboolean("yt","suppressStreamLogging"):
    disable_stream_logging()
