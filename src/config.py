# This module is very simple.  It imports the configuration
# we have written for yt.
# Everything will be returned in a global config dictionary: YTConfig

import ConfigParser, os

ytcfg = ConfigParser.ConfigParser()
ytcfg.read(os.path.join(os.getenv("HOME"),".yt/config"))

