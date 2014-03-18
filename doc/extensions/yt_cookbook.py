# This extension is quite simple:
#  1. It accepts a script name
#  2. This script is added to the document in a literalinclude
#  3. Any _static images found will be added

from sphinx.util.compat import Directive
from docutils.parsers.rst import directives
import os, glob, shutil

# Some of this magic comes from the matplotlib plot_directive.

def setup(app):
    app.add_directive('yt_cookbook', CookbookScript)
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir

data_patterns = ["*.h5", "*.out", "*.dat"]

class CookbookScript(Directive):
    required_arguments = 1
    optional_arguments = 0

    def run(self):
        rst_file = self.state_machine.document.attributes['source']
        rst_dir = os.path.abspath(os.path.dirname(rst_file))
        script_fn = directives.path(self.arguments[0])
        script_bn = os.path.basename(script_fn)
        script_name = os.path.basename(self.arguments[0]).split(".")[0]

        # This magic is from matplotlib
        dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir,
                                                os.path.dirname(script_fn)))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir) # no problem here for me, but just use built-ins

        rel_dir = os.path.relpath(rst_dir, setup.confdir)
        place = os.path.join(dest_dir, rel_dir)
        if not os.path.isdir(place): os.makedirs(place)
        shutil.copyfile(os.path.join(rst_dir, script_fn),
                        os.path.join(place, script_bn))

        im_path = os.path.join(rst_dir, "_static")
        images = sorted(glob.glob(os.path.join(im_path, "%s__*.png" % script_name)))
        lines = []
        lines.append("(`%s <%s>`__)" % (script_bn, script_fn))
        lines.append("\n")
        lines.append("\n")
        lines.append(".. literalinclude:: %s" % self.arguments[0])
        lines.append("\n")
        lines.append("\n")
        for im in images:
            im_name = os.path.join("_static", os.path.basename(im))
            lines.append(".. image:: %s" % im_name)
            lines.append("   :width: 400")
            lines.append("   :target: ../_images/%s" % os.path.basename(im))
            lines.append("\n")
        lines.append("\n")
        for ext in data_patterns:
            data_files = sorted(glob.glob(os.path.join(
                im_path, "%s__*.%s" % (script_name, ext))))
            for df in data_files:
                df_bn = os.path.basename(df)
                shutil.copyfile(os.path.join(rst_dir, df),
                                os.path.join(dest_dir, rel_dir, df_bn))
                lines.append(" * Data: `%s <%s>`__)" % (df_bn, df))
            lines.append("\n")
        self.state_machine.insert_input(lines, rst_file)
        return []
