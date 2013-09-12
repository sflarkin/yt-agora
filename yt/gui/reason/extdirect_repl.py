"""
A read-eval-print-loop that is served up through Bottle and accepts its
commands through ExtDirect calls



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import json
import os
import stat
import cStringIO
import logging
import uuid
import numpy as np
import time
import urllib
import urllib2
import pprint
import traceback
import tempfile
import base64
import imp
import threading
import Queue
import zipfile
try:
    import Pyro4
except ImportError:
    pass

from yt.funcs import *
from yt.utilities.logger import ytLogger, ufstring
from yt.utilities.definitions import inv_axis_names
from yt.visualization.image_writer import apply_colormap
from yt.visualization.api import Streamlines
from .widget_store import WidgetStore

from .bottle_mods import preroute, BottleDirectRouter, notify_route, \
                         PayloadHandler, lockit
from yt.extern.bottle import response, request, route, static_file
from .utils import get_list_of_datasets
from .basic_repl import ProgrammaticREPL

try:
    import pygments
    import pygments.lexers
    import pygments.formatters
    def _highlighter():
        pl = pygments.lexers.PythonLexer()
        hf = pygments.formatters.HtmlFormatter(
                linenos='table', linenospecial=2)
        def __highlighter(a):
            return pygments.highlight(a, pl, hf)
        return __highlighter, hf.get_style_defs()
    # We could add an additional '.highlight_pyg' in the call
    highlighter, highlighter_css = _highlighter()
except ImportError:
    highlighter = lambda a: a
    highlight_css = ''

local_dir = os.path.dirname(__file__)

class ExecutionThread(threading.Thread):
    def __init__(self, repl):
        self.repl = repl
        self.payload_handler = PayloadHandler()
        self.queue = Queue.Queue()
        threading.Thread.__init__(self)
        self.daemon = True

    def heartbeat(self):
        return

    def run(self):
        while 1:
            #print "Checking for a queue ..."
            try:
                task = self.queue.get(True, 1)
            except Queue.Empty:
                if self.repl.stopped: return
                continue
            print "Received the task", task
            if task['type'] != 'code':
                raise NotImplementedError
            print task
            self.execute_one(task['code'], task['hide'], task['result_id'])
            self.queue.task_done()

    def wait(self):
        self.queue.join()

    def execute_one(self, code, hide, result_id):
        self.repl.executed_cell_texts.append(code)
        result = ProgrammaticREPL.execute(self.repl, code)
        if self.repl.debug:
            print "==================== Cell Execution ===================="
            print code
            print "====================                ===================="
            print result
            print "========================================================"
        self.payload_handler.add_payload(
            {'type': 'cell',
             'output': result,
             'input': highlighter(code),
             'image_data': '',
             'result_id': result_id,
             'hide': hide,
             'raw_input': code},
            )
        objs = get_list_of_datasets()
        self.payload_handler.add_payload(
            {'type': 'dataobjects',
             'objs': objs})

class PyroExecutionThread(ExecutionThread):
    def __init__(self, repl):
        ExecutionThread.__init__(self, repl)
        hmac_key = raw_input("HMAC_KEY? ").strip()
        uri = raw_input("URI? ").strip()
        Pyro4.config.HMAC_KEY = hmac_key
        self.executor = Pyro4.Proxy(uri)

    def execute_one(self, code, hide, result_id):
        self.repl.executed_cell_texts.append(code)
        print code
        result = self.executor.execute(code)
        if not hide:
            self.repl.payload_handler.add_payload(
                {'type': 'cell',
                 'output': result,
                 'input': highlighter(code),
                 'result_id': result_id,
                 'raw_input': code},
                )
        ph = self.executor.deliver()
        for p in ph:
            self.repl.payload_handler.add_payload(p)

    def heartbeat(self):
        ph = self.executor.deliver()
        for p in ph:
            self.repl.payload_handler.add_payload(p)

def reason_pylab():
    from .utils import deliver_image
    def _canvas_deliver(canvas):
        tf = tempfile.TemporaryFile()
        canvas.print_png(tf)
        tf.seek(0)
        img_data = base64.b64encode(tf.read())
        tf.close()
        deliver_image(img_data)
    def reason_draw_if_interactive():
        if matplotlib.is_interactive():
            figManager =  Gcf.get_active()
            if figManager is not None:
                _canvas_deliver(figManager.canvas)
    def reason_show(mainloop = True):
        # We ignore mainloop here
        for manager in Gcf.get_all_fig_managers():
            _canvas_deliver(manager.canvas)
    # Matplotlib has very nice backend overriding.
    # We should really use that.  This is just a hack.
    import matplotlib
    matplotlib.use("agg") # Hotfix for when we import pylab below
    new_agg = imp.new_module("reason_agg")
    import matplotlib.backends.backend_agg as bagg
    new_agg.__dict__.update(bagg.__dict__)
    new_agg.__dict__.update(
        {'show': reason_show,
         'draw_if_interactive': reason_draw_if_interactive})
    sys.modules["reason_agg"] = new_agg
    bagg.draw_if_interactive = reason_draw_if_interactive
    from matplotlib._pylab_helpers import Gcf
    matplotlib.rcParams["backend"] = "module://reason_agg"
    import pylab
    pylab.switch_backend("module://reason_agg")

_startup_template = r"""\
import pylab
from yt.mods import *
from yt.gui.reason.utils import load_script, deliver_image
from yt.gui.reason.widget_store import WidgetStore
from yt.data_objects.static_output import _cached_pfs

pylab.ion()
data_objects = []
widget_store = WidgetStore()
"""

class ExtDirectREPL(ProgrammaticREPL, BottleDirectRouter):
    _skip_expose = ('index')
    my_name = "ExtDirectREPL"
    timeout = 660 # a minute longer than the rocket server timeout
    server = None
    _heartbeat_timer = None

    def __init__(self, reasonjs_path, locals=None,
                 use_pyro=False):
        # First we do the standard initialization
        self.reasonjs_file = zipfile.ZipFile(reasonjs_path, 'r')
        ProgrammaticREPL.__init__(self, locals)
        # Now, since we want to only preroute functions we know about, and
        # since they have different arguments, and most of all because we only
        # want to add them to the routing tables (which are a singleton for the
        # entire interpreter state) we apply all the pre-routing now, rather
        # than through metaclasses or other fancy decorating.
        preroute_table = dict(index = ("/", "GET"),
                              _help_html = ("/help.html", "GET"),
                              _myapi = ("/ext-repl-api.js", "GET"),
                              _session_py = ("/session.py", "GET"),
                              _highlighter_css = ("/highlighter.css", "GET"),
                              _reasonjs = ("/reason-js/:path#.+#", "GET"),
                              _app = ("/reason/:path#.+#", "GET"),
                              )
        for v, args in preroute_table.items():
            preroute(args[0], method=args[1])(getattr(self, v))
        # This has to be routed to the root directory
        self.api_url = "repl"
        BottleDirectRouter.__init__(self)
        self.payload_handler = PayloadHandler()
        if use_pyro:
            self.execution_thread = PyroExecutionThread(self)
        else:
            self.execution_thread = ExecutionThread(self)
        # We pass in a reference to ourself
        self.execute(_startup_template)
        self.widget_store = WidgetStore(self)
        # Now we load up all the yt.mods stuff, but only after we've finished
        # setting up.
        reason_pylab()

    def activate(self):
        self.payload_handler._prefix = self._global_token
        self._setup_logging_handlers()
        # Setup our heartbeat
        self.last_heartbeat = time.time()
        self._check_heartbeat()
        self.execute("widget_store._global_token = '%s'" % self._global_token)
        self.execution_thread.start()

    def exception_handler(self, exc):
        result = {'type': 'cell',
                  'input': 'ERROR HANDLING IN REASON',
                  'result_id': None,
                  'output': traceback.format_exc()}
        return result

    def _setup_logging_handlers(self):
        handler = PayloadLoggingHandler()
        formatter = logging.Formatter(ufstring)
        handler.setFormatter(formatter)
        ytLogger.addHandler(handler)

    def index(self):
        root = os.path.join(local_dir, "html")
        return static_file("index.html", root)

    def heartbeat(self):
        self.last_heartbeat = time.time()
        if self.debug: print "### Heartbeat ... started: %s" % (time.ctime())
        for i in range(30):
            # Check for stop
            if self.debug: print "    ###"
            if self.stopped: return {'type':'shutdown'} # No race condition
            if self.payload_handler.event.wait(1): # One second timeout
                if self.debug: print "    ### Delivering payloads"
                rv = self.payload_handler.deliver_payloads()
                if self.debug: print "    ### Got back, returning"
                return rv
            self.execution_thread.heartbeat()
        if self.debug: print "### Heartbeat ... finished: %s" % (time.ctime())
        return []

    def _check_heartbeat(self):
        if self.server is not None:
            if not all((s._monitor.is_alive() for s in self.server.values())):
                self.shutdown()
                return
        if time.time() - self.last_heartbeat > self.timeout:
            print "Shutting down after a timeout of %s" % (self.timeout)
            #sys.exit(0)
            # Still can't shut down yet, because bottle doesn't return the
            # server instance by default.
            self.shutdown()
            return
        if self._heartbeat_timer is not None: return
        self._heartbeat_timer = threading.Timer(10, self._check_heartbeat)
        self._heartbeat_timer.start()

    def shutdown(self):
        if self.server is None:
            return
        self._heartbeat_timer.cancel()
        self.stopped = True
        self.payload_handler.event.set()
        for v in self.server.values():
            v.stop()
        for t in threading.enumerate():
            print "Found a living thread:", t

    def _help_html(self):
        root = os.path.join(local_dir, "html")
        return static_file("help.html", root)

    def _reasonjs(self, path):
        pp = os.path.join("reason-js", path)
        try:
            f = self.reasonjs_file.open(pp)
        except KeyError:
            response.status = 404
            return
        if path[-4:].lower() in (".png", ".gif", ".jpg"):
            response.headers['Content-Type'] = "image/%s" % (path[-3:].lower())
        elif path[-4:].lower() == ".css":
            response.headers['Content-Type'] = "text/css"
        elif path[-3:].lower() == ".js":
            response.headers['Content-Type'] = "text/javascript"
        return f.read()

    def _app(self, path):
        root = os.path.join(local_dir, "html")
        return static_file(path, root)

    def _highlighter_css(self):
        response.headers['Content-Type'] = "text/css"
        return highlighter_css

    def execute(self, code, hide = False, result_id = None):
        task = {'type': 'code',
                'code': code,
                'hide': hide,
                'result_id': result_id,
                }
        self.execution_thread.queue.put(task)
        return dict(status = True)

    def get_history(self):
        return self.executed_cell_texts[:]

    @lockit
    def save_session(self, filename):
        if filename.startswith('~'):
            filename = os.path.expanduser(filename)
        elif not filename.startswith('/'):
            filename = os.path.join(os.getcwd(), filename)
        if os.path.exists(filename):
            return {'status': 'FAIL', 'filename': filename,
                    'error': 'File exists!'}
        try:
            f = open(filename, 'w')
            f.write("\n######\n".join(self.executed_cell_texts))
            f.close()
        except IOError as (errno, strerror):
            return {'status': 'FAIL', 'filename': filename,
                    'error': strerror}
        except:
            return {'status': 'FAIL', 'filename': filename,
                    'error': 'Unexpected error.'}
        return {'status': 'SUCCESS', 'filename': filename}

    @lockit
    def paste_session(self):
        import xmlrpclib, cStringIO
        p = xmlrpclib.ServerProxy(
            "http://paste.yt-project.org/xmlrpc/",
            allow_none=True)
        cs = cStringIO.StringIO()
        cs.write("\n######\n".join(self.executed_cell_texts))
        cs = cs.getvalue()
        ret = p.pastes.newPaste('python', cs, None, '', '', True)
        site = "http://paste.yt-project.org/show/%s" % ret
        return {'status': 'SUCCESS', 'site': site}

    @lockit
    def paste_text(self, to_paste):
        import xmlrpclib, cStringIO
        p = xmlrpclib.ServerProxy(
            "http://paste.yt-project.org/xmlrpc/",
            allow_none=True)
        ret = p.pastes.newPaste('python', to_paste, None, '', '', True)
        site = "http://paste.yt-project.org/show/%s" % ret
        return {'status': 'SUCCESS', 'site': site}

    _api_key = 'f62d550859558f28c4c214136bc797c7'
    def upload_image(self, image_data, caption):
        if not image_data.startswith("data:"): return {'uploaded':False}
        prefix = "data:image/png;base64,"
        image_data = image_data[len(prefix):]
        parameters = {'key':self._api_key, 'image':image_data, type:'base64',
                      'caption': caption, 'title': "Uploaded Image from reason"}
        data = urllib.urlencode(parameters)
        req = urllib2.Request('http://api.imgur.com/2/upload.json', data)
        try:
            response = urllib2.urlopen(req).read()
        except urllib2.HTTPError as e:
            print "ERROR", e
            return {'uploaded':False}
        rv = json.loads(response)
        rv['uploaded'] = True
        pprint.pprint(rv)
        return rv

    @lockit
    def _session_py(self):
        cs = cStringIO.StringIO()
        cs.write("\n######\n".join(self.executed_cell_texts))
        cs.seek(0)
        response.headers["content-disposition"] = "attachment;"
        return cs

    @lockit
    def load(self, base_dir, filename):
        pp = os.path.join(base_dir, filename)
        funccall = "pfs.append(load('%s'))" % pp
        self.execute(funccall)
        return []

    def file_listing(self, base_dir, sub_dir):
        if base_dir == "":
            cur_dir = os.getcwd()
        elif sub_dir == "":
            cur_dir = base_dir
        else:
            cur_dir = os.path.join(base_dir, sub_dir)
            cur_dir = os.path.abspath(cur_dir)
        if not os.path.isdir(cur_dir):
            return {'change':False}
        fns = os.listdir(cur_dir)
        results = [("..", 0, "directory")]
        for fn in sorted((os.path.join(cur_dir, f) for f in fns)):
            if not os.access(fn, os.R_OK): continue
            if os.path.isfile(fn):
                size = os.path.getsize(fn)
                t = "file"
            else:
                size = 0
                t = "directory"
            results.append((os.path.basename(fn), size, t))
        return dict(objs = results, cur_dir=cur_dir)

class PayloadLoggingHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        logging.StreamHandler.__init__(self, *args, **kwargs)
        self.payload_handler = PayloadHandler()

    def emit(self, record):
        msg = self.format(record)
        self.payload_handler.add_payload(
            {'type':'logentry',
             'log_entry':msg})

if os.path.exists(os.path.expanduser("~/.yt/favicon.ico")):
    ico = os.path.expanduser("~/.yt/")
else:
    ico = os.path.join(local_dir, "html", "resources", "images")
@route("/favicon.ico", method="GET")
def _favicon_ico():
    print ico
    return static_file("favicon.ico", ico)

class ExtProgressBar(object):
    def __init__(self, title, maxval):
        self.title = title
        self.maxval = maxval
        self.last = 0
        # Now we add a payload for the progress bar
        self.payload_handler = PayloadHandler()
        self.payload_handler.add_payload(
            {'type': 'widget',
             'widget_type': 'progressbar',
             'varname': 'pbar_top',
             'data': {'title':title}
            })

    def update(self, val):
        # An update is only meaningful if it's on the order of 1/100 or greater

        if (val - self.last) > (self.maxval / 100.0):
            self.last = val
            self.payload_handler.add_payload(
                {'type': 'widget_payload',
                 'widget_id': 'pbar_top',
                 'value': float(val) / self.maxval})

    def finish(self):
        self.payload_handler.add_payload(
            {'type': 'widget_payload',
             'widget_id': 'pbar_top',
             'value': -1})
