"""
This is a library of yt-defined exceptions



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------


# We don't need to import 'exceptions'
#import exceptions
import os.path

class YTException(Exception):
    def __init__(self, pf = None):
        Exception.__init__(self)
        self.pf = pf

# Data access exceptions:

class YTOutputNotIdentified(YTException):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return "Supplied %s %s, but could not load!" % (
            self.args, self.kwargs)

class YTSphereTooSmall(YTException):
    def __init__(self, pf, radius, smallest_cell):
        YTException.__init__(self, pf)
        self.radius = radius
        self.smallest_cell = smallest_cell

    def __str__(self):
        return "%0.5e < %0.5e" % (self.radius, self.smallest_cell)

class YTAxesNotOrthogonalError(YTException):
    def __init__(self, axes):
        self.axes = axes

    def __str__(self):
        return "The supplied axes are not orthogonal.  %s" % (self.axes)

class YTNoDataInObjectError(YTException):
    def __init__(self, obj):
        self.obj_type = getattr(obj, "_type_name", "")

    def __str__(self):
        s = "The object requested has no data included in it."
        if self.obj_type == "slice":
            s += "  It may lie on a grid face.  Try offsetting slightly."
        return s

class YTSimulationNotIdentified(YTException):
    def __init__(self, sim_type):
        YTException.__init__(self)
        self.sim_type = sim_type

    def __str__(self):
        return "Simulation time-series type %s not defined." % self.sim_type

class YTCannotParseFieldDisplayName(YTException):
    def __init__(self, field_name, display_name, mathtext_error):
        self.field_name = field_name
        self.display_name = display_name
        self.mathtext_error = mathtext_error

    def __str__(self):
        return ("The display name \"%s\" "
                "of the derived field %s " 
                "contains the following LaTeX parser errors:\n" ) \
                % (self.display_name, self.field_name) + self.mathtext_error

class YTCannotParseUnitDisplayName(YTException):
    def __init__(self, field_name, display_unit, mathtext_error):
        self.field_name = field_name
        self.unit_name = unit_name
        self.mathtext_error = mathtext_error

    def __str__(self):
        return ("The unit display name \"%s\" "
                "of the derived field %s " 
                "contains the following LaTeX parser errors:\n" ) \
            % (self.unit_name, self.field_name) + self.mathtext_error

class InvalidSimulationTimeSeries(YTException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
            
class MissingParameter(YTException):
    def __init__(self, pf, parameter):
        YTException.__init__(self, pf)
        self.parameter = parameter

    def __str__(self):
        return "Parameter file %s is missing %s parameter." % \
            (self.pf, self.parameter)

class NoStoppingCondition(YTException):
    def __init__(self, pf):
        YTException.__init__(self, pf)

    def __str__(self):
        return "Simulation %s has no stopping condition.  StopTime or StopCycle should be set." % \
            self.pf

class YTNotInsideNotebook(YTException):
    def __str__(self):
        return "This function only works from within an IPython Notebook."

class YTNotDeclaredInsideNotebook(YTException):
    def __str__(self):
        return "You have not declared yourself to be inside the IPython" + \
               "Notebook.  Do so with this command:\n\n" + \
               "ytcfg['yt','ipython_notebook'] = 'True'"

class YTUnitNotRecognized(YTException):
    def __init__(self, unit):
        self.unit = unit

    def __str__(self):
        return "This parameter file doesn't recognize %s" % self.unit

class YTHubRegisterError(YTException):
    def __str__(self):
        return "You must create an API key before uploading.  See " + \
               "https://data.yt-project.org/getting_started.html"

class YTNoFilenamesMatchPattern(YTException):
    def __init__(self, pattern):
        self.pattern = pattern

    def __str__(self):
        return "No filenames were found to match the pattern: " + \
               "'%s'" % (self.pattern)

class YTNoOldAnswer(YTException):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return "There is no old answer available.\n" + \
               str(self.path)

class YTCloudError(YTException):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return "Failed to retrieve cloud data. Connection may be broken.\n" + \
               str(self.path)

class YTEllipsoidOrdering(YTException):
    def __init__(self, pf, A, B, C):
        YTException.__init__(self, pf)
        self._A = A
        self._B = B
        self._C = C

    def __str__(self):
        return "Must have A>=B>=C"

class EnzoTestOutputFileNonExistent(YTException):
    def __init__(self, filename):
        self.filename = filename
        self.testname = os.path.basename(os.path.dirname(filename))

    def __str__(self):
        return "Enzo test output file (OutputLog) not generated for: " + \
            "'%s'" % (self.testname) + ".\nTest did not complete."

class YTNoAPIKey(YTException):
    def __init__(self, service, config_name):
        self.service = service
        self.config_name = config_name

    def __str__(self):
        return "You need to set an API key for %s in ~/.yt/config as %s" % (
            self.service, self.config_name)

class YTTooManyVertices(YTException):
    def __init__(self, nv, fn):
        self.nv = nv
        self.fn = fn

    def __str__(self):
        s = "There are too many vertices (%s) to upload to Sketchfab. " % (self.nv)
        s += "Your model has been saved as %s .  You should upload manually." % (self.fn)
        return s

class YTInvalidWidthError(YTException):
    def __init__(self, error):
        self.error = error

    def __str__(self):
        return str(self.error)

class YTEmptyProfileData(Exception):
    pass

class YTDuplicateFieldInProfile(Exception):
    def __init__(self, field, new_spec, old_spec):
        self.field = field
        self.new_spec = new_spec
        self.old_spec = old_spec

    def __str__(self):
        r = """Field %s already exists with field spec:
               %s
               But being asked to add it with:
               %s""" % (self.field, self.old_spec, self.new_spec)
        return r
