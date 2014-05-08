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
import os.path

class YTException(Exception):
    def __init__(self, message = None, pf = None):
        Exception.__init__(self, message)
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
        YTException.__init__(self, pf=pf)
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

class YTFieldNotFound(YTException):
    def __init__(self, fname, pf):
        self.fname = fname
        self.pf = pf

    def __str__(self):
        return "Could not find field '%s' in %s." % (self.fname, self.pf)

class YTCouldNotGenerateField(YTFieldNotFound):
    def __str__(self):
        return "Could field '%s' in %s could not be generated." % (self.fname, self.pf)

class YTFieldTypeNotFound(YTException):
    def __init__(self, fname):
        self.fname = fname

    def __str__(self):
        return "Could not find field '%s'." % (self.fname)

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
    def __init__(self, field_name, unit_name, mathtext_error):
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
        YTException.__init__(self, pf=pf)
        self.parameter = parameter

    def __str__(self):
        return "Parameter file %s is missing %s parameter." % \
            (self.pf, self.parameter)

class NoStoppingCondition(YTException):
    def __init__(self, pf):
        YTException.__init__(self, pf=pf)

    def __str__(self):
        return "Simulation %s has no stopping condition.  StopTime or StopCycle should be set." % \
            self.pf

class YTNotInsideNotebook(YTException):
    def __str__(self):
        return "This function only works from within an IPython Notebook."

class YTGeometryNotSupported(YTException):
    def __init__(self, geom):
        self.geom = geom

    def __str__(self):
        return "We don't currently support %s geometry" % self.geom

class YTCoordinateNotImplemented(YTException):
    def __str__(self):
        return "This coordinate is not implemented for this geometry type."

class YTUnitNotRecognized(YTException):
    def __init__(self, unit):
        self.unit = unit

    def __str__(self):
        return "This parameter file doesn't recognize %s" % self.unit

class YTUnitOperationError(YTException, ValueError):
    def __init__(self, operation, unit1, unit2=None):
        self.operation = operation
        self.unit1 = unit1
        self.unit2 = unit2
        YTException.__init__(self)

    def __str__(self):
        err = "The %s operator for YTArrays with units (%s) " % (self.operation, self.unit1, )
        if self.unit2 is not None:
            err += "and (%s) " % self.unit2
        err += "is not well defined."
        return err

class YTUnitConversionError(YTException):
    def __init__(self, unit1, dimension1, unit2, dimension2):
        self.unit1 = unit1
        self.unit2 = unit2
        self.dimension1 = dimension1
        self.dimension2 = dimension2
        YTException.__init__(self)

    def __str__(self):
        err = "Unit dimensionalities do not match. Tried to convert between " \
          "%s (dim %s) and %s (dim %s)." \
          % (self.unit1, self.dimension1, self.unit2, self.dimension2)
        return err

class YTUfuncUnitError(YTException):
    def __init__(self, ufunc, unit1, unit2):
        self.ufunc = ufunc
        self.unit1 = unit1
        self.unit2 = unit2
        YTException.__init__(self)

    def __str__(self):
        err = "The NumPy %s operation is only allowed on objects with " \
        "identical units. Convert one of the arrays to the other\'s " \
        "units first. Received units (%s) and (%s)." % \
        (self.ufunc, self.unit1, self.unit2)
        return err

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
        YTException.__init__(self, pf=pf)
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
    def __init__(self, width):
        self.error = "width (%s) is invalid" % str(width)

    def __str__(self):
        return str(self.error)

class YTFieldNotParseable(YTException):
    def __init__(self, field):
        self.field = field

    def __str__(self):
        return "Cannot identify field %s" % (self.field,)

class YTDataSelectorNotImplemented(YTException):
    def __init__(self, class_name):
        self.class_name = class_name

    def __str__(self):
        return "Data selector '%s' not implemented." % (self.class_name)

class YTParticleDepositionNotImplemented(YTException):
    def __init__(self, class_name):
        self.class_name = class_name

    def __str__(self):
        return "Particle deposition method '%s' not implemented." % (self.class_name)

class YTDomainOverflow(YTException):
    def __init__(self, mi, ma, dle, dre):
        self.mi = mi
        self.ma = ma
        self.dle = dle
        self.dre = dre

    def __str__(self):
        return "Particle bounds %s and %s exceed domain bounds %s and %s" % (
            self.mi, self.ma, self.dle, self.dre)

class YTIllDefinedFilter(YTException):
    def __init__(self, filter, s1, s2):
        self.filter = filter
        self.s1 = s1
        self.s2 = s2

    def __str__(self):
        return "Filter '%s' ill-defined.  Applied to shape %s but is shape %s." % (
            self.filter, self.s1, self.s2)

class YTIllDefinedBounds(YTException):
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def __str__(self):
        v =  "The bounds %0.3e and %0.3e are ill-defined. " % (self.lb, self.ub)
        v += "Typically this happens when a log binning is specified "
        v += "and zero or negative values are given for the bounds."
        return v

class YTObjectNotImplemented(YTException):
    def __init__(self, pf, obj_name):
        self.pf = pf
        self.obj_name = obj_name

    def __str__(self):
        v  = r"The object type '%s' is not implemented for the parameter file "
        v += r"'%s'."
        return v % (self.obj_name, self.pf)

class YTRockstarMultiMassNotSupported(YTException):
    def __init__(self, mi, ma, ptype):
        self.mi = mi
        self.ma = ma
        self.ptype = ptype

    def __str__(self):
        v = "Particle type '%s' has minimum mass %0.3e and maximum " % (
            self.ptype, self.mi)
        v += "mass %0.3e.  Multi-mass particles are not currently supported." % (
            self.ma)
        return v

class YTEmptyProfileData(Exception):
    pass

class YTTooParallel(YTException):
    def __str__(self):
        return "You've used too many processors for this dataset."

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

class YTInvalidPositionArray(Exception):
    def __init__(self, shape, dimensions):
        self.shape = shape
        self.dimensions = dimensions

    def __str__(self):
        r = """Position arrays must be length and shape (N,3).
               But this one has %s and %s.""" % (self.dimensions, self.shape)
        return r
