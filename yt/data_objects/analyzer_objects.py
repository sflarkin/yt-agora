"""
Analyzer objects for time series datasets



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import inspect

from yt.funcs import *
from yt.extern.six import add_metaclass

analysis_task_registry = {}

class RegisteredTask(type):
    def __init__(cls, name, b, d):
        type.__init__(cls, name, b, d)
        if hasattr(cls, "skip") and cls.skip == False:
            return
        analysis_task_registry[cls.__name__] = cls

@add_metaclass(RegisteredTask)
class AnalysisTask(object):

    def __init__(self, *args, **kwargs):
        # This should only get called if the subclassed object
        # does not override
        if len(args) + len(kwargs) != len(self._params):
            raise RuntimeError
        self.__dict__.update(zip(self._params, args))
        self.__dict__.update(kwargs)

    def __repr__(self):
        # Stolen from YTDataContainer.__repr__
        s = "%s: " % (self.__class__.__name__)
        s += ", ".join(["%s=%s" % (i, getattr(self,i))
                       for i in self._params])
        return s

def analysis_task(params = None):
    if params is None: params = tuple()
    def create_new_class(func):
        cls = type(func.__name__, (AnalysisTask,),
                   dict(eval = func, _params = params))
        return cls
    return create_new_class

@analysis_task(('field',))
def MaximumValue(params, data_object):
    v = data_object.quantities["MaxLocation"](
            params.field)[0]
    return v

@analysis_task()
def CurrentTimeYears(params, ds):
    return ds.current_time * ds["years"]

class SlicePlotDataset(AnalysisTask):
    _params = ['field', 'axis', 'center']

    def __init__(self, *args, **kwargs):
        from yt.visualization.api import SlicePlot
        self.SlicePlot = SlicePlot
        AnalysisTask.__init__(self, *args, **kwargs)

    def eval(self, ds):
        slc = self.SlicePlot(ds, self.axis, self.field, center = self.center)
        return slc.save()

class QuantityProxy(AnalysisTask):
    _params = None
    quantity_name = None

    def __repr__(self):
        # Stolen from YTDataContainer.__repr__
        s = "%s: " % (self.__class__.__name__)
        s += ", ".join(["%s" % [arg for arg in self.args]])
        s += ", ".join(["%s=%s" % (k,v) for k, v in self.kwargs.items()])
        return s

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def eval(self, data_object):
        rv = data_object.quantities[self.quantity_name](
            *self.args, **self.kwargs)
        return rv

class ParameterValue(AnalysisTask):
    _params = ['parameter']

    def __init__(self, parameter, cast=None):
        self.parameter = parameter
        if cast is None:
            cast = lambda a: a
        self.cast = cast

    def eval(self, ds):
        return self.cast(ds.get_parameter(self.parameter))

def create_quantity_proxy(quantity_object):
    args, varargs, kwargs, defaults = inspect.getargspec(quantity_object[1])
    # Strip off 'data' which is on every quantity function
    params = args[1:] 
    if kwargs is not None: params += kwargs
    dd = dict(_params = params, quantity_name = quantity_object[0])
    cls = type(quantity_object[0], (QuantityProxy,), dd)
    return cls
