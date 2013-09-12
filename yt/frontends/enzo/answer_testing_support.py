"""
Answer Testing support for Enzo.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.testing import *
from yt.config import ytcfg
from yt.mods import *

from yt.utilities.answer_testing.framework import \
     AnswerTestingTest, \
     can_run_pf, \
     FieldValuesTest, \
     GridHierarchyTest, \
     GridValuesTest, \
     ProjectionValuesTest, \
     ParentageRelationshipsTest, \
     temp_cwd, \
     AssertWrapper

def requires_outputlog(path = ".", prefix = ""):
    def ffalse(func):
        return lambda: None
    def ftrue(func):
        @wraps(func)
        def fyielder(*args, **kwargs):
            with temp_cwd(path):
                for t in func(*args, **kwargs):
                    if isinstance(t, AnswerTestingTest):
                        t.prefix = prefix
                    yield t
        return fyielder
    if os.path.exists("OutputLog"):
        return ftrue
    with temp_cwd(path):
        if os.path.exists("OutputLog"):
            return ftrue
    return ffalse
     
def standard_small_simulation(pf_fn, fields):
    if not can_run_pf(pf_fn): return
    dso = [None]
    tolerance = ytcfg.getint("yt", "answer_testing_tolerance")
    bitwise = ytcfg.getboolean("yt", "answer_testing_bitwise")
    for field in fields:
        if bitwise:
            yield GridValuesTest(pf_fn, field)
        if 'particle' in field: continue
        for ds in dso:
            for axis in [0, 1, 2]:
                for weight_field in [None, "Density"]:
                    yield ProjectionValuesTest(
                        pf_fn, axis, field, weight_field,
                        ds, decimals=tolerance)
            yield FieldValuesTest(
                    pf_fn, field, ds, decimals=tolerance)
                    
class ShockTubeTest(object):
    def __init__(self, data_file, solution_file, fields, 
                 left_edges, right_edges, rtol, atol):
        self.solution_file = solution_file
        self.data_file = data_file
        self.fields = fields
        self.left_edges = left_edges
        self.right_edges = right_edges
        self.rtol = rtol
        self.atol = atol

    def __call__(self):
        # Read in the pf
        pf = load(self.data_file)  
        exact = self.get_analytical_solution() 

        ad = pf.h.all_data()
        position = ad['x']
        for k in self.fields:
            field = ad[k]
            for xmin, xmax in zip(self.left_edges, self.right_edges):
                mask = (position >= xmin)*(position <= xmax)
                exact_field = np.interp(position[mask], exact['pos'], exact[k]) 
                myname = "ShockTubeTest_%s" % k
                # yield test vs analytical solution 
                yield AssertWrapper(myname, assert_allclose, field[mask], 
                                    exact_field, self.rtol, self.atol)

    def get_analytical_solution(self):
        # Reads in from file 
        pos, dens, vel, pres, inte = \
                np.loadtxt(self.solution_file, unpack=True)
        exact = {}
        exact['pos'] = pos
        exact['Density'] = dens
        exact['x-velocity'] = vel
        exact['Pressure'] = pres
        exact['ThermalEnergy'] = inte
        return exact
