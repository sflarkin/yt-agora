"""
Answer Testing using Nose as a starting point

Author: Matthew Turk <matthewturk@gmail.com>
Affiliation: Columbia University
Homepage: http://yt-project.org/
License:
  Copyright (C) 2012 Matthew Turk.  All Rights Reserved.

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

import logging
import os
import hashlib
import contextlib
import urllib2

from .plugin import AnswerTesting, run_big_data
from yt.testing import *
from yt.config import ytcfg
from yt.mods import *
import cPickle

mylog = logging.getLogger('nose.plugins.answer-testing')

_latest = "SomeValue"
_url_path = "http://yt-answer-tests.s3-website-us-east-1.amazonaws.com/%s_%s"

class AnswerTestOpener(object):
    def __init__(self, reference_name):
        self.reference_name = reference_name
        self.cache = {}

    def get(self, pf_name, default = None):
        if pf_name in self.cache: return self.cache[pf_name]
        url = _url_path % (self.reference_name, pf_name)
        try:
            resp = urllib2.urlopen(url)
            # This is dangerous, but we have a controlled S3 environment
            data = resp.read()
            rv = cPickle.loads(data)
        except urllib2.HTTPError as ex:
            raise YTNoOldAnswer(url)
            mylog.warning("Missing %s (%s)", url, ex)
            rv = default
        self.cache[pf_name] = rv
        return rv

@contextlib.contextmanager
def temp_cwd(cwd):
    oldcwd = os.getcwd()
    os.chdir(cwd)
    yield
    os.chdir(oldcwd)

def can_run_pf(pf_fn):
    path = ytcfg.get("yt", "test_data_dir")
    with temp_cwd(path):
        try:
            load(pf_fn)
        except:
            return False
    return AnswerTestingTest.result_storage is not None

def data_dir_load(pf_fn):
    path = ytcfg.get("yt", "test_data_dir")
    with temp_cwd(path):
        pf = load(pf_fn)
        pf.h
        return pf

class AnswerTestingTest(object):
    reference_storage = None
    def __init__(self, pf_fn):
        self.pf = data_dir_load(pf_fn)

    def __call__(self):
        nv = self.run()
        if self.reference_storage is not None:
            dd = self.reference_storage.get(str(self.pf))
            if dd is None: raise YTNoOldAnswer()
            ov = dd[self.description]
            self.compare(nv, ov)
        else:
            ov = None
        self.result_storage[str(self.pf)][self.description] = nv

    def compare(self, new_result, old_result):
        raise RuntimeError

    def create_obj(self, pf, obj_type):
        # obj_type should be tuple of
        #  ( obj_name, ( args ) )
        if obj_type is None:
            return pf.h.all_data()
        cls = getattr(pf.h, obj_type[0])
        obj = cls(*obj_type[1])
        return obj

    @property
    def sim_center(self):
        """
        This returns the center of the domain.
        """
        return 0.5*(self.pf.domain_right_edge + self.pf.domain_left_edge)

    @property
    def max_dens_location(self):
        """
        This is a helper function to return the location of the most dense
        point.
        """
        return self.pf.h.find_max("Density")[1]

    @property
    def entire_simulation(self):
        """
        Return an unsorted array of values that cover the entire domain.
        """
        return self.pf.h.all_data()

    @property
    def description(self):
        obj_type = getattr(self, "obj_type", None)
        if obj_type is None:
            oname = "all"
        else:
            oname = "_".join((str(s) for s in obj_type))
        args = [self._type_name, str(self.pf), oname]
        args += [str(getattr(self, an)) for an in self._attrs]
        return "_".join(args)
        
class FieldValuesTest(AnswerTestingTest):
    _type_name = "FieldValues"
    _attrs = ("field", )

    def __init__(self, pf_fn, field, obj_type = None):
        super(FieldValuesTest, self).__init__(pf_fn)
        self.obj_type = obj_type
        self.field = field

    def run(self):
        obj = self.create_obj(self.pf, self.obj_type)
        avg = obj.quantities["WeightedAverageQuantity"](self.field,
                             weight="Ones")
        (mi, ma), = obj.quantities["Extrema"](self.field)
        return np.array([avg, mi, ma])

    def compare(self, new_result, old_result):
        assert_equal(new_result, old_result)

class ProjectionValuesTest(AnswerTestingTest):
    _type_name = "ProjectionValues"
    _attrs = ("field", "axis", "weight_field")

    def __init__(self, pf_fn, axis, field, weight_field = None,
                 obj_type = None):
        super(ProjectionValuesTest, self).__init__(pf_fn)
        self.axis = axis
        self.field = field
        self.weight_field = field
        self.obj_type = obj_type

    def run(self):
        if self.obj_type is not None:
            obj = self.create_obj(self.pf, self.obj_type)
        else:
            obj = None
        proj = self.pf.h.proj(self.axis, self.field,
                              weight_field=self.weight_field,
                              data_source = obj)
        return proj.field_data

    def compare(self, new_result, old_result):
        assert(len(new_result) == len(old_result))
        for k in new_result:
            assert (k in old_result)
        for k in new_result:
            assert_equal(new_result[k], old_result[k])

class PixelizedProjectionValuesTest(AnswerTestingTest):
    _type_name = "PixelizedProjectionValues"
    _attrs = ("field", "axis", "weight_field")

    def __init__(self, pf_fn, axis, field, weight_field = None,
                 obj_type = None):
        super(PixelizedProjectionValuesTest, self).__init__(pf_fn)
        self.axis = axis
        self.field = field
        self.weight_field = field
        self.obj_type = obj_type

    def run(self):
        if self.obj_type is not None:
            obj = self.create_obj(self.pf, self.obj_type)
        else:
            obj = None
        proj = self.pf.h.proj(self.axis, self.field,
                              weight_field=self.weight_field,
                              data_source = obj)
        frb = proj.to_frb((1.0, 'unitary'), 256)
        frb[self.field]
        frb[self.weight_field]
        d = frb.data
        d.update( dict( (("%s_sum" % f, proj[f].sum(dtype="float64"))
                         for f in proj.field_data.keys()) ) )
        return d

    def compare(self, new_result, old_result):
        assert(len(new_result) == len(old_result))
        for k in new_result:
            assert (k in old_result)
        for k in new_result:
            assert_rel_equal(new_result[k], old_result[k], 10)

class GridValuesTest(AnswerTestingTest):
    _type_name = "GridValues"
    _attrs = ("field",)

    def __init__(self, pf_fn, field):
        super(GridValuesTest, self).__init__(pf_fn)
        self.field = field

    def run(self):
        hashes = {}
        for g in self.pf.h.grids:
            hashes[g.id] = hashlib.md5(g[self.field].tostring()).hexdigest()
            g.clear_data()
        return hashes

    def compare(self, new_result, old_result):
        assert(len(new_result) == len(old_result))
        for k in new_result:
            assert (k in old_result)
        for k in new_result:
            assert_equal(new_result[k], old_result[k])

class GridHierarchyTest(AnswerTestingTest):
    _type_name = "GridHierarchy"
    _attrs = ()

    def run(self):
        result = {}
        result["grid_dimensions"] = self.pf.h.grid_dimensions
        result["grid_left_edges"] = self.pf.h.grid_left_edge
        result["grid_right_edges"] = self.pf.h.grid_right_edge
        result["grid_levels"] = self.pf.h.grid_levels
        result["grid_particle_count"] = self.pf.h.grid_particle_count
        return result

    def compare(self, new_result, old_result):
        for k in new_result:
            assert_equal(new_result[k], old_result[k])

class ParentageRelationshipsTest(AnswerTestingTest):
    _type_name = "ParentageRelationships"
    _attrs = ()
    def run(self):
        result = {}
        result["parents"] = []
        result["children"] = []
        for g in self.pf.h.grids:
            p = g.Parent
            if p is None:
                result["parents"].append(None)
            elif hasattr(p, "id"):
                result["parents"].append(p.id)
            else:
                result["parents"].append([pg.id for pg in p])
            result["children"].append([c.id for c in g.Children])
        return result

    def compare(self, new_result, old_result):
        for newp, oldp in zip(new_result["parents"], old_result["parents"]):
            assert(newp == oldp)
        for newc, oldc in zip(new_result["children"], old_result["children"]):
            assert(newp == oldp)

def requires_pf(pf_fn, big_data = False):
    def ffalse(func):
        return lambda: None
    def ftrue(func):
        return func
    if run_big_data == False and big_data == True:
        return ffalse
    elif not can_run_pf(pf_fn):
        return ffalse
    else:
        return ftrue

def small_patch_amr(pf_fn, fields):
    if not can_run_pf(pf_fn): return
    dso = [ None, ("sphere", ("max", (0.1, 'unitary')))]
    yield GridHierarchyTest(pf_fn)
    yield ParentageRelationshipsTest(pf_fn)
    for field in fields:
        yield GridValuesTest(pf_fn, field)
        for axis in [0, 1, 2]:
            for ds in dso:
                for weight_field in [None, "Density"]:
                    yield ProjectionValuesTest(
                        pf_fn, axis, field, weight_field,
                        ds)
                yield FieldValuesTest(
                        pf_fn, field, ds)

def big_patch_amr(pf_fn, fields):
    if not can_run_pf(pf_fn): return
    dso = [ None, ("sphere", ("max", (0.1, 'unitary')))]
    yield GridHierarchyTest(pf_fn)
    yield ParentageRelationshipsTest(pf_fn)
    for field in fields:
        yield GridValuesTest(pf_fn, field)
        for axis in [0, 1, 2]:
            for ds in dso:
                for weight_field in [None, "Density"]:
                    yield PixelizedProjectionValuesTest(
                        pf_fn, axis, field, weight_field,
                        ds)
