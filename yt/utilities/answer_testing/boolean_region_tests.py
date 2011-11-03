from yt.mods import *
import matplotlib
import pylab
from output_tests import SingleOutputTest, YTStaticOutputTest, create_test
import hashlib
import numpy as np

# Tests to make sure that grid quantities are identical that should
# be identical for the AND operator.
class TestBooleanANDGridQuantity(YTStaticOutputTest):
    def run(self):
        domain = self.pf.domain_right_edge - self.pf.domain_left_edge
        four = 0.4 * domain + self.pf.domain_left_edge
        five = 0.5 * domain + self.pf.domain_left_edge
        six = 0.6 * domain + self.pf.domain_left_edge
        re1 = self.pf.h.region_strict(five, four, six)
        re2 = self.pf.h.region_strict(five, five, six)
        re = self.pf.h.boolean([re1, "AND", re2])
        # re should look like re2.
        x2 = re2['x']
        x = re['x']
        x2 = x2[x2.argsort()]
        x = x[x.argsort()]
        self.result = (x2, x)
    
    def compare(self, old_result):
        self.compare_array_delta(self.result[0], self.result[1], 1e-10)
    
    def plot(self):
        return []

# OR
class TestBooleanORGridQuantity(YTStaticOutputTest):
    def run(self):
        domain = self.pf.domain_right_edge - self.pf.domain_left_edge
        four = 0.4 * domain + self.pf.domain_left_edge
        five = 0.5 * domain + self.pf.domain_left_edge
        six = 0.6 * domain + self.pf.domain_left_edge
        re1 = self.pf.h.region_strict(five, four, six)
        re2 = self.pf.h.region_strict(five, five, six)
        re = self.pf.h.boolean([re1, "OR", re2])
        # re should look like re1
        x1 = re1['x']
        x = re['x']
        x1 = x1[x1.argsort()]
        x = x[x.argsort()]
        self.result = (x1, x)
    
    def compare(self, old_result):
        self.compare_array_delta(self.result[0], self.result[1], 1e-10)
    
    def plot(self):
        return []

# NOT
class TestBooleanNOTGridQuantity(YTStaticOutputTest):
    def run(self):
        domain = self.pf.domain_right_edge - self.pf.domain_left_edge
        four = 0.4 * domain + self.pf.domain_left_edge
        five = 0.5 * domain + self.pf.domain_left_edge
        six = 0.6 * domain + self.pf.domain_left_edge
        re1 = self.pf.h.region_strict(five, four, six)
        re2 = self.pf.h.region_strict(five, five, six)
        # Bottom base
        re3 = self.pf.h.region_strict(five, four, [six[0], six[1], five[2]])
        # Side
        re4 = self.pf.h.region_strict(five, [four[0], four[1], five[2]],
            [five[0], six[1], six[2]])
        # Last small cube
        re5 = self.pf.h.region_strict(five, [five[0], four[0], four[2]],
            [six[0], five[1], six[2]])
        # re1 NOT re2 should look like re3 OR re4 OR re5
        re = self.pf.h.boolean([re1, "NOT", re2])
        reo = self.pf.h.boolean([re3, "OR", re4, "OR", re5])
        x = re['x']
        xo = reo['x']
        x = x[x.argsort()]
        xo = xo[xo.argsort()]
        self.result = (x, xo)
    
    def compare(self, old_result):
        self.compare_array_delta(self.result[0], self.result[1], 1e-10)
    
    def plot(self):
        return []

# Tests to make sure that particle quantities are identical that should
# be identical for the AND operator.
class TestBooleanANDParticleQuantity(YTStaticOutputTest):
    def run(self):
        domain = self.pf.domain_right_edge - self.pf.domain_left_edge
        four = 0.4 * domain + self.pf.domain_left_edge
        five = 0.5 * domain + self.pf.domain_left_edge
        six = 0.6 * domain + self.pf.domain_left_edge
        re1 = self.pf.h.region_strict(five, four, six)
        re2 = self.pf.h.region_strict(five, five, six)
        re = self.pf.h.boolean([re1, "AND", re2])
        # re should look like re2.
        x2 = re2['particle_position_x']
        x = re['particle_position_x']
        x2 = x2[x2.argsort()]
        x = x[x.argsort()]
        self.result = (x2, x)
    
    def compare(self, old_result):
        self.compare_array_delta(self.result[0], self.result[1], 1e-10)
    
    def plot(self):
        return []

# OR
class TestBooleanORParticleQuantity(YTStaticOutputTest):
    def run(self):
        domain = self.pf.domain_right_edge - self.pf.domain_left_edge
        four = 0.4 * domain + self.pf.domain_left_edge
        five = 0.5 * domain + self.pf.domain_left_edge
        six = 0.6 * domain + self.pf.domain_left_edge
        re1 = self.pf.h.region_strict(five, four, six)
        re2 = self.pf.h.region_strict(five, five, six)
        re = self.pf.h.boolean([re1, "OR", re2])
        # re should look like re1
        x1 = re1['particle_position_x']
        x = re['particle_position_x']
        x1 = x1[x1.argsort()]
        x = x[x.argsort()]
        self.result = (x1, x)
    
    def compare(self, old_result):
        self.compare_array_delta(self.result[0], self.result[1], 1e-10)
    
    def plot(self):
        return []

# NOT
class TestBooleanNOTParticleQuantity(YTStaticOutputTest):
    def run(self):
        domain = self.pf.domain_right_edge - self.pf.domain_left_edge
        four = 0.4 * domain + self.pf.domain_left_edge
        five = 0.5 * domain + self.pf.domain_left_edge
        six = 0.6 * domain + self.pf.domain_left_edge
        re1 = self.pf.h.region_strict(five, four, six)
        re2 = self.pf.h.region_strict(five, five, six)
        # Bottom base
        re3 = self.pf.h.region_strict(five, four, [six[0], six[1], five[2]])
        # Side
        re4 = self.pf.h.region_strict(five, [four[0], four[1], five[2]],
            [five[0], six[1], six[2]])
        # Last small cube
        re5 = self.pf.h.region_strict(five, [five[0], four[0], four[2]],
            [six[0], five[1], six[2]])
        # re1 NOT re2 should look like re3 OR re4 OR re5
        re = self.pf.h.boolean([re1, "NOT", re2])
        reo = self.pf.h.boolean([re3, "OR", re4, "OR", re5])
        x = re['particle_position_x']
        xo = reo['particle_position_x']
        x = x[x.argsort()]
        xo = xo[xo.argsort()]
        self.result = (x, xo)
    
    def compare(self, old_result):
        self.compare_array_delta(self.result[0], self.result[1], 1e-10)
    
    def plot(self):
        return []

