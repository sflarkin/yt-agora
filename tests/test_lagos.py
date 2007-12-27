"""
Test that we can get outputs, and interact with them in some primitive ways.
"""

# @TODO: Add unit test for deleting field from fieldInfo
# @TODO: Profile unit testing, including for small spheres

import unittest, glob, os.path, os, sys, StringIO

print "Reporting from %s" % (os.getcwd())
sys.path = ['.'] + sys.path

from yt.config import ytcfg
ytcfg["yt","LogLevel"] = '50'
ytcfg["yt","logFile"] = "False"
ytcfg["yt","suppressStreamLogging"] = "True"
ytcfg["lagos","serialize"] = "False"

import yt.lagos
import numpy as na

# The dataset used is located at:
# http://yt.spacepope.org/DD0018.zip
fn = "DD0000/moving7_0000"
fn = os.path.join(os.path.dirname(__file__), fn)

class LagosTestingBase:
    def setUp(self):
        self.OutputFile = yt.lagos.EnzoStaticOutput(fn)
        self.hierarchy = self.OutputFile.hierarchy
        self.v, self.c = self.hierarchy.find_max("Density")
        gp = os.path.join(os.path.dirname(fn),"*.yt")
        ytFiles = glob.glob(gp)
        for i in ytFiles:
            #print "Removing %s" % (i)
            os.unlink(i)

    def tearDown(self):
        del self.OutputFile, self.hierarchy

class TestHierarchy(LagosTestingBase, unittest.TestCase):
    def testGetHierarchy(self):
        self.assert_(self.OutputFile.hierarchy != None)

    def testGetUnits(self):
        self.assert_(self.OutputFile["cm"] != 1.0)

    def testGetSmallestDx(self):
        self.assertAlmostEqual(self.hierarchy.get_smallest_dx(),
                               0.015625, 5)

    def testGetNumberOfGrids(self):
        self.assertEqual(self.hierarchy.num_grids, len(self.hierarchy.grids))
        self.assertEqual(self.hierarchy.num_grids, 3)

    def testChildrenOfRootGrid(self):
        for child in self.hierarchy.grids[0].Children:
            self.assert_(child.Parent.id == self.hierarchy.grids[0].id)

    def testGetSelectLevels(self):
        for level in range(self.hierarchy.maxLevel+1):
            for grid in self.hierarchy.select_grids(level):
                self.assert_(grid.Level == level)

    def testPrintStats(self):
        a = sys.stdout
        sys.stdout = StringIO.StringIO()
        try:
            self.hierarchy.print_stats()
            worked = True
        except:
            worked = False
        sys.stdout = a
        self.assert_(worked)

    def testDataTypes(self):
        r=self.hierarchy.region(
                     [0.5,0.5,0.5],[0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0],
                     ["CellMass","Temperature"])
            # Testing multiple fields fed in
        s=self.hierarchy.sphere(
                     [0.5,0.5,0.5],2.0,
                     ["CellMass","Temperature"])
        ms = s["CellMass"].sum() # Testing adding new field transparently
        mr = r["CellMass"].sum() # Testing adding new field transparently
        self.assertEqual(ms,mr)  # Asserting equality between the two

    def testProjectionMaking(self):
        p = self.hierarchy.proj(0,"Density") # Unweighted
        p = self.hierarchy.proj(1,"Temperature","Density") # Weighted
        p = self.hierarchy.proj(2,"Entropy") # Derived field

    def testProjectionCorrectness(self):
        # Now we test that we get good answers
        for axis in range(3):
            p = self.hierarchy.proj(axis, "Ones") # Derived field
            self.assertAlmostEqual(p["Ones"].prod(), 1.0, 7)

# Now we test each datatype in turn

def _returnFieldFunction(field):
    def field_function(self):
        try:
            self.data[field.name]
            if not field.variable_length and self.data[field.name].size > 1:
                self.assertEqual(na.product(self.data["Density"].shape),
                                 na.product(self.data[field.name].shape))
            del self.data[field.name]
        except yt.lagos.ValidationException:
            pass
    return field_function

class DataTypeTestingBase:
    def setUp(self):
        LagosTestingBase.setUp(self)
for field in yt.lagos.fieldInfo.values():
    #if field.name.find("particle") > -1:
        #continue
    func = _returnFieldFunction(field)
    setattr(DataTypeTestingBase, "test%s" % field.name, func)

class TestRegionDataType(DataTypeTestingBase, LagosTestingBase, unittest.TestCase):
    def setUp(self):
        DataTypeTestingBase.setUp(self)
        self.data=self.hierarchy.region(
                     [0.5,0.5,0.5],[0.0, 0.0, 0.0],
                     [1.0, 1.0, 1.0])
    def testVolume(self):
        vol = self.data["CellVolume"].sum() / self.data.convert("cm")**3.0
        self.assertAlmostEqual(vol,1.0,7)

class TestSphereDataType(DataTypeTestingBase, LagosTestingBase, unittest.TestCase):
    def setUp(self):
        DataTypeTestingBase.setUp(self)
        self.data=self.hierarchy.sphere([0.5,0.5,0.5],2.0)
    def testVolume(self):
        vol = self.data["CellVolume"].sum() / self.data.convert("cm")**3.0
        self.assertAlmostEqual(vol,1.0,7)

class TestSliceDataType(DataTypeTestingBase, LagosTestingBase, unittest.TestCase):
    def setUp(self):
        DataTypeTestingBase.setUp(self)
        self.data = self.hierarchy.slice(0,0.5)

class TestGridDataType(DataTypeTestingBase, LagosTestingBase, unittest.TestCase):
    def setUp(self):
        DataTypeTestingBase.setUp(self)
        self.data = self.hierarchy.grids[0]

class TestExtractFromSphere(TestSphereDataType):
    def setUp(self):
        TestSphereDataType.setUp(self)
        self.region = self.data
        self.ind_to_get = na.where(self.region["Temperature"]>500)
        self.data = self.region.extract_region(self.ind_to_get)
    def testNumberOfEntries(self):
        self.assertEqual(self.ind_to_get[0].shape,
                        self.data["Density"].shape)
    def testVolume(self):
        self.ind_to_get = na.where(self.region["CellVolume"]>0.0)
        vol = self.region.extract_region(self.ind_to_get)["CellVolume"].sum() \
            / self.data.convert("cm")**3.0
        self.assertAlmostEqual(vol,1.0,7)

class TestExtractFromRegion(TestRegionDataType):
    def setUp(self):
        TestRegionDataType.setUp(self)
        self.region = self.data
        self.ind_to_get = na.where(self.region["Temperature"]>500)
        self.data = self.region.extract_region(self.ind_to_get)
    def testNumberOfEntries(self):
        self.assertEqual(self.ind_to_get[0].shape,
                        self.data["Density"].shape)
    def testVolume(self):
        ind_to_get = na.where(self.region["CellVolume"]>0.0)
        vol = self.region.extract_region(ind_to_get)["CellVolume"].sum() \
            / self.data.convert("cm")**3.0
        self.assertAlmostEqual(vol,1.0,7)


if __name__ == "__main__":
    unittest.main()
