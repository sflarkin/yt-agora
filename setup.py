from distutils.core import setup, Extension
import numpy
import sys, time

numpyincludedirs=numpy.get_include()

if not hasattr(sys, 'version_info') or sys.version_info < (2,2,0,'alpha',0):
    raise SysError

_ec = Extension('yt.lagos.EnzoCombine',['src/lagos/EnzoCombine.c'], \
                include_dirs=[numpyincludedirs], \
                libraries=['m'])

setup(name = "yt",
    version = time.strftime("%y%m%d"),
    description = "A set of classes for manipulating Enzo data",
    url = "http://www.stanford.edu/~mturk/raven.html",
    author="Matthew Turk",
    author_email="mturk@stanford.edu",
    package_dir={"yt":"src"},
    packages = ["yt.enki", "yt.enki.mes", "yt.fido", "yt.lagos", "yt", "yt.raven", "yt.deliverator"],
    scripts = ["scripts/fdigup","scripts/fimport","scripts/frevert","scripts/fbranch", 
               "scripts/ffetch", "scripts/yt", "scripts/fbury","scripts/fido"],
    data_files = [('share/doc/yt/', ['examples/test_enki.py',
      'examples/test_fido.py', 'examples/test_raven.py', 'examples/test_lagos.py',
      'examples/get_particles_enzorun.py'])],
    license="GPL-2",
    ext_modules=[_ec]
    )
