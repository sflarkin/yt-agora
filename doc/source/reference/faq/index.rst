.. _faq:


Frequently Asked Questions
==========================

A few problems that people encounter when installing or running ``yt``
come up regularly. Here are the solutions to some of the most common
problems.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

.. _determining-version:

How can I tell what version of ``yt`` I'm using?
------------------------------------------------

If you run into problems with ``yt`` and you're writing to the mailing list
or contacting developers on IRC, they will likely want to know what version of
``yt`` you're using.  Oftentimes, you'll want to know both the yt version, 
as well as the last changeset that was committed to the branch you're using.  
To reveal this, go to a command line and type:

.. code-block:: bash
    
    $ yt version

    yt module located at:
        /Users/username/src/yt-x86_64/src/yt-hg
    The supplemental repositories are located at:
        /Users/username/src/yt-x86_64/src/yt-supplemental

    The current version and changeset for the code is:

    ---
    Version = 2.7-dev
    Changeset = 6bffc737a67a
    ---

    This installation CAN be automatically updated.
    yt dependencies were last updated on
    Wed Dec  4 15:47:40 MST 2013

    To update all dependencies, run "yt update --all".

If the changeset is displayed followed by a "+", it means you have made 
modifications to the code since the last changeset.

.. _getting-sample-data:

How can I get some sample data for ``yt``?
------------------------------------------

Many different sample datasets can be found at http://yt-project.org/data/ .
These can be downloaded, unarchived, and they will each create their own
directory.  If you set the option ``test_data_dir``, in the section ``[yt]``,
in ``~/.yt/config``, ``yt`` will search this path for them.

This means you can download these datasets to ``/big_drive/data_for_yt`` , add
the appropriate item to ``~/.yt/config``, and no matter which directory you are
in when running ``yt``, it will also check in *that* directory.

.. _faq-scroll-up:

I can't scroll-up to previous commands inside iyt
-------------------------------------------------

If the up-arrow key does not recall the most recent commands, there is
probably an issue with the readline library. To ensure the yt python
environment can use readline, run the following command:

.. code-block:: bash

   $ ~/yt/bin/pip install readline

.. _faq-new-field:

How do I modify whether or not ``yt`` takes the log of a particular field?
--------------------------------------------------------------------------

``yt`` sets up defaults for many fields for whether or not a field is presented
in log or linear space. To override this behavior, you can modify the
``field_info`` dictionary.  For example, if you prefer that ``Density`` not be
logged, you could type:

.. code-block:: python
    
    pf = load("my_data")
    pf.h
    pf.field_info['density'].take_log = False

From that point forward, data products such as slices, projections, etc., would
be presented in linear space. Note that you have to instantiate pf.h before you
can access pf.field info.

.. _faq-handling-log-vs-linear-space:

I added a new field to my simulation data, can ``yt`` see it?
-------------------------------------------------------------

Yes! ``yt`` identifies all the fields in the simulation's output file
and will add them to its ``field_list`` even if they aren't listed in
:ref:`field-list`. These can then be accessed in the usual manner. For
example, if you have created a field for the potential called
``PotentialField``, you could type:

.. code-block:: python

   pf = load("my_data")
   dd = pf.h.all_data()
   potential_field = dd["PotentialField"]

The same applies to fields you might derive inside your ``yt`` script
via :ref:`creating-derived-fields`. To check what fields are
available, look at the properties ``field_list`` and ``derived_field_list``:

.. code-block:: python

   print pf.h.field_list
   print pf.h.derived_field_list

.. _faq-old-data:

``yt`` seems to be plotting from old data
------------------------------------------

``yt`` does check the time stamp of the simulation so that if you
overwrite your data outputs, the new set will be read in fresh by
``yt``. However, if you have problems or the ``yt`` output seems to be
in someway corrupted, try deleting the ``.yt`` and
``.harray`` files from inside your data directory. If this proves to
be a persistent problem add the line:

.. code-block:: python

   from yt.config import ytcfg; ytcfg["yt","serialize"] = "False"

to the very top of your ``yt`` script. 

.. _faq-mpi4py:

``yt`` complains that it needs the mpi4py module
------------------------------------------------

For ``yt`` to be able to incorporate parallelism on any of its analysis, 
it needs to be able to use MPI libraries.  This requires the ``mpi4py``
module to be installed in your version of python.  Unfortunately, 
installation of ``mpi4py`` is *just* tricky enough to elude the yt
batch installer.  So if you get an error in yt complaining about mpi4py like:

.. code-block:: bash

    ImportError: No module named mpi4py

then you should install ``mpi4py``.  The easiest way to install it is through
the pip interface.  At the command line, type:

.. code-block:: bash

    pip install mpi4py

What this does is it finds your default installation of python (presumably
in the yt source directory), and it installs the mpi4py module.  If this
action is successful, you should never have to worry about your aforementioned
problems again.  If, on the other hand, this installation fails (as it does on
such machines as NICS Kraken, NASA Pleaides and more), then you will have to
take matters into your own hands.  Usually when it fails, it is due to pip
being unable to find your MPI C/C++ compilers (look at the error message).
If this is the case, you can specify them explicitly as per:

.. code-block:: bash

    env MPICC=/path/to/MPICC pip install mpi4py

So for example, on Kraken, I switch to the gnu C compilers (because yt 
doesn't work with the portland group C compilers), then I discover that
cc is the mpi-enabled C compiler (and it is in my path), so I run:

.. code-block:: bash

    module swap PrgEnv-pgi PrgEnv-gnu
    env MPICC=cc pip install mpi4py

And voila!  It installs!  If this *still* fails for you, then you can 
build and install from source and specify the mpi-enabled c and c++ 
compilers in the mpi.cfg file.  See the `mpi4py installation page <http://mpi4py.scipy.org/docs/usrman/install.html>`_ for details.

``yt`` fails saying that it cannot import yt modules
----------------------------------------------------

This is likely because you need to rebuild the source.  You can do 
this automatically by running:

.. code-block:: bash

    cd $YT_DEST/src/yt-hg
    python setup.py develop


Unresolved Installation Problem on OSX 10.6
-------------------------------------------
When installing on some instances of OSX 10.6, a few users have noted a failure
when yt tries to build with OpenMP support:

    Symbol not found: _GOMP_barrier
        Referenced from: <YT_DEST>/src/yt-hg/yt/utilities/lib/grid_traversal.so

        Expected in: dynamic lookup

To resolve this, please make a symbolic link:

.. code-block:: bash

  $ ln -s /usr/local/lib/x86_64 <YT_DEST>/lib64

where ``<YT_DEST>`` is replaced by the path to the root of the directory
containing the yt install, which will usually be ``yt-<arch>``. After doing so, 
you should be able to cd to <YT_DEST>/src/yt-hg and run:

.. code-block:: bash

  $ python setup.py install

.. _plugin-file:

What is the "Plugin File"?
--------------------------

The plugin file is a means of modifying the available fields, quantities, data
objects and so on without modifying the source code of yt.  The plugin file
will be executed if it is detected, and it must be:

.. code-block:: bash

   $HOME/.yt/my_plugins.py

The code in this file can thus add fields, add derived quantities, add
datatypes, and on and on.  It is executed at the bottom of ``yt.mods``, and so
it is provided with the entire namespace available in the module ``yt.mods`` --
which is the primary entry point to yt, and which contains most of the
functionality of yt.  For example, if I created a plugin file containing:

.. code-block:: python

   def _myfunc(field, data):
       return np.random.random(data["density"].shape)
   add_field("SomeQuantity", function=_myfunc)

then all of my data objects would have access to the field "SomeQuantity"
despite its lack of use.

You can also define other convenience functions in your plugin file.  For
instance, you could define some variables or functions, and even import common
modules:

.. code-block:: python

   import os

   HOMEDIR="/home/username/"
   RUNDIR="/scratch/runs/"

   def load_run(fn):
       if not os.path.exists(RUNDIR + fn):
           return None
       return load(RUNDIR + fn)

In this case, we've written ``load_run`` to look in a specific directory to see
if it can find an output with the given name.  So now we can write scripts that
use this function:

.. code-block:: python

   from yt.mods import *

   my_run = load_run("hotgasflow/DD0040/DD0040")

And because we have imported from ``yt.mods`` we have access to the
``load_run`` function defined in our plugin file.

How do I cite yt?
-----------------

If you use yt in a publication, we'd very much appreciate a citation!  You
should feel free to cite the `ApJS paper
<http://adsabs.harvard.edu/abs/2011ApJS..192....9T>`_ with the following BibTeX
entry: ::

   @ARTICLE{2011ApJS..192....9T,
      author = {{Turk}, M.~J. and {Smith}, B.~D. and {Oishi}, J.~S. and {Skory}, S. and 
   	{Skillman}, S.~W. and {Abel}, T. and {Norman}, M.~L.},
       title = "{yt: A Multi-code Analysis Toolkit for Astrophysical Simulation Data}",
     journal = {\apjs},
   archivePrefix = "arXiv",
      eprint = {1011.3514},
    primaryClass = "astro-ph.IM",
    keywords = {cosmology: theory, methods: data analysis, methods: numerical },
        year = 2011,
       month = jan,
      volume = 192,
       pages = {9-+},
         doi = {10.1088/0067-0049/192/1/9},
      adsurl = {http://adsabs.harvard.edu/abs/2011ApJS..192....9T},
     adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }

If you use the Parallel Halo Finder, we have a 
`separate paper <http://adsabs.harvard.edu/abs/2010ApJS..191...43S>`_ that describes
its implementation: ::

   @ARTICLE{2010ApJS..191...43S,
      author = {{Skory}, S. and {Turk}, M.~J. and {Norman}, M.~L. and {Coil}, A.~L.
   	},
       title = "{Parallel HOP: A Scalable Halo Finder for Massive Cosmological Data Sets}",
     journal = {\apjs},
   archivePrefix = "arXiv",
      eprint = {1001.3411},
    primaryClass = "astro-ph.CO",
    keywords = {galaxies: halos, methods: data analysis, methods: numerical },
        year = 2010,
       month = nov,
      volume = 191,
       pages = {43-57},
         doi = {10.1088/0067-0049/191/1/43},
      adsurl = {http://adsabs.harvard.edu/abs/2010ApJS..191...43S},
     adsnote = {Provided by the SAO/NASA Astrophysics Data System}
   }
