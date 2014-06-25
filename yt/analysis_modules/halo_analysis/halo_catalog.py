"""
HaloCatalog object



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import h5py
import numpy as np
import os

from yt.funcs import \
     ensure_dir, \
     mylog
from yt.utilities.parallel_tools.parallel_analysis_interface import \
     ParallelAnalysisInterface, \
     parallel_blocking_call, \
     parallel_objects
     
from .halo_object import \
     Halo
from .operator_registry import \
     callback_registry, \
     filter_registry, \
     finding_method_registry, \
     quantity_registry

class HaloCatalog(ParallelAnalysisInterface):
    r"""Create a HaloCatalog: an object that allows for the creation and association
    of data with a set of halo objects.

    A HaloCatalog object pairs a simulation dataset and the output from a halo finder, 
    allowing the user to perform analysis on each of the halos found by the halo finder.
    Analysis is performed by providing callbacks: functions that accept a Halo object 
    and perform independent analysis, return a quantity to be associated with the halo, 
    or return True or False whether a halo meets various criteria.  The resulting set of 
    quantities associated with each halo is then written out to disk at a "halo catalog."
    This halo catalog can then be loaded in with yt as any other simulation dataset.
    
    Parameters
    ----------
    halos_pf : str
        Dataset created by a halo finder.  If None, a halo finder should be 
        provided with the finder_method keyword.
    data_pf : str
        Dataset created by a simulation.
    data_source : data container
        Data container associated with either the halos_pf or the data_pf.
    finder_method : str
        Halo finder to be used if no halos_pf is given.
    output_dir : str
        The top level directory into which analysis output will be written.
        Default: "."
    finder_kwargs : dict
        Arguments to pass to the halo finder if finder_method is given.

    Examples
    --------

    # create profiles or overdensity vs. radius for each halo and save to disk
    >>> from yt.mods import *
    >>> from yt.analysis_modules.halo_analysis.api import *
    >>> data_pf = load("DD0064/DD0064")
    >>> halos_pf = load("rockstar_halos/halos_64.0.bin",
    ...                 output_dir="halo_catalogs/catalog_0064")
    >>> hc = HaloCatalog(data_pf=data_pf, halos_pf=halos_pf)
    # filter out halos with mass < 1e13 Msun
    >>> hc.add_filter("quantity_value", "particle_mass", ">", 1e13, "Msun")
    # create a sphere object with radius of 2 times the virial_radius field
    >>> hc.add_callback("sphere", factor=2.0, radius_field="virial_radius")
    # make radial profiles
    >>> hc.add_callback("profile", "radius", [("gas", "overdensity")],
    ...                 weight_field="cell_volume", accumulation=True)
    # save the profiles to disk
    >>> hc.add_callback("save_profiles", output_dir="profiles")
    # create the catalog
    >>> hc.create()


    # load in the saved halo catalog and all the profile data
    >>> halos_pf = load("halo_catalogs/catalog_0064/catalog_0064.0.h5")
    >>> hc = HaloCatalog(halos_pf=halos_pf,
                         output_dir="halo_catalogs/catalog_0064")
    >>> hc.add_callback("load_profiles", output_dir="profiles")
    >>> hc.load()

    See Also
    --------
    add_callback, add_filter, add_finding_method, add_quantity
    
    """
    
    def __init__(self, halos_pf=None, data_pf=None, 
                 data_source=None, finder_method=None, 
                 finder_kwargs=None,
                 output_dir="halo_catalogs/catalog"):
        ParallelAnalysisInterface.__init__(self)
        self.halos_pf = halos_pf
        self.data_pf = data_pf
        self.output_dir = ensure_dir(output_dir)
        if os.path.basename(self.output_dir) != ".":
            self.output_prefix = os.path.basename(self.output_dir)
        else:
            self.output_prefix = "catalog"

        if halos_pf is None:
            if data_pf is None:
                raise RuntimeError("Must specify a halos_pf, data_pf, or both.")
            if finder_method is None:
                raise RuntimeError("Must specify a halos_pf or a finder_method.")

        if data_source is None:
            if halos_pf is not None:
                data_source = halos_pf.h.all_data()
            else:
                data_source = data_pf.h.all_data()
        self.data_source = data_source

        if finder_method is not None:
            finder_method = finding_method_registry.find(finder_method, **finder_kwargs)
        self.finder_method = finder_method            
        
        # all of the analysis actions to be performed: callbacks, filters, and quantities
        self.actions = []
        # fields to be written to the halo catalog
        self.quantities = []
        if not self.halos_pf is None:
            self.add_default_quantities()

    def add_callback(self, callback, *args, **kwargs):
        r"""
        Add a callback to the halo catalog action list.

        A callback is a function that accepts and operates on a Halo object and 
        does not return anything.  Callbacks must exist within the callback_registry.  
        Give additional args and kwargs to be passed to the callback here.

        Parameters
        ----------
        callback : string
            The name of the callback.

        Examples
        --------

        # Here, a callback is defined and added to the registry.
        def _say_something(halo, message):
            my_id = halo.quantities['particle_identifier']
            print "Halo %d: here is a message - %s." % (my_id, message)
        add_callback("hello_world", _say_something)

        # Now this callback is accessible to the HaloCatalog object
        >>> hc.add_callback("hello_world", "this is my message")
            
        """
        callback = callback_registry.find(callback, *args, **kwargs)
        if "output_dir" in kwargs is not None:
            ensure_dir(os.path.join(self.output_dir, kwargs["output_dir"]))
        self.actions.append(("callback", callback))

    def add_quantity(self, key, *args, **kwargs):
        r"""
        Add a quantity to the halo catalog action list.

        A quantity is a function that accepts a Halo object and return a value or 
        values.  These values are stored in a "quantities" dictionary associated 
        with the Halo object.  Quantities must exist within the quantity_registry.  
        Give additional args and kwargs to be passed to the quantity function here.

        Parameters
        ----------
        key : string
            The name of the callback.
        field_type : string
            If not None, the quantity is the value of the field provided by the 
            key parameter, taken from the halo finder dataset.  This is the way 
            one pulls values for the halo from the halo dataset.
            Default : None

        Examples
        --------

        # pull the virial radius from the halo finder dataset
        hc.add_quantity("virial_radius", field_type="halos")

        # define a custom quantity and add it to the register
        def _mass_squared(halo):
            # assume some entry "particle_mass" exists in the quantities dict
            return halo.quantities["particle_mass"]**2
        add_quantity("mass_squared", _mass_squared)

        # add it to the halo catalog action list
        >>> hc.add_quantity("mass_squared")
        
        """
        
        if "field_type" in kwargs:
            field_type = kwargs.pop("field_type")
        else:
            field_type = None
        if field_type is None:
            quantity = quantity_registry.find(key, *args, **kwargs)
        elif (field_type, key) in self.halos_pf.field_info:
            quantity = (field_type, key)
        else:
            raise RuntimeError("HaloCatalog quantity must be a registered function or a field of a known type.")
        self.quantities.append(key)
        self.actions.append(("quantity", (key, quantity)))

    def add_filter(self, halo_filter, *args, **kwargs):
        r"""
        Add a filter to the halo catalog action list.

        A filter is a function that accepts a Halo object and returns either True 
        or False.  If True, any additional actions added to the list are carried out 
        and the results are added to the final halo catalog.  If False, any further 
        actions are skipped and the halo will be omitted from the final catalog.
        Filters must exist within the filter_registry.  Give additional args and kwargs 
        to be passed to the filter function here.

        Parameters
        ----------
        halo_filter : string
            The name of the filter.

        Examples
        --------

        # define a filter and add it to the register.
        def _my_filter(halo, mass_value):
            if halo.quantities["particle_mass"] > YTQuantity(mass_value, "Msun"):
                return True
            else:
                return False
        # add it to the register
        add_filter("mass_filter", _my_filter)

        # add the filter to the halo catalog actions
        >>> hc.add_filter("mass_value", 1e12)
        
        """
        
        halo_filter = filter_registry.find(halo_filter, *args, **kwargs)
        self.actions.append(("filter", halo_filter))

    def create(self, save_halos=False, save_catalog=True, njobs=-1, dynamic=False):
        r"""
        Create the halo catalog given the callbacks, quantities, and filters that 
        have been provided.

        This is a wrapper around the main _run function with default arguments tuned 
        for halo catalog creation.  By default, halo objects are not saved but the 
        halo catalog is written, opposite to the behavior of the load function.

        Parameters
        ----------
        save_halos : bool
            If True, a list of all Halo objects is retained under the "halo_list"
            attribute.  If False, only the compiles quantities are saved under the 
            "catalog" attribute.
            Default: False
        save_catalog : bool
            If True, save the final catalog to disk.
            Default: True
        njobs : int
            The number of jobs over which to divide halo analysis.  Choose -1 
            to allocate one processor per halo.
            Default: -1
        dynamic : int
            If False, halo analysis is divided evenly between all available processors.
            If True, parallelism is performed via a task queue.
            Default: False

        See Also
        --------
        load
        
        """
        self._run(save_halos, save_catalog, njobs=njobs, dynamic=dynamic)

    def load(self, save_halos=True, save_catalog=False, njobs=-1, dynamic=False):
        r"""
        Load a previously created halo catalog.

        This is a wrapper around the main _run function with default arguments tuned 
        for reloading halo catalogs and associated data.  By default, halo objects are 
        saved and the halo catalog is not written, opposite to the behavior of the 
        create function.
        
        Parameters
        ----------
        save_halos : bool
            If True, a list of all Halo objects is retained under the "halo_list"
            attribute.  If False, only the compiles quantities are saved under the 
            "catalog" attribute.
            Default: True
        save_catalog : bool
            If True, save the final catalog to disk.
            Default: False
        njobs : int
            The number of jobs over which to divide halo analysis.  Choose -1 
            to allocate one processor per halo.
            Default: -1
        dynamic : int
            If False, halo analysis is divided evenly between all available processors.
            If True, parallelism is performed via a task queue.
            Default: False

        See Also
        --------
        create
        
        """
        self._run(save_halos, save_catalog, njobs=njobs, dynamic=dynamic)
        
    @parallel_blocking_call
    def _run(self, save_halos, save_catalog, njobs=-1, dynamic=False):
        r"""
        Run the requested halo analysis.

        Parameters
        ----------
        save_halos : bool
            If True, a list of all Halo objects is retained under the "halo_list"
            attribute.  If False, only the compiles quantities are saved under the 
            "catalog" attribute.
        save_catalog : bool
            If True, save the final catalog to disk.
        njobs : int
            The number of jobs over which to divide halo analysis.  Choose -1 
            to allocate one processor per halo.
            Default: -1
        dynamic : int
            If False, halo analysis is divided evenly between all available processors.
            If True, parallelism is performed via a task queue.
            Default: False

        See Also
        --------
        create, load
        
        """
        self.catalog = []
        if save_halos: self.halo_list = []

        if self.halos_pf is None:
            # Find the halos and make a dataset of them
            self.halos_pf = self.finder_method(self.data_pf)
            if self.halos_pf is None:
                mylog.warning('No halos were found for {0}'.format(\
                        self.data_pf.basename))
                if save_catalog:
                    self.halos_pf = self.data_pf
                    self.save_catalog()
                    self.halos_pf = None
                return

            # Assign pf and data sources appropriately
            self.data_source = self.halos_pf.all_data()

            # Add all of the default quantities that all halos must have
            self.add_default_quantities('all')

        my_index = np.argsort(self.data_source["particle_identifier"])
        for i in parallel_objects(my_index, njobs=njobs, dynamic=dynamic):
            new_halo = Halo(self)
            halo_filter = True
            for action_type, action in self.actions:
                if action_type == "callback":
                    action(new_halo)
                elif action_type == "filter":
                    halo_filter = action(new_halo)
                    if not halo_filter: break
                elif action_type == "quantity":
                    key, quantity = action
                    if quantity in self.halos_pf.field_info:
                        new_halo.quantities[key] = \
                          self.data_source[quantity][int(i)].in_cgs()
                    elif callable(quantity):
                        new_halo.quantities[key] = quantity(new_halo)
                else:
                    raise RuntimeError("Action must be a callback, filter, or quantity.")

            if halo_filter:
                self.catalog.append(new_halo.quantities)

            if save_halos and halo_filter:
                self.halo_list.append(new_halo)
            else:
                del new_halo

        self.catalog.sort(key=lambda a:a['particle_identifier'].to_ndarray())
        if save_catalog:
            self.save_catalog()

    def save_catalog(self):
        "Write out hdf5 file with all halo quantities."

        filename = os.path.join(self.output_dir, "%s.%d.h5" %
                                (self.output_prefix, self.comm.rank))
        n_halos = len(self.catalog)
        mylog.info("Saving halo catalog (%d halos) to %s." %
                   (n_halos, os.path.join(self.output_dir, 
                                         self.output_prefix)))
        out_file = h5py.File(filename, 'w')
        for attr in ["current_redshift", "current_time",
                     "domain_dimensions",
                     "cosmological_simulation", "omega_lambda",
                     "omega_matter", "hubble_constant"]:
            out_file.attrs[attr] = getattr(self.halos_pf, attr)
        for attr in ["domain_left_edge", "domain_right_edge"]:
            out_file.attrs[attr] = getattr(self.halos_pf, attr).in_cgs()
        out_file.attrs["data_type"] = "halo_catalog"
        out_file.attrs["num_halos"] = n_halos
        if n_halos > 0:
            field_data = np.empty(n_halos)
            for key in self.quantities:
                units = ""
                if hasattr(self.catalog[0][key], "units"):
                    units = str(self.catalog[0][key].units)
                for i in xrange(n_halos):
                    field_data[i] = self.catalog[i][key]
                dataset = out_file.create_dataset(str(key), data=field_data)
                dataset.attrs["units"] = units
        out_file.close()

    def add_default_quantities(self, field_type='halos'):
        self.add_quantity("particle_identifier", field_type=field_type)
        self.add_quantity("particle_mass", field_type=field_type)
        self.add_quantity("particle_position_x", field_type=field_type)
        self.add_quantity("particle_position_y", field_type=field_type)
        self.add_quantity("particle_position_z", field_type=field_type)
        self.add_quantity("virial_radius", field_type=field_type)

