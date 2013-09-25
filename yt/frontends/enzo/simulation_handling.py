"""
EnzoSimulation class and member functions.



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.funcs import *

import numpy as np
import glob
import os

from yt.data_objects.time_series import \
    SimulationTimeSeries, TimeSeriesData
from yt.utilities.cosmology import \
    Cosmology, \
    EnzoCosmology
from yt.utilities.definitions import \
    sec_conversion
from yt.utilities.exceptions import \
    InvalidSimulationTimeSeries, \
    MissingParameter, \
    NoStoppingCondition
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_objects

from yt.convenience import \
    load

class EnzoSimulation(SimulationTimeSeries):
    r"""Class for creating TimeSeriesData object from an Enzo
    simulation parameter file.
    """
    def __init__(self, parameter_filename, find_outputs=False):
        r"""Initialize an Enzo Simulation object.

        Upon creation, the parameter file is parsed and the time and redshift
        are calculated and stored in all_outputs.  A time units dictionary is
        instantiated to allow for time outputs to be requested with physical
        time units.  The get_time_series can be used to generate a
        TimeSeriesData object.

        parameter_filename : str
            The simulation parameter file.
        find_outputs : bool
            If True, subdirectories within the GlobalDir directory are
            searched one by one for datasets.  Time and redshift
            information are gathered by temporarily instantiating each
            dataset.  This can be used when simulation data was created
            in a non-standard way, making it difficult to guess the
            corresponding time and redshift information.
            Default: False.

        Examples
        --------
        >>> from yt.mods import *
        >>> es = ES.EnzoSimulation("my_simulation.par")
        >>> print es.all_outputs

        """
        SimulationTimeSeries.__init__(self, parameter_filename,
                                      find_outputs=find_outputs)

    def get_time_series(self, time_data=True, redshift_data=True,
                        initial_time=None, final_time=None, time_units='1',
                        initial_redshift=None, final_redshift=None,
                        initial_cycle=None, final_cycle=None,
                        times=None, redshifts=None, tolerance=None,
                        parallel=True):

        """
        Instantiate a TimeSeriesData object for a set of outputs.

        If no additional keywords given, a TimeSeriesData object will be
        created with all potential datasets created by the simulation.

        Outputs can be gather by specifying a time or redshift range
        (or combination of time and redshift), with a specific list of
        times or redshifts, a range of cycle numbers (for cycle based
        output), or by simply searching all subdirectories within the
        simulation directory.

        time_data : bool
            Whether or not to include time outputs when gathering
            datasets for time series.
            Default: True.
        redshift_data : bool
            Whether or not to include redshift outputs when gathering
            datasets for time series.
            Default: True.
        initial_time : float
            The earliest time for outputs to be included.  If None,
            the initial time of the simulation is used.  This can be
            used in combination with either final_time or
            final_redshift.
            Default: None.
        final_time : float
            The latest time for outputs to be included.  If None,
            the final time of the simulation is used.  This can be
            used in combination with either initial_time or
            initial_redshift.
            Default: None.
        times : array_like
            A list of times for which outputs will be found.
            Default: None.
        time_units : str
            The time units used for requesting outputs by time.
            Default: '1' (code units).
        initial_redshift : float
            The earliest redshift for outputs to be included.  If None,
            the initial redshift of the simulation is used.  This can be
            used in combination with either final_time or
            final_redshift.
            Default: None.
        final_time : float
            The latest redshift for outputs to be included.  If None,
            the final redshift of the simulation is used.  This can be
            used in combination with either initial_time or
            initial_redshift.
            Default: None.
        redshifts : array_like
            A list of redshifts for which outputs will be found.
            Default: None.
        initial_cycle : float
            The earliest cycle for outputs to be included.  If None,
            the initial cycle of the simulation is used.  This can
            only be used with final_cycle.
            Default: None.
        final_cycle : float
            The latest cycle for outputs to be included.  If None,
            the final cycle of the simulation is used.  This can
            only be used in combination with initial_cycle.
            Default: None.
        tolerance : float
            Used in combination with "times" or "redshifts" keywords,
            this is the tolerance within which outputs are accepted
            given the requested times or redshifts.  If None, the
            nearest output is always taken.
            Default: None.
        parallel : bool/int
            If True, the generated TimeSeriesData will divide the work
            such that a single processor works on each dataset.  If an
            integer is supplied, the work will be divided into that
            number of jobs.
            Default: True.

        Examples
        --------
        >>> es.get_time_series(initial_redshift=10, final_time=13.7,
                               time_units='Gyr', redshift_data=False)

        >>> es.get_time_series(redshifts=[3, 2, 1, 0])

        >>> es.get_time_series(final_cycle=100000)

        >>> es.get_time_series(find_outputs=True)

        >>> # after calling get_time_series
        >>> for pf in es.piter():
        >>>     pc = PlotCollection(pf, 'c')
        >>>     pc.add_projection('Density', 0)
        >>>     pc.save()

        """

        if (initial_redshift is not None or \
            final_redshift is not None) and \
            not self.cosmological_simulation:
            raise InvalidSimulationTimeSeries('An initial or final redshift has been given for a noncosmological simulation.')

        if time_data and redshift_data:
            my_all_outputs = self.all_outputs
        elif time_data:
            my_all_outputs = self.all_time_outputs
        elif redshift_data:
            my_all_outputs = self.all_redshift_outputs
        else:
            raise InvalidSimulationTimeSeries('Both time_data and redshift_data are False.')

        if not my_all_outputs:
            TimeSeriesData.__init__(self, outputs=[], parallel=parallel)
            mylog.info("0 outputs loaded into time series.")
            return

        # Apply selection criteria to the set.
        if times is not None:
            my_outputs = self._get_outputs_by_time(times, tolerance=tolerance,
                                                   outputs=my_all_outputs,
                                                   time_units=time_units)

        elif redshifts is not None:
            my_outputs = self._get_outputs_by_redshift(redshifts, tolerance=tolerance,
                                                       outputs=my_all_outputs)

        elif initial_cycle is not None or final_cycle is not None:
            if initial_cycle is None:
                initial_cycle = 0
            else:
                initial_cycle = max(initial_cycle, 0)
            if final_cycle is None:
                final_cycle = self.parameters['StopCycle']
            else:
                final_cycle = min(final_cycle, self.parameters['StopCycle'])

            my_outputs = my_all_outputs[int(ceil(float(initial_cycle) /
                                                 self.parameters['CycleSkipDataDump'])):
                                        (final_cycle /  self.parameters['CycleSkipDataDump'])+1]

        else:
            if initial_time is not None:
                my_initial_time = initial_time / self.time_units[time_units]
            elif initial_redshift is not None:
                my_initial_time = self.enzo_cosmology.ComputeTimeFromRedshift(initial_redshift) / \
                    self.enzo_cosmology.TimeUnits
            else:
                my_initial_time = self.initial_time

            if final_time is not None:
                my_final_time = final_time / self.time_units[time_units]
            elif final_redshift is not None:
                my_final_time = self.enzo_cosmology.ComputeTimeFromRedshift(final_redshift) / \
                    self.enzo_cosmology.TimeUnits
            else:
                my_final_time = self.final_time

            my_times = np.array(map(lambda a:a['time'], my_all_outputs))
            my_indices = np.digitize([my_initial_time, my_final_time], my_times)
            if my_initial_time == my_times[my_indices[0] - 1]: my_indices[0] -= 1
            my_outputs = my_all_outputs[my_indices[0]:my_indices[1]]

        init_outputs = []
        for output in my_outputs:
            if os.path.exists(output['filename']):
                init_outputs.append(output['filename'])
            
        TimeSeriesData.__init__(self, outputs=init_outputs, parallel=parallel)
        mylog.info("%d outputs loaded into time series.", len(init_outputs))

    def _parse_parameter_file(self):
        """
        Parses the parameter file and establishes the various
        dictionaries.
        """

        self.conversion_factors = {}
        redshift_outputs = []

        # Let's read the file
        lines = open(self.parameter_filename).readlines()
        for line in (l.strip() for l in lines):
            if '#' in line: line = line[0:line.find('#')]
            if '//' in line: line = line[0:line.find('//')]
            if len(line) < 2: continue
            param, vals = (i.strip() for i in line.split("=", 1))
            # First we try to decipher what type of value it is.
            vals = vals.split()
            # Special case approaching.
            if "(do" in vals: vals = vals[:1]
            if len(vals) == 0:
                pcast = str # Assume NULL output
            else:
                v = vals[0]
                # Figure out if it's castable to floating point:
                try:
                    float(v)
                except ValueError:
                    pcast = str
                else:
                    if any("." in v or "e" in v for v in vals):
                        pcast = float
                    elif v == "inf":
                        pcast = str
                    else:
                        pcast = int
            # Now we figure out what to do with it.
            if param.endswith("Units") and not param.startswith("Temperature"):
                dataType = param[:-5]
                # This one better be a float.
                self.conversion_factors[dataType] = float(vals[0])
            if param.startswith("CosmologyOutputRedshift["):
                index = param[param.find("[")+1:param.find("]")]
                redshift_outputs.append({'index':int(index), 'redshift':float(vals[0])})
            elif len(vals) == 0:
                vals = ""
            elif len(vals) == 1:
                vals = pcast(vals[0])
            else:
                vals = np.array([pcast(i) for i in vals if i != "-99999"])
            self.parameters[param] = vals
        self.refine_by = self.parameters["RefineBy"]
        self.dimensionality = self.parameters["TopGridRank"]
        if self.dimensionality > 1:
            self.domain_dimensions = self.parameters["TopGridDimensions"]
            if len(self.domain_dimensions) < 3:
                tmp = self.domain_dimensions.tolist()
                tmp.append(1)
                self.domain_dimensions = np.array(tmp)
            self.domain_left_edge = np.array(self.parameters["DomainLeftEdge"],
                                             "float64").copy()
            self.domain_right_edge = np.array(self.parameters["DomainRightEdge"],
                                             "float64").copy()
        else:
            self.domain_left_edge = np.array(self.parameters["DomainLeftEdge"],
                                             "float64")
            self.domain_right_edge = np.array(self.parameters["DomainRightEdge"],
                                             "float64")
            self.domain_dimensions = np.array([self.parameters["TopGridDimensions"],1,1])

        if self.parameters["ComovingCoordinates"]:
            cosmo_attr = {'box_size': 'CosmologyComovingBoxSize',
                          'omega_lambda': 'CosmologyOmegaLambdaNow',
                          'omega_matter': 'CosmologyOmegaMatterNow',
                          'hubble_constant': 'CosmologyHubbleConstantNow',
                          'initial_redshift': 'CosmologyInitialRedshift',
                          'final_redshift': 'CosmologyFinalRedshift'}
            self.cosmological_simulation = 1
            for a, v in cosmo_attr.items():
                if not v in self.parameters:
                    raise MissingParameter(self.parameter_filename, v)
                setattr(self, a, self.parameters[v])
        else:
            self.omega_lambda = self.omega_matter = \
                self.hubble_constant = self.cosmological_simulation = 0.0

        # make list of redshift outputs
        self.all_redshift_outputs = []
        if not self.cosmological_simulation: return
        for output in redshift_outputs:
            output['filename'] = os.path.join(self.parameters['GlobalDir'],
                                              "%s%04d" % (self.parameters['RedshiftDumpDir'],
                                                          output['index']),
                                              "%s%04d" % (self.parameters['RedshiftDumpName'],
                                                          output['index']))
            del output['index']
        self.all_redshift_outputs = redshift_outputs

    def _calculate_redshift_dump_times(self):
        "Calculates time from redshift of redshift outputs."

        if not self.cosmological_simulation: return
        for output in self.all_redshift_outputs:
            output['time'] = self.enzo_cosmology.ComputeTimeFromRedshift(output['redshift']) / \
                self.enzo_cosmology.TimeUnits
        self.all_redshift_outputs.sort(key=lambda obj:obj['time'])

    def _calculate_time_outputs(self):
        "Calculate time outputs and their redshifts if cosmological."

        if self.final_time is None or \
            not 'dtDataDump' in self.parameters or \
            self.parameters['dtDataDump'] <= 0.0: return []

        self.all_time_outputs = []
        index = 0
        current_time = self.initial_time
        while current_time <= self.final_time + self.parameters['dtDataDump']:
            filename = os.path.join(self.parameters['GlobalDir'],
                                    "%s%04d" % (self.parameters['DataDumpDir'], index),
                                    "%s%04d" % (self.parameters['DataDumpName'], index))

            output = {'index': index, 'filename': filename, 'time': current_time}
            output['time'] = min(output['time'], self.final_time)
            if self.cosmological_simulation:
                output['redshift'] = self.enzo_cosmology.ComputeRedshiftFromTime(
                    current_time * self.enzo_cosmology.TimeUnits)

            self.all_time_outputs.append(output)
            if np.abs(self.final_time - current_time) / self.final_time < 1e-4: break
            current_time += self.parameters['dtDataDump']
            index += 1

    def _calculate_cycle_outputs(self):
        "Calculate cycle outputs."

        mylog.warn('Calculating cycle outputs.  Dataset times will be unavailable.')

        if self.stop_cycle is None or \
            not 'CycleSkipDataDump' in self.parameters or \
            self.parameters['CycleSkipDataDump'] <= 0.0: return []

        self.all_time_outputs = []
        index = 0
        for cycle in range(0, self.stop_cycle+1, self.parameters['CycleSkipDataDump']):
            filename = os.path.join(self.parameters['GlobalDir'],
                                    "%s%04d" % (self.parameters['DataDumpDir'], index),
                                    "%s%04d" % (self.parameters['DataDumpName'], index))

            output = {'index': index, 'filename': filename, 'cycle': cycle}
            self.all_time_outputs.append(output)
            index += 1

    def _get_all_outputs(self, find_outputs=False):
        "Get all potential datasets and combine into a time-sorted list."

        # Create the set of outputs from which further selection will be done.
        if find_outputs:
            self._find_outputs()

        elif self.parameters['dtDataDump'] > 0 and \
          self.parameters['CycleSkipDataDump'] > 0:
            mylog.info("Simulation %s has both dtDataDump and CycleSkipDataDump set.", self.parameter_filename )
            mylog.info("    Unable to calculate datasets.  Attempting to search in the current directory")
            self._find_outputs()

        else:
            # Get all time or cycle outputs.
            if self.parameters['CycleSkipDataDump'] > 0:
                self._calculate_cycle_outputs()
            else:
                self._calculate_time_outputs()

            # Calculate times for redshift outputs.
            self._calculate_redshift_dump_times()

            self.all_outputs = self.all_time_outputs + self.all_redshift_outputs
            if self.parameters['CycleSkipDataDump'] <= 0:
                self.all_outputs.sort(key=lambda obj:obj['time'])

    def _calculate_simulation_bounds(self):
        """
        Figure out the starting and stopping time and redshift for the simulation.
        """

        if 'StopCycle' in self.parameters:
            self.stop_cycle = self.parameters['StopCycle']

        # Convert initial/final redshifts to times.
        if self.cosmological_simulation:
            # Instantiate EnzoCosmology object for units and time conversions.
            self.enzo_cosmology = EnzoCosmology(HubbleConstantNow=
                                                (100.0 * self.parameters['CosmologyHubbleConstantNow']),
                                                OmegaMatterNow=self.parameters['CosmologyOmegaMatterNow'],
                                                OmegaLambdaNow=self.parameters['CosmologyOmegaLambdaNow'],
                                                InitialRedshift=self.parameters['CosmologyInitialRedshift'])
            self.initial_time = self.enzo_cosmology.ComputeTimeFromRedshift(self.initial_redshift) / \
                self.enzo_cosmology.TimeUnits
            self.final_time = self.enzo_cosmology.ComputeTimeFromRedshift(self.final_redshift) / \
                self.enzo_cosmology.TimeUnits

        # If not a cosmology simulation, figure out the stopping criteria.
        else:
            if 'InitialTime' in self.parameters:
                self.initial_time = self.parameters['InitialTime']
            else:
                self.initial_time = 0.

            if 'StopTime' in self.parameters:
                self.final_time = self.parameters['StopTime']
            else:
                self.final_time = None
            if not ('StopTime' in self.parameters or
                    'StopCycle' in self.parameters):
                raise NoStoppingCondition(self.parameter_filename)
            if self.final_time is None:
                mylog.warn('Simulation %s has no stop time set, stopping condition will be based only on cycles.',
                           self.parameter_filename)

    def _set_parameter_defaults(self):
        "Set some default parameters to avoid problems if they are not in the parameter file."

        self.parameters['GlobalDir'] = self.directory
        self.parameters['DataDumpName'] = "data"
        self.parameters['DataDumpDir'] = "DD"
        self.parameters['RedshiftDumpName'] = "RedshiftOutput"
        self.parameters['RedshiftDumpDir'] = "RD"
        self.parameters['ComovingCoordinates'] = 0
        self.parameters['TopGridRank'] = 3
        self.parameters['DomainLeftEdge'] = np.zeros(self.parameters['TopGridRank'])
        self.parameters['DomainRightEdge'] = np.ones(self.parameters['TopGridRank'])
        self.parameters['RefineBy'] = 2 # technically not the enzo default
        self.parameters['StopCycle'] = 100000
        self.parameters['dtDataDump'] = 0.
        self.parameters['CycleSkipDataDump'] = 0.
        self.parameters['TimeUnits'] = 1.

    def _set_time_units(self):
        """
        Set up a dictionary of time units conversions.
        """

        self.time_units = {}
        if self.cosmological_simulation:
            self.parameters['TimeUnits'] = 2.52e17 / np.sqrt(self.omega_matter) \
                / self.hubble_constant / (1 + self.initial_redshift)**1.5
        self.time_units['1'] = 1.
        self.time_units['seconds'] = self.parameters['TimeUnits']
        for unit in sec_conversion.keys():
            self.time_units[unit] = self.parameters['TimeUnits'] / sec_conversion[unit]

    def _find_outputs(self):
        """
        Search for directories matching the data dump keywords.
        If found, get dataset times py opening the pf.
        """

        # look for time outputs.
        potential_time_outputs = \
          glob.glob(os.path.join(self.parameters['GlobalDir'],
                                 "%s*" % self.parameters['DataDumpDir']))
        self.all_time_outputs = \
          self._check_for_outputs(potential_time_outputs)
        self.all_time_outputs.sort(key=lambda obj: obj['time'])

        # look for redshift outputs.
        potential_redshift_outputs = \
          glob.glob(os.path.join(self.parameters['GlobalDir'],
                                 "%s*" % self.parameters['RedshiftDumpDir']))
        self.all_redshift_outputs = \
          self._check_for_outputs(potential_redshift_outputs)
        self.all_redshift_outputs.sort(key=lambda obj: obj['time'])

        self.all_outputs = self.all_time_outputs + self.all_redshift_outputs
        self.all_outputs.sort(key=lambda obj: obj['time'])
        only_on_root(mylog.info, "Located %d total outputs.", len(self.all_outputs))

        # manually set final time and redshift with last output
        if self.all_outputs:
            self.final_time = self.all_outputs[-1]['time']
            if self.cosmological_simulation:
                self.final_redshift = self.all_outputs[-1]['redshift']

    def _check_for_outputs(self, potential_outputs):
        r"""Check a list of files to see if they are valid datasets."""

        only_on_root(mylog.info, "Checking %d potential outputs.", 
                     len(potential_outputs))

        my_outputs = {}
        for my_storage, output in parallel_objects(potential_outputs, 
                                                   storage=my_outputs):
            if self.parameters['DataDumpDir'] in output:
                dir_key = self.parameters['DataDumpDir']
                output_key = self.parameters['DataDumpName']
            else:
                dir_key = self.parameters['RedshiftDumpDir']
                output_key = self.parameters['RedshiftDumpName']
            index = output[output.find(dir_key) + len(dir_key):]
            filename = os.path.join(self.parameters['GlobalDir'],
                                    "%s%s" % (dir_key, index),
                                    "%s%s" % (output_key, index))
            if os.path.exists(filename):
                try:
                    pf = load(filename)
                    if pf is not None:
                        my_storage.result = {'filename': filename,
                                             'time': pf.current_time}
                        if pf.cosmological_simulation:
                            my_storage.result['redshift'] = pf.current_redshift
                except YTOutputNotIdentified:
                    mylog.error('Failed to load %s', filename)
        my_outputs = [my_output for my_output in my_outputs.values() \
                      if my_output is not None]

        return my_outputs

    def _get_outputs_by_key(self, key, values, tolerance=None, outputs=None):
        r"""Get datasets at or near to given values.

        Parameters
        ----------
        key: str
            The key by which to retrieve outputs, usually 'time' or
            'redshift'.
        values: array_like
            A list of values, given as floats.
        tolerance : float
            If not None, do not return a dataset unless the value is
            within the tolerance value.  If None, simply return the
            nearest dataset.
            Default: None.
        outputs : list
            The list of outputs from which to choose.  If None,
            self.all_outputs is used.
            Default: None.

        Examples
        --------
        >>> datasets = es.get_outputs_by_key('redshift', [0, 1, 2], tolerance=0.1)

        """

        values = ensure_list(values)
        if outputs is None:
            outputs = self.all_outputs
        my_outputs = []
        if not outputs:
            return my_outputs
        for value in values:
            outputs.sort(key=lambda obj:np.fabs(value - obj[key]))
            if (tolerance is None or np.abs(value - outputs[0][key]) <= tolerance) \
                    and outputs[0] not in my_outputs:
                my_outputs.append(outputs[0])
            else:
                mylog.error("No dataset added for %s = %f.", key, value)

        outputs.sort(key=lambda obj: obj['time'])
        return my_outputs

    def _get_outputs_by_redshift(self, redshifts, tolerance=None, outputs=None):
        r"""Get datasets at or near to given redshifts.

        Parameters
        ----------
        redshifts: array_like
            A list of redshifts, given as floats.
        tolerance : float
            If not None, do not return a dataset unless the value is
            within the tolerance value.  If None, simply return the
            nearest dataset.
            Default: None.
        outputs : list
            The list of outputs from which to choose.  If None,
            self.all_outputs is used.
            Default: None.

        Examples
        --------
        >>> datasets = es.get_outputs_by_redshift([0, 1, 2], tolerance=0.1)

        """

        return self._get_outputs_by_key('redshift', redshifts, tolerance=tolerance,
                                     outputs=outputs)

    def _get_outputs_by_time(self, times, tolerance=None, outputs=None,
                             time_units='1'):
        r"""Get datasets at or near to given times.

        Parameters
        ----------
        times: array_like
            A list of times, given in code units as floats.
        tolerance : float
            If not None, do not return a dataset unless the time is
            within the tolerance value.  If None, simply return the
            nearest dataset.
            Default = None.
        outputs : list
            The list of outputs from which to choose.  If None,
            self.all_outputs is used.
            Default: None.
        time_units : str
            The units of the list of times.
            Default: '1' (code units).

        Examples
        --------
        >>> datasets = es.get_outputs_by_time([600, 500, 400], tolerance=10.)

        """

        times = np.array(times) / self.time_units[time_units]
        return self._get_outputs_by_key('time', times, tolerance=tolerance,
                                        outputs=outputs)

    def _write_cosmology_outputs(self, filename, outputs, start_index,
                                 decimals=3):
        r"""Write cosmology output parameters for a cosmology splice.
        """

        mylog.info("Writing redshift output list to %s.", filename)
        f = open(filename, 'w')
        for q, output in enumerate(outputs):
            z_string = "%%s[%%d] = %%.%df" % decimals
            f.write(("CosmologyOutputRedshift[%d] = %."
                     + str(decimals) + "f\n") %
                    ((q + start_index), output['redshift']))
        f.close()
