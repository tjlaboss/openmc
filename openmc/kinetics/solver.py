from __future__ import division

from collections import OrderedDict
from numbers import Integral
import warnings
import os
import sys
import copy
import itertools

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

import openmc
import openmc.checkvalue as cv
import openmc.mgxs
import openmc.kinetics
from openmc.kinetics.clock import TIME_POINTS

if sys.version_info[0] >= 3:
    basestring = str


class Solver(object):
    """Solver to propagate the neutron flux and power forward in time.

    Parameters
    ----------
    name : str
        A name for this solve.

    directory : str
        A directory to save the transient simulation data.

    Attributes
    ----------
    name : str
        A name for this solve.

    directory : str
        A directory to save the transient simulation data.

    mesh : openmc.mesh.Mesh
        Mesh which specifies the dimensions of coarse mesh.

    unity_mesh : openmc.mesh.Mesh
        Mesh which contains only one cell.

    pin_cell_mesh : openmc.mesh.Mesh
        Mesh over the pin cells.

    assembly_mesh : openmc.mesh.Mesh
        Mesh over the assemblies.

    geometry : openmc.geometry.Geometry
        Geometry which describes the problem being solved.

    settings_file : openmc.settings.SettingsFile
        Settings file describing the general settings for each simulation.

    materials_file : openmc.materials.MaterialsFile
        Materials file containing the materials info for each simulation.

    clock : openmc.kinetics.Clock
        Clock object.

    one_group : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the a one-energy-group structure.

    energy_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the energy groups structure.

    fine_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups used to tally the transport cross section that will be
        condensed to get the diffusion coefficients in the coarse group
        structure.

    initial_power : float
        The initial core power (in MWth).

    k_crit : float
        The initial eigenvalue.

    mpi_procs : int
        The number of MPI processes to use.

    chi_delayed_by_delayed_group : bool
        Whether to use delayed groups in representing chi-delayed.

    chi_delayed_by_mesh : bool
        Whether to use a mesh in representing chi-delayed.

    num_delayed_groups : int
        The number of delayed neutron precursor groups.

    states : OrderedDict of openmc.kinetics.State
        States of the problem.

    use_pregenerated_sps : bool
        Whether to use pregenerated statepoint files.

    constant_seed : bool
        Whether to use a constant seed in the OpenMC solve.

    times : np.array
        The times at which the powers are saved.

    core_powers : np.array
        The core powers at different time points during the solve.

    pin_powers : np.array
        The pin powers at different time points during the solve.

    assembly_powers : np.array
        The assembly powers at different time points during the solve.

    core_volume : float
        The core volume used to normalize the initial power.

    """

    def __init__(self, name='kinetics_solve', directory='.'):

        # Initialize Solver class attributes
        self.name = name
        self.directory = directory
        self._mesh = None
        self._unity_mesh = None
        self._pin_cell_mesh = None
        self._assembly_mesh = None
        self._geometry = None
        self._settings_file = None
        self._materials_file = None
        self._clock = None
        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._k_crit = 1.0
        self._num_delayed_groups = 6
        self._initial_power = 1.
        self._states = OrderedDict()
        self._constant_seed = True
        self._times = []
        self._core_powers = []
        self._mesh_powers = []
        self._pin_powers = []
        self._assembly_powers = []
        self._core_volume = 1.
        self._chi_delayed_by_delayed_group = False
        self._chi_delayed_by_mesh = False
        self._mpi_procs = 4
        self._use_pregenerated_sps = False

    @property
    def name(self):
        return self._name

    @property
    def directory(self):
        return self._directory

    @property
    def use_pregenerated_sps(self):
        return self._use_pregenerated_sps

    @property
    def core_volume(self):
        return self._core_volume

    @property
    def core_powers(self):
        return self._core_powers

    @property
    def mesh_powers(self):
        return self._mesh_powers

    @property
    def pin_powers(self):
        return self._pin_powers

    @property
    def assembly_powers(self):
        return self._assembly_powers

    @property
    def times(self):
        return self._times

    @property
    def constant_seed(self):
        return self._constant_seed

    @property
    def states(self):
        return self._states

    @property
    def mesh(self):
        return self._mesh

    @property
    def unity_mesh(self):
        return self._unity_mesh

    @property
    def pin_cell_mesh(self):
        return self._pin_cell_mesh

    @property
    def assembly_mesh(self):
        return self._assembly_mesh

    @property
    def nxyz(self):
        return np.prod(self.mesh.dimension)

    @property
    def ng(self):
        return self.energy_groups.num_groups

    @property
    def geometry(self):
        return self._geometry

    @property
    def settings_file(self):
        return self._settings_file

    @property
    def materials_file(self):
        return self._materials_file

    @property
    def clock(self):
        return self._clock

    @property
    def one_group(self):
        return self._one_group

    @property
    def energy_groups(self):
        return self._energy_groups

    @property
    def fine_groups(self):
        return self._fine_groups

    @property
    def initial_power(self):
        return self._initial_power

    @property
    def k_crit(self):
        return self._k_crit

    @property
    def chi_delayed_by_delayed_group(self):
        return self._chi_delayed_by_delayed_group

    @property
    def chi_delayed_by_mesh(self):
        return self._chi_delayed_by_mesh

    @property
    def num_delayed_groups(self):
        return self._num_delayed_groups

    @property
    def mpi_procs(self):
        return self._mpi_procs

    @property
    def run_directory(self):
        return self.directory + '/' + self.name

    @name.setter
    def name(self, name):
        self._name = name

    @directory.setter
    def directory(self, directory):
        self._directory = directory

    @use_pregenerated_sps.setter
    def use_pregenerated_sps(self, use_pregenerated_sps):
        self._use_pregenerated_sps = use_pregenerated_sps

    @core_volume.setter
    def core_volume(self, core_volume):
        self._core_volume = core_volume

    @core_powers.setter
    def core_powers(self, core_powers):
        self._core_powers = core_powers

    @mesh_powers.setter
    def mesh_powers(self, mesh_powers):
        self._mesh_powers = mesh_powers

    @pin_powers.setter
    def pin_powers(self, pin_powers):
        self._pin_powers = pin_powers

    @assembly_powers.setter
    def assembly_powers(self, assembly_powers):
        self._assembly_powers = assembly_powers

    @times.setter
    def times(self, times):
        self._times = times

    @constant_seed.setter
    def constant_seed(self, constant_seed):
        self._constant_seed = constant_seed

    @states.setter
    def states(self, states):
        self._states = states

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

        self._unity_mesh = openmc.Mesh()
        self._unity_mesh.type = mesh.type
        self._unity_mesh.dimension = [1,1,1]
        self._unity_mesh.lower_left  = mesh.lower_left
        self._unity_mesh.width = [i*j for i,j in zip(mesh.dimension, mesh.width)]

    @pin_cell_mesh.setter
    def pin_cell_mesh(self, pin_cell_mesh):
        self._pin_cell_mesh = pin_cell_mesh

    @assembly_mesh.setter
    def assembly_mesh(self, assembly_mesh):
        self._assembly_mesh = assembly_mesh

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry

    @settings_file.setter
    def settings_file(self, settings_file):
        self._settings_file = settings_file

    @materials_file.setter
    def materials_file(self, materials_file):
        self._materials_file = materials_file

    @clock.setter
    def clock(self, clock):
        self._clock = clock

    @one_group.setter
    def one_group(self, one_group):
        self._one_group = one_group

    @energy_groups.setter
    def energy_groups(self, energy_groups):
        self._energy_groups = energy_groups

    @fine_groups.setter
    def fine_groups(self, fine_groups):
        self._fine_groups = fine_groups

    @initial_power.setter
    def initial_power(self, initial_power):
        self._initial_power = initial_power

    @k_crit.setter
    def k_crit(self, k_crit):
        self._k_crit = k_crit

    @chi_delayed_by_delayed_group.setter
    def chi_delayed_by_delayed_group(self, chi_delayed_by_delayed_group):
        self._chi_delayed_by_delayed_group = chi_delayed_by_delayed_group

    @chi_delayed_by_mesh.setter
    def chi_delayed_by_mesh(self, chi_delayed_by_mesh):
        self._chi_delayed_by_mesh = chi_delayed_by_mesh

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        self._num_delayed_groups = num_delayed_groups

    @mpi_procs.setter
    def mpi_procs(self, mpi_procs):
        self._mpi_procs = mpi_procs

    def transfer_states(self, time_from, time_to='all'):

        if time_to == 'all':
            for time in TIME_POINTS:
                if time != time_from:
                    self.states[time] = copy.deepcopy(self.states[time_from])
                    self.states[time].time = time
        else:
            self.states[time_to] = copy.deepcopy(self.states[time_from])
            self.states[time_to].time = time

    def run_openmc(self, time):

        # Create a new random seed for the xml file
        if not self.constant_seed:
            self.settings_file.seed = np.random.randint(1, 1e6, 1)[0]

        # Create MGXS
        self.states[time].initialize_mgxs()

        # Create the xml files
        self.geometry.time = self.clock.times[time]
        self.geometry.export_to_xml(self.run_directory + '/geometry.xml')
        self.materials_file.export_to_xml(self.run_directory + '/materials.xml')
        self.settings_file.export_to_xml(self.run_directory + '/settings.xml')
        self.generate_tallies_file(time)

        # Names of the statepoint and summary files
        sp_old_name = '{}/statepoint.{}.h5'.format(self.run_directory,
                                                   self.settings_file.batches)
        sp_new_name = '{}/statepoint_{:.6f}_sec.{}.h5'\
                      .format(self.run_directory, self.clock.times[time],
                              self.settings_file.batches)
        sum_old_name = '{}/summary.h5'.format(self.run_directory)
        sum_new_name = '{}/summary_{:.6f}_sec.h5'.format(self.run_directory,
                                                         self.clock.times[time])

        # Run OpenMC
        if not self.use_pregenerated_sps:
            openmc.run(mpi_procs=self.mpi_procs, cwd=self.run_directory)

            # Rename the statepoint and summary files
            os.rename(sp_old_name, sp_new_name)
            os.rename(sum_old_name, sum_new_name)

        # Load the summary and statepoint files
        summary_file = openmc.Summary(sum_new_name)
        statepoint_file = openmc.StatePoint(sp_new_name, False)
        statepoint_file.link_with_summary(summary_file)

        for mgxs in self.states[time].mgxs_lib.values():
            mgxs.load_from_statepoint(statepoint_file)

    def create_state(self, time, derived=False):

        if derived:
            state = openmc.kinetics.DerivedState(self.states)
        else:
            state = openmc.kinetics.State()
        state.mesh = self.mesh
        state.unity_mesh = self.unity_mesh
        state.pin_cell_mesh = self.pin_cell_mesh
        state.assembly_mesh = self.assembly_mesh
        state.energy_groups = self.energy_groups
        state.fine_groups = self.fine_groups
        state.one_group = self.one_group
        state.num_delayed_groups = self.num_delayed_groups
        state.time = time
        state.clock = self.clock
        state.k_crit = self.k_crit
        state.core_volume = self.core_volume
        state.chi_delayed_by_delayed_group = self.chi_delayed_by_delayed_group
        state.chi_delayed_by_mesh = self.chi_delayed_by_mesh
        self.states[time] = state

    def compute_initial_flux(self):

        # Create the test directory if it doesn't exist
        if not os.path.exists(self.run_directory):
            os.makedirs(self.run_directory)

        # Create states and run initial OpenMC on initial state
        self.create_state('START')
        self.run_openmc('START')

        # Extract the flux from the first solve
        state = self.states['START']
        mgxs_lib = state.mgxs_lib
        flux = mgxs_lib['absorption'].tallies['flux'].get_values()
        flux.shape = (self.nxyz, self.ng)
        flux = flux[:, ::-1]
        state.flux = flux

        # Compute the initial eigenvalue
        flux, self.k_crit = self.compute_eigenvalue(state.destruction_matrix,
                                                    state.production_matrix,
                                                    flux)

        # Compute the initial adjoint eigenvalue
        adjoint_flux = np.ones(self.nxyz * self.ng)
        adjoint_flux, k_adjoint = self.compute_eigenvalue\
                                  (state.adjoint_destruction_matrix,
                                   state.production_matrix.transpose(),
                                   adjoint_flux)

        # Normalize the initial flux
        state.flux = flux
        state.adjoint_flux = adjoint_flux
        norm_factor = self.initial_power / state.core_power_density
        state.flux *= norm_factor
        state.adjoint_flux *= norm_factor
        state.k_crit = self.k_crit

        # Compute the initial precursor concentration
        state.compute_initial_precursor_concentration()

        # Create the arrays with the data
        self.times.append(self.clock.times['START'])
        self.core_powers.append(state.core_power_density)
        self.mesh_powers.append(state.mesh_powers)
        self.pin_powers.append(state.pin_powers)
        self.assembly_powers.append(state.assembly_powers)

        # Copy data to all other states
        for time in TIME_POINTS:
            if time != 'START':
                if time in ['PREVIOUS_IN', 'FORWARD_IN']:
                    self.create_state(time, True)
                    self.copy_states('START', time)
                else:
                    self.create_state(time, False)
                    self.copy_states('START', time, True)

    def copy_states(self, time_from, time_to='ALL', copy_mgxs=False):

        state_from = self.states[time_from]

        if time_to == 'ALL':
            for time in TIME_POINTS:
                if time != time_from:
                    state_to = self.states[time]
                    state_to.flux = copy.deepcopy(state_from.flux)
                    state_to.adjoint_flux = copy.deepcopy(state_from.adjoint_flux)
                    state_to.precursors = copy.deepcopy(state_from.precursors)
                    self.clock.times[time] = self.clock.times[time_from]

                    if copy_mgxs:
                        state_to.mgxs_lib = state_from.mgxs_lib
        else:
            state_from = self.states[time_from]
            state_to = self.states[time_to]
            state_to.flux = copy.deepcopy(state_from.flux)
            state_to.adjoint_flux = copy.deepcopy(state_from.adjoint_flux)
            state_to.precursors = copy.deepcopy(state_from.precursors)
            self.clock.times[time_to] = self.clock.times[time_from]

            if copy_mgxs:
                state_to.mgxs_lib = state_from.mgxs_lib

    def take_outer_step(self):

        # Increment clock
        clock = self.clock
        times = clock.times
        times['FORWARD_OUT'] += clock.dt_outer
        state_pre = self.states['PREVIOUS_IN']
        state_fwd = self.states['FORWARD_IN']

        # Run OpenMC on forward out state
        self.run_openmc('FORWARD_OUT')

        while (times['FORWARD_IN'] < times['FORWARD_OUT'] - 1.e-8):

            # Increment forward in time
            times['FORWARD_IN'] += clock.dt_inner

            # Get the transient matrix and time source
            time_source = state_fwd.time_source_matrix * \
                          state_pre.flux.flatten()
            source = time_source - state_fwd.decay_source(state_pre).flatten()

            # Solve for the flux at FORWARD_IN
            state_fwd.flux = spsolve(state_fwd.transient_matrix, source)

            # Propagate the precursors
            state_fwd.propagate_precursors(state_pre)

            # Update the values for the time step
            self.copy_states('FORWARD_IN', 'PREVIOUS_IN')

            # Save the core power at FORWARD_IN
            self.times.append(times['FORWARD_IN'])
            self.core_powers.append(state_fwd.core_power_density)
            self.mesh_powers.append(state_fwd.mesh_powers)
            self.pin_powers.append(state_fwd.pin_powers)
            self.assembly_powers.append(state_fwd.assembly_powers)
            print('t: {0:1.3f} s, P: {1:1.3e} W/cm^3, rho: {2:+1.3f} pcm, beta_eff: {3:1.5f}, pnl: {4:1.3e} s'.\
                  format(self.times[-1], self.core_powers[-1], state_fwd.reactivity * 1.e5,
                         state_fwd.beta_eff, state_fwd.pnl))

        # Copy the flux, precursors, and time from FORWARD_IN to FORWARD_OUT
        self.copy_states('FORWARD_IN', 'FORWARD_OUT')

        # Copy the flux, precursors, time, and MGXS from FORWARD_OUT to PREVIOUS_OUT
        self.copy_states('FORWARD_OUT', 'PREVIOUS_OUT', True)

    def compute_eigenvalue(self, A, M, flux):

        # Ensure flux is a 1D array
        flux = flux.flatten()

        # Compute the initial source
        old_source = M * flux
        norm = old_source.mean()
        old_source /= norm
        flux /= norm

        for i in range(10000):

            # Solve linear system
            flux = spsolve(A, old_source)

            # Compute new source
            new_source = M * flux

            # Compute and set k-eff
            k_eff = new_source.mean()

            # Scale the new source by 1 / k-eff
            new_source /= k_eff

            # Compute the residual
            residual_array = (new_source - old_source) / new_source
            residual_array = np.nan_to_num(residual_array)
            residual_array = np.square(residual_array)
            residual = np.sqrt(residual_array.mean())

            # Copy new source to old source
            old_source = np.copy(new_source)

            print('linear solver iter {0} resid {1:1.5e} k-eff {2:1.6f}'\
                  .format(i, residual, k_eff))

            if residual < 1.e-8 and i > 10:
                break

        return flux, k_eff

    def generate_tallies_file(self, time):

        # Generate a new tallies file
        tallies_file = openmc.Tallies()

        # Get the MGXS library
        mgxs_lib = self.states[time].mgxs_lib

        # Add the tallies to the file
        for mgxs in mgxs_lib.values():
            tallies_file += mgxs.tallies.values()

        # Export the tallies file to xml
        tallies_file.export_to_xml(self.run_directory + '/tallies.xml')
