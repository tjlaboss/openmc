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

    Attributes
    ----------
    mesh : openmc.mesh.Mesh
        Mesh which specifies the dimensions of coarse mesh.

    geometry : openmc.geometry.Geometry
        Geometry which describes the problem being solved.

    settings_file : openmc.settings.SettingsFile
        Settings file describing the general settings for each simulation.

    materials_file : openmc.materials.MaterialsFile
        Materials file containing the materials info for each simulation.

    clock : openmc.kinetics.Clock
        Clock object.

    one_groups : openmc.mgxs.groups.EnergyGroups
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

    num_delayed_groups : int
        The number of delayed neutron precursor groups.

    states : OrderedDict of openmc.kinetics.State
        States of the problem.

    """

    def __init__(self):

        # Initialize Solver class attributes
        self._mesh = None
        self._geometry = None
        self._settings_file = None
        self._materials_file = None
        self._clock = None
        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._k_crit = None
        self._num_delayed_groups = 6
        self._initial_power = 1.
        self._states = OrderedDict()
        self._constant_seed = True
        self._times = []
        self._core_powers = []
        self._mesh_powers = []
        self._core_volume = 1.

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
    def num_delayed_groups(self):
        return self._num_delayed_groups

    @core_volume.setter
    def core_volume(self, core_volume):
        self._core_volume = core_volume

    @core_powers.setter
    def core_powers(self, core_powers):
        self._core_powers = core_powers

    @mesh_powers.setter
    def mesh_powers(self, mesh_powers):
        self._mesh_powers = mesh_powers

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

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        self._num_delayed_groups = num_delayed_groups

    def transfer_states(self, time_from, time_to='all'):

        if time_to == 'all':
            for time in TIME_POINTS:
                if time != time_from:
                    self.states[time] = copy.deepcopy(self.states[time_from])
        else:
            self.states[time_to] = copy.deepcopy(self.states[time_from])

    def run_openmc(self, time):

        # Get the dimensions of the mesh
        dx, dy, dz = self.mesh.width
        dxyz = dx * dy * dz

        # Create a new random seed for the xml file
        if not self.constant_seed:
            self.settings_file.seed = np.random.randint(1, 1e6, 1)[0]

        # Create MGXS
        self.states[time].initialize_mgxs()

        # Create the xml files
        self.geometry.export_to_xml()
        self.materials_file.export_to_xml()
        self.settings_file.export_to_xml()
        self.generate_tallies_file(time)

        # Run OpenMC
        openmc.run(mpi_procs=8)

        # Load MGXS from statepoint
        os.rename('statepoint.{}.h5'.format(self.settings_file.batches),
                  'statepoint_{0}.{1}.h5'.format(time, self.settings_file.batches))
        statepoint_file = openmc.StatePoint(
            'statepoint_{0}.{1}.h5'.format(time, self.settings_file.batches))

        for mgxs in self.states[time].mgxs_lib.values():
            mgxs.load_from_statepoint(statepoint_file)

    def create_state(self, time):

        state = openmc.kinetics.State()
        state.mesh = self.mesh
        state.energy_groups = self.energy_groups
        state.fine_groups = self.fine_groups
        state.one_group = self.one_group
        state.num_delayed_groups = self.num_delayed_groups
        state.time = time
        state.clock = self.clock
        state.k_crit = self.k_crit
        state.core_volume = self.core_volume
        self.states[time] = state

    def compute_initial_flux(self):

        # Create states and run initial OpenMC on initial state
        self.create_state('START')
        self.run_openmc('START')

        # Extract the destruction and production matrices
        state = self.states['START']
        mgxs_lib = state.mgxs_lib
        flux = mgxs_lib['absorption'].tallies['flux'].get_values().flatten()
        state.flux = flux
        A = state.get_destruction_matrix()
        M = state.get_production_matrix()

        # Compute the initial eigenvalue
        flux, self.k_crit = self.compute_eigenvalue(A, M, flux)

        # Normalize the initial flux
        state.flux = flux
        initial_power = state.get_core_power_density()
        norm_factor = self.initial_power / initial_power
        state.flux *= norm_factor
        state.k_crit = self.k_crit

        # Compute the initial precursor concentration
        state.compute_initial_precursor_concentration()

        # Copy data to all other states
        for time in TIME_POINTS:
            if time != 'START':
                self.create_state(time)
                self.copy_states('START', time, True)

    def copy_states(self, time_from, time_to='ALL', copy_mgxs=False):

        state_from = self.states[time_from]

        if time_to == 'ALL':
            for time in TIME_POINTS:
                if time != time_from:
                    state_to = self.states[time]
                    state_to.flux = copy.deepcopy(state_from.flux)
                    precursor_conc = state_from.precursor_conc
                    state_to.precursor_conc = copy.deepcopy(precursor_conc)
                    self.clock.times[time] = self.clock.times[time_from]

                    if copy_mgxs:
                        state_to.mgxs_lib = state_from.mgxs_lib
        else:
            state_from = self.states[time_from]
            state_to = self.states[time_to]
            state_to.flux = copy.deepcopy(state_from.flux)
            state_to.precursor_conc = copy.deepcopy(state_from.precursor_conc)
            self.clock.times[time_to] = self.clock.times[time_from]

            if copy_mgxs:
                state_to.mgxs_lib = state_from.mgxs_lib

    def take_outer_step(self):

        # Increment clock
        clock = self.clock
        times = clock.times
        times['FORWARD_OUT'] += clock.dt_outer

        # Run OpenMC on forward out state
        self.run_openmc('FORWARD_OUT')

        # Get the relevant states
        state_fwd_out    = self.states['FORWARD_OUT']
        state_prev_out   = self.states['PREVIOUS_OUT']
        state_fwd_in     = self.states['FORWARD_IN']
        state_prev_in    = self.states['PREVIOUS_IN']
        state_fwd_in_old = self.states['FORWARD_IN_OLD']

        # Get the forward and backward transient matrices
        trans_matrix_fwd  = state_fwd_out.get_transient_matrix()
        trans_matrix_prev = state_prev_out.get_transient_matrix()

        while (times['FORWARD_IN'] < times['FORWARD_OUT'] - 1.e-8):

            # Increment forward in time
            times['FORWARD_IN'] += clock.dt_inner
            times['FORWARD_IN_OLD'] = times['FORWARD_IN']

            # Get the weight for this time point
            wgt = clock.get_inner_weight()

            # Get the transient matrix
            trans_matrix = wgt * trans_matrix_fwd + (1 - wgt) * trans_matrix_prev

            res = 1.e10
            while (res > 1.e-6):

                # Get the source
                source = state_prev_in.get_source()

                # Solve for the flux at FORWARD_IN
                state_fwd_in.flux = spsolve(trans_matrix, source)

                # Propagate the precursors
                state_fwd_in.propagate_precursors(state_prev_in,
                                                  clock.dt_inner)

                # Compute the residual
                flux_res = (state_fwd_in.flux - state_fwd_in_old.flux) / \
                           state_fwd_in.flux
                flux_res = np.nan_to_num(flux_res)
                res = np.linalg.norm(flux_res)

                # Copy to FORWARD_IN_OLD
                self.copy_states('FORWARD_IN', 'FORWARD_IN_OLD')

                #print('time: {0:1.4f} s, res: {1:1.4e} RMSE'.\
                #      format(times['FORWARD_IN'], res))

            # Update the values for the time step
            self.copy_states('FORWARD_IN', 'PREVIOUS_IN')

            # Save the core power at FORWARD_IN
            self.times.append(times['FORWARD_IN'])
            self.core_powers.append(state_fwd_in.get_core_power_density())
            self.mesh_powers.append(state_fwd_in.get_mesh_powers())
            print('time: {0:1.4f} s, power: {1:1.4e} W/cm^3'.\
                  format(self.times[-1], self.core_powers[-1]))

        # Propagate times
        self.copy_states('FORWARD_IN')
        self.copy_states('FORWARD_OUT', 'PREVIOUS_OUT', True)

    def compute_eigenvalue(self, A, M, flux):

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
        tallies_file.export_to_xml()
