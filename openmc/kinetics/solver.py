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
        self._initial_power = 1.e6

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

    def initialize(self):

        self.states = OrderedDict()

        for time in TIME_POINTS:
            state = openmc.kinetics.State()
            state.mesh = self.mesh
            state.energy_groups = self.energy_groups
            state.fine_groups = self.fine_groups
            state.one_group = self.one_group
            state.num_delayed_groups = self.num_delayed_groups
            state.time = time
            state.initialize_mgxs()
            self.states[time] = state

    def compute_initial_flux(self):

        # Create states and run initial OpenMC on initial state
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
        initial_power = state.get_powers().sum()
        norm_factor = self.initial_power / initial_power
        state.flux *= norm_factor
        state.k_crit = self.k_crit

        # Compute the initial precursor concentration
        state.compute_initial_precursor_concentration()

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

            if residual < 1.e-10 and i > 100:
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
