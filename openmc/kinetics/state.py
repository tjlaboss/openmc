from collections import OrderedDict
from numbers import Integral
import warnings
import copy
import itertools
import sys

import numpy as np
np.set_printoptions(precision=6)
import scipy.sparse as sps
from scipy.sparse import block_diag

import openmc
import openmc.checkvalue as cv
import openmc.mgxs
import openmc.kinetics
from openmc.kinetics.clock import TIME_POINTS
import h5py
from abc import ABCMeta
from six import add_metaclass, string_types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.version_info[0] >= 3:
    basestring = str


@add_metaclass(ABCMeta)
class State(object):
    """State to store all the variables that describe a specific state of the system.

    Attributes
    ----------
    flux_mesh : openmc.mesh.Mesh
        Mesh by which shape is computed on.

    pin_mesh : openmc.mesh.Mesh
        Mesh by which power is reconstructed on.

    unity_mesh : openmc.mesh.Mesh
        Mesh with one cell convering the entire geometry..

    one_group : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the a one-energy-group structure.

    energy_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the energy groups structure.

    fine_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups used to tally the transport cross section that will be
        condensed to get the diffusion coefficients in the coarse group
        structure.

    shape : np.ndarray
        Numpy array used to store the shape function.

    amplitude : np.ndarray
        Numpy array used to store the amplitude.

    adjoint_flux : np.ndarray
        Numpy array used to store the adjoint flux.

    precursors : np.ndarray
        Numpy array used to store the precursor concentrations.

    mgxs_lib : OrderedDict of OrderedDict of openmc.tallies
        Dict of Dict of tallies. The first Dict is indexed by time point
        and the second Dict is indexed by rxn type.

    k_crit : float
        The initial eigenvalue.

    chi_delayed_by_delayed_group : bool
        Whether to use delayed groups in representing chi-delayed.

    chi_delayed_by_mesh : bool
        Whether to use a mesh in representing chi-delayed.

    num_delayed_groups : int
        The number of delayed neutron precursor groups.

    time_point : str
        The time point of this state.

    clock : openmc.kinetics.Clock
        A clock object to indicate the current simulation time.

    core_volume : float
        The core volume used to normalize the initial power.

    log_file : str
        Log file name (including directory prefix).

    multi_group : bool
        Whether the OpenMC run is multi-group or continuous-energy.

    """

    def __init__(self, states):

        # Initialize Solver class attributes
        self._flux_mesh = None
        self._pin_mesh = None
        self._unity_mesh = None

        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._num_delayed_groups = 6

        self._flux = None
        self._precursors = None
        self._adjoint_flux = None

        self._k_crit = 1.0

        self._time_point = None
        self._clock = None
        self._core_volume = 1.

        self._log_file = None
        self._multi_group = True
        self.states = states

    @property
    def states(self):
        return self._states

    @property
    def flux_mesh(self):
        return self._flux_mesh

    @property
    def pin_mesh(self):
        return self._pin_mesh

    @property
    def unity_mesh(self):
        return self._unity_mesh

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
    def flux(self):
        self._flux.shape = (self.flux_nxyz, self.ng)
        return self._flux

    @property
    def adjoint_flux(self):
        self._adjoint_flux.shape = (self.flux_nxyz, self.ng)
        return self._adjoint_flux

    @property
    def precursors(self):
        self._precursors.shape = (self.flux_nxyz, self.nd)
        return self._precursors

    @property
    def k_crit(self):
        return self._k_crit

    @property
    def num_delayed_groups(self):
        return self._num_delayed_groups

    @property
    def time_point(self):
        return self._time_point

    @property
    def clock(self):
        return self._clock

    @property
    def core_volume(self):
        return self._core_volume

    @property
    def log_file(self):
        return self._log_file

    @property
    def multi_group(self):
        return self._multi_group

    @states.setter
    def states(self, states):
        self._states = states

    @flux_mesh.setter
    def flux_mesh(self, mesh):
        self._flux_mesh = mesh

    @pin_mesh.setter
    def pin_mesh(self, mesh):
        self._pin_mesh = mesh

    @unity_mesh.setter
    def unity_mesh(self, mesh):
        self._unity_mesh = mesh

    @one_group.setter
    def one_group(self, one_group):
        self._one_group = one_group

    @energy_groups.setter
    def energy_groups(self, energy_groups):
        self._energy_groups = energy_groups

    @fine_groups.setter
    def fine_groups(self, fine_groups):
        self._fine_groups = fine_groups

    @flux.setter
    def flux(self, flux):
        self._flux = copy.deepcopy(flux)
        self._flux.shape = (self.flux_nxyz, self.ng)

    @adjoint_flux.setter
    def adjoint_flux(self, adjoint_flux):
        self._adjoint_flux = copy.deepcopy(adjoint_flux)
        self._adjoint_flux.shape = (self.flux_nxyz, self.ng)

    @precursors.setter
    def precursors(self, precursors):
        self._precursors = copy.deepcopy(precursors)
        self._precursors.shape = (self.flux_nxyz, self.nd)

    @k_crit.setter
    def k_crit(self, k_crit):
        self._k_crit = k_crit

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        self._num_delayed_groups = num_delayed_groups

    @time_point.setter
    def time_point(self, time_point):
        self._time_point = time_point

    @clock.setter
    def clock(self, clock):
        self._clock = clock

    @core_volume.setter
    def core_volume(self, core_volume):
        self._core_volume = core_volume

    @log_file.setter
    def log_file(self, log_file):
        self._log_file = log_file

    @multi_group.setter
    def multi_group(self, multi_group):
        self._multi_group = multi_group

    @property
    def flux_dimension(self):
        return tuple(self.flux_mesh.dimension[::-1])

    @property
    def pin_dimension(self):
        return tuple(self.pin_mesh.dimension[::-1])

    @property
    def flux_zyxg(self):
        return self.flux_dimension + (self.ng,)

    @property
    def pin_zyxg(self):
        return self.pin_dimension + (self.ng,)

    @property
    def ng(self):
        return self.energy_groups.num_groups

    @property
    def nd(self):
        return self.num_delayed_groups

    @property
    def dt_inner(self):
        return self.clock.dt_inner

    @property
    def dt_outer(self):
        return self.clock.dt_outer

    @property
    def flux_nxyz(self):
        return np.prod(self.flux_dimension)

    @property
    def pin_nxyz(self):
        return np.prod(self.pin_dimension)

    @property
    def flux_dxyz(self):
        return np.prod(self.flux_mesh.width)

    @property
    def pin_dxyz(self):
        return np.prod(self.pin_mesh.width)

    @property
    def power(self):
        return (self.flux_dxyz * self.kappa_fission * self.flux / self.k_crit).sum(axis=1)

    @property
    def core_power_density(self):
        mesh_volume = self.flux_dxyz * self.flux_nxyz
        return self.power.sum() * mesh_volume / self.core_volume

    @property
    def pin_flux(self):
        flux = self.flux
        flux.shape = self.flux_zyxg
        flux = openmc.kinetics.map_array(flux, self.pin_zyxg, normalize=True)
        return flux.reshape((self.pin_nxyz, self.ng)) * self.pin_shape

    @property
    def pin_power(self):
        power = (self.pin_dxyz * self.pin_kappa_fission * self.pin_flux / self.k_crit).sum(axis=1)
        pin_core_power = power.sum() * (self.pin_dxyz * self.pin_nxyz) / self.core_volume
        return power * self.core_power_density / pin_core_power

    @property
    def delayed_production(self):
        chi_delayed = np.repeat(self.chi_delayed, self.ng)
        chi_delayed.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)
        delayed_nu_fission = np.tile(self.delayed_nu_fission, self.ng)
        delayed_nu_fission.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)
        return (chi_delayed * delayed_nu_fission)

    @property
    def prompt_production(self):
        chi_prompt = np.repeat(self.chi_prompt, self.ng)
        chi_prompt.shape = (self.flux_nxyz, self.ng, self.ng)
        prompt_nu_fission = np.tile(self.prompt_nu_fission, self.ng)
        prompt_nu_fission.shape = (self.flux_nxyz, self.ng, self.ng)
        return (chi_prompt * prompt_nu_fission)

    @property
    def delayed_production_matrix(self):
        delayed_production = self.delayed_production.sum(axis=1) * self.flux_dxyz / self.k_crit
        return openmc.kinetics.block_diag(delayed_production)

    @property
    def production_matrix(self):
        return self.prompt_production_matrix + self.delayed_production_matrix

    @property
    def prompt_production_matrix(self):
        return openmc.kinetics.block_diag(self.prompt_production * self.flux_dxyz / self.k_crit)

    @property
    def destruction_matrix(self):

        linear, non_linear = self.coupling_matrix
        inscatter       = self.inscatter * self.flux_dxyz
        absorb_outscat  = self.outscatter + self.absorption
        absorb_outscat  = absorb_outscat * self.flux_dxyz
        inscatter.shape = (self.flux_nxyz, self.ng, self.ng)
        total = sps.diags([absorb_outscat.flatten()], [0]) - openmc.kinetics.block_diag(inscatter)

        return total + linear + non_linear

    @property
    def decay_source(self):
        decay_source = self.decay_rate * self.precursors
        decay_source = np.repeat(decay_source, self.ng)
        decay_source.shape = (self.flux_nxyz, self.nd, self.ng)
        decay_source *= self.chi_delayed
        return decay_source.sum(axis=1).flatten()

    @property
    def coupling_matrix(self):

        diags, dc_linear_data, dc_nonlinear_data = self.coupling_terms

        # Form a matrix of the surface diffusion coefficients corrections
        dc_linear_matrix    = sps.diags(dc_linear_data   , diags)
        dc_nonlinear_matrix = sps.diags(dc_nonlinear_data, diags)

        return dc_linear_matrix, dc_nonlinear_matrix


class OuterState(State):

    def __init__(self, states):
        super(OuterState, self).__init__(states)

        # Initialize Solver class attributes
        self._mgxs_lib = None
        self._chi_delayed_by_delayed_group = False
        self._chi_delayed_by_mesh = False
        self._method = 'ADIABATIC'
        self._mgxs_loaded = False

    @property
    def mgxs_lib(self):
        return self._mgxs_lib

    @property
    def chi_delayed_by_delayed_group(self):
        return self._chi_delayed_by_delayed_group

    @property
    def chi_delayed_by_mesh(self):
        return self._chi_delayed_by_mesh

    @property
    def method(self):
        return self._method

    @property
    def mgxs_loaded(self):
        return self._mgxs_loaded

    @mgxs_lib.setter
    def mgxs_lib(self, mgxs_lib):
        self._mgxs_lib = mgxs_lib

    @chi_delayed_by_delayed_group.setter
    def chi_delayed_by_delayed_group(self, chi_delayed_by_delayed_group):
        self._chi_delayed_by_delayed_group = chi_delayed_by_delayed_group

    @chi_delayed_by_mesh.setter
    def chi_delayed_by_mesh(self, chi_delayed_by_mesh):
        self._chi_delayed_by_mesh = chi_delayed_by_mesh

    @method.setter
    def method(self, method):
        self._method = method

    @mgxs_loaded.setter
    def mgxs_loaded(self, mgxs_loaded):
        self._mgxs_loaded = mgxs_loaded

    @property
    def inscatter(self):
        if not self.mgxs_loaded:
            if self.multi_group:
                self._inscatter = self.mgxs_lib['nu-scatter matrix'].get_xs(row_column='outin')
            else:
                self._inscatter = self.mgxs_lib['consistent nu-scatter matrix'].get_xs(row_column='outin')

        self._inscatter.shape = (self.flux_nxyz, self.ng, self.ng)
        return self._inscatter

    @property
    def outscatter(self):
        return self.inscatter.sum(axis=1)

    @property
    def absorption(self):
        if not self.mgxs_loaded:
            self._absorption = self.mgxs_lib['absorption'].get_xs()

        self._absorption.shape = (self.flux_nxyz, self.ng)
        return self._absorption

    @property
    def kappa_fission(self):
        if not self.mgxs_loaded:
            self._kappa_fission = self.mgxs_lib['kappa-fission'].get_xs()

        self._kappa_fission.shape = (self.flux_nxyz, self.ng)
        return self._kappa_fission

    @property
    def pin_kappa_fission(self):
        if not self.mgxs_loaded:
            self._pin_kappa_fission = self.mgxs_lib['pin-kappa-fission'].get_xs()

        self._pin_kappa_fission.shape = (self.pin_nxyz, self.ng)
        return self._pin_kappa_fission

    @property
    def chi_prompt(self):
        if not self.mgxs_loaded:
            self._chi_prompt = self.mgxs_lib['chi-prompt'].get_xs()

        self._chi_prompt.shape = (self.flux_nxyz, self.ng)
        return self._chi_prompt

    @property
    def prompt_nu_fission(self):
        if not self.mgxs_loaded:
            self._prompt_nu_fission = self.mgxs_lib['prompt-nu-fission'].get_xs()

        self._prompt_nu_fission.shape = (self.flux_nxyz, self.ng)
        return self._prompt_nu_fission

    @property
    def chi_delayed(self):

        if not self.mgxs_loaded:
            self._chi_delayed = self.mgxs_lib['chi-delayed'].get_xs()

            if self.chi_delayed_by_mesh:
                if not self.chi_delayed_by_delayed_group:
                    self._chi_delayed.shape = (self.flux_nxyz, self.ng)
                    self._chi_delayed = np.tile(self._chi_delayed, self.nd)
            else:
                if self.chi_delayed_by_delayed_group:
                    self._chi_delayed = np.tile(self._chi_delayed.flatten(), self.flux_nxyz)
                else:
                    self._chi_delayed = np.tile(self._chi_delayed.flatten(), self.flux_nxyz)
                    self._chi_delayed.shape = (self.flux_nxyz, self.ng)
                    self._chi_delayed = np.tile(self._chi_delayed, self.nd)

        self._chi_delayed.shape = (self.flux_nxyz, self.nd, self.ng)
        return self._chi_delayed

    @property
    def delayed_nu_fission(self):
        if not self.mgxs_loaded:
            self._delayed_nu_fission = self.mgxs_lib['delayed-nu-fission'].get_xs()

        self._delayed_nu_fission.shape = (self.flux_nxyz, self.nd, self.ng)
        return self._delayed_nu_fission

    @property
    def inverse_velocity(self):
        if not self.mgxs_loaded:
            self._inverse_velocity = self.mgxs_lib['inverse-velocity'].get_xs()

        self._inverse_velocity.shape = (self.flux_nxyz, self.ng)
        return self._inverse_velocity

    @property
    def decay_rate(self):
        if not self.mgxs_loaded:
            self._decay_rate = self.mgxs_lib['decay-rate'].get_xs()
            self._decay_rate[self._decay_rate < 1.e-5] = 0.

        self._decay_rate.shape = (self.flux_nxyz, self.nd)
        return self._decay_rate

    @property
    def flux_tallied(self):
        if not self.mgxs_loaded:
            self._flux_tallied = self.mgxs_lib['kappa-fission'].tallies['flux'].get_values()
            self._flux_tallied.shape = (self.flux_nxyz, self.ng)
            self._flux_tallied = self._flux_tallied[:, ::-1]

        self._flux_tallied.shape = (self.flux_nxyz, self.ng)
        return self._flux_tallied

    @property
    def pin_flux_tallied(self):
        if not self.mgxs_loaded:
            self._pin_flux_tallied = self.mgxs_lib['pin-kappa-fission'].tallies['flux'].get_values()
            self._pin_flux_tallied.shape = (self.pin_nxyz, self.ng)
            self._pin_flux_tallied = self._pin_flux_tallied[:, ::-1]

        self._pin_flux_tallied.shape = (self.pin_nxyz, self.ng)
        return self._pin_flux_tallied

    @property
    def current_tallied(self):
        if not self.mgxs_loaded:
            self._current_tallied = self.mgxs_lib['current'].get_xs()

        return self._current_tallied

    @property
    def diffusion_coefficient(self):
        if not self.mgxs_loaded:
            self._diffusion_coefficient = self.mgxs_lib['diffusion-coefficient']
            self._diffusion_coefficient = self._diffusion_coefficient.get_condensed_xs(self.energy_groups).get_xs()

        self._diffusion_coefficient.shape = (self.flux_nxyz, self.ng)
        return self._diffusion_coefficient

    @property
    def flux_frequency(self):

        state_pre = self.states['PREVIOUS_OUTER']

        freq = (1. / self.dt_outer - state_pre.flux / self.flux / self.dt_outer)
        freq = openmc.kinetics.nan_inf_to_zero(freq)

        freq.shape = self.flux_dimension + (self.ng,)
        coarse_shape = (1,1,1,self.ng)
        freq = openmc.kinetics.map_array(freq, coarse_shape, normalize=True)
        freq.shape = (1, self.ng)
        return freq

    @property
    def precursor_frequency(self):

        flux = np.tile(self.flux, self.nd)
        flux.shape = (self.flux_nxyz, self.nd, self.ng)
        del_fis_rate = self.delayed_nu_fission * flux
        freq = del_fis_rate.sum(axis=2) / self.precursors / self.k_crit * self.flux_dxyz - self.decay_rate

        freq = self.decay_rate / (freq + self.decay_rate)
        freq = openmc.kinetics.nan_inf_to_zero(freq)

        return freq

    @property
    def pin_shape(self):

        # Normalize the power mesh flux to the shape mesh
        pm_flux = self.pin_flux_tallied
        pm_flux.shape = self.pin_zyxg
        sm_amp = openmc.kinetics.map_array(pm_flux, self.flux_zyxg, normalize=True)
        pm_amp = openmc.kinetics.map_array(sm_amp, self.pin_zyxg, normalize=True)

        pm_shape = pm_flux / pm_amp
        pm_shape = openmc.kinetics.nan_inf_to_zero(pm_shape)
        pm_shape = (self.pin_nxyz, self.ng)

        return pm_shape

    @property
    def dump_to_log_file(self):

        time_point = str(self.clock.times[self.time_point])
        f = h5py.File(self._log_file, 'a')
        if time_point not in f['OUTER_STEPS'].keys():
            f['OUTER_STEPS'].require_group(time_point)

        if 'pin_shape' not in f['OUTER_STEPS'][time_point].keys():
            f['OUTER_STEPS'][time_point].create_dataset('pin_shape', data=self.pin_shape)
            f['OUTER_STEPS'][time_point].create_dataset('pin_kappa_fission', data=self.pin_kappa_fission)
        else:
            pin_shape = f['OUTER_STEPS'][time_point]['pin_shape']
            pin_shape[...] = self.pin_shape
            pin_kappa_fission = f['OUTER_STEPS'][time_point]['pin_kappa_fission']
            pin_kappa_fission[...] = self.pin_kappa_fission

        f.close()

    def compute_initial_precursor_concentration(self):
        flux = np.tile(self.flux, self.nd).flatten()
        del_fis_rate = self.delayed_nu_fission.flatten() * flux
        del_fis_rate.shape = (self.flux_nxyz, self.nd, self.ng)
        precursors = del_fis_rate.sum(axis=2) / self.decay_rate / self.k_crit * self.flux_dxyz
        self.precursors = openmc.kinetics.nan_inf_to_zero(precursors)

    def load_mgxs(self):
        self.mgxs_loaded = False
        self.inscatter
        self.absorption
        self.chi_prompt
        self.prompt_nu_fission
        self.chi_delayed
        self.delayed_nu_fission
        self.kappa_fission
        self.pin_kappa_fission
        self.inverse_velocity
        self.decay_rate
        self.flux_tallied
        self.pin_flux_tallied
        self.current_tallied
        self.diffusion_coefficient
        self.mgxs_loaded = True

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Instantiate a list of the delayed groups
        delayed_groups = list(range(1,self.nd + 1))

        # Create elements and ordered dicts and initialize to None
        self._mgxs_lib = OrderedDict()

        mgxs_types = ['absorption', 'diffusion-coefficient', 'decay-rate',
                      'kappa-fission', 'chi-prompt', 'chi-delayed', 'inverse-velocity',
                      'prompt-nu-fission', 'current', 'delayed-nu-fission']

        if self.multi_group:
            mgxs_types.append('nu-scatter matrix')
        else:
            mgxs_types.append('consistent nu-scatter matrix')

        # Add the pin-wise kappa fission
        self._mgxs_lib['pin-kappa-fission'] = openmc.mgxs.MGXS.get_mgxs(
            'kappa-fission', domain=self.pin_mesh, domain_type='mesh',
            energy_groups=self.energy_groups, by_nuclide=False,
            name= self.time_point + ' - ' + 'pin-kappa-fission')

        # Populate the MGXS in the MGXS lib
        for mgxs_type in mgxs_types:
            mesh = self.flux_mesh
            if mgxs_type == 'diffusion-coefficient':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=self.fine_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
            elif 'nu-scatter matrix' in mgxs_type:
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=self.energy_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
                self._mgxs_lib[mgxs_type].correction = None
            elif mgxs_type == 'decay-rate':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=self.one_group,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type == 'chi-prompt':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=self.energy_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type in openmc.mgxs.MGXS_TYPES:
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=self.energy_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type == 'chi-delayed':
                if self.chi_delayed_by_delayed_group:
                    if self.chi_delayed_by_mesh:
                        self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                            mgxs_type, domain=mesh, domain_type='mesh',
                            energy_groups=self.energy_groups,
                            delayed_groups=delayed_groups, by_nuclide=False,
                            name= self.time_point + ' - ' + mgxs_type)
                    else:
                        self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                            mgxs_type, domain=self.unity_mesh, domain_type='mesh',
                            energy_groups=self.energy_groups,
                            delayed_groups=delayed_groups, by_nuclide=False,
                            name= self.time_point + ' - ' + mgxs_type)
                else:
                    if self.chi_delayed_by_mesh:
                        self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                            mgxs_type, domain=mesh, domain_type='mesh',
                            energy_groups=self.energy_groups, by_nuclide=False,
                            name= self.time_point + ' - ' + mgxs_type)
                    else:
                        self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                            mgxs_type, domain=self.unity_mesh, domain_type='mesh',
                            energy_groups=self.energy_groups, by_nuclide=False,
                            name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type in openmc.mgxs.MDGXS_TYPES:
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=self.energy_groups,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)

        self.mgxs_loaded = False

    @property
    def coupling_terms(self):

        # Get the dimensions of the mesh
        nz , ny , nx  = self.flux_dimension
        dx , dy , dz  = self.flux_mesh.width
        ng            = self.ng

        # Get the array of the surface-integrated surface net currents
        partial_current = copy.deepcopy(self.current_tallied)
        partial_current = partial_current.reshape(np.prod(partial_current.shape) / 12, 12)
        net_current = partial_current[:, range(0,12,2)] - partial_current[:, range(1,13,2)]
        net_current[:, 0:6:2] = -net_current[:, 0:6:2]
        net_current.shape = (nz, ny, nx, ng, 6)

        # Convert from surface-integrated to surface-averaged net current
        net_current[..., 0:2]  = net_current[..., 0:2] / (dy * dz)
        net_current[..., 2:4]  = net_current[..., 2:4] / (dx * dz)
        net_current[..., 4:6]  = net_current[..., 4:6] / (dx * dy)

        # Get the flux
        flux = copy.deepcopy(self.flux_tallied)
        flux.shape = self.flux_zyxg

        # Convert from volume-integrated to volume-averaged flux
        flux  = flux / (dx * dy * dz)

        # Create an array of the neighbor cell fluxes
        flux_nbr = np.zeros((nz, ny, nx, ng, 6))
        flux_nbr[:  , :  , 1: , :, 0] = flux[:  , :  , :-1, :]
        flux_nbr[:  , :  , :-1, :, 1] = flux[:  , :  , 1: , :]
        flux_nbr[:  , 1: , :  , :, 2] = flux[:  , :-1, :  , :]
        flux_nbr[:  , :-1, :  , :, 3] = flux[:  , 1: , :  , :]
        flux_nbr[1: , :  , :  , :, 4] = flux[:-1, :  , :  , :]
        flux_nbr[:-1, :  , :  , :, 5] = flux[1: , :  , :  , :]

        # Get the diffusion coefficients tally
        dc       = self.diffusion_coefficient
        dc.shape = self.flux_zyxg
        dc_nbr   = np.zeros(self.flux_zyxg + (6,))

        # Create array of neighbor cell diffusion coefficients
        dc_nbr[:  , :  , 1: , :, 0] = dc[:  , :  , :-1, :]
        dc_nbr[:  , :  , :-1, :, 1] = dc[:  , :  , 1: , :]
        dc_nbr[:  , 1: , :  , :, 2] = dc[:  , :-1, :  , :]
        dc_nbr[:  , :-1, :  , :, 3] = dc[:  , 1: , :  , :]
        dc_nbr[1: , :  , :  , :, 4] = dc[:-1, :  , :  , :]
        dc_nbr[:-1, :  , :  , :, 5] = dc[1: , :  , :  , :]

        # Compute the linear finite difference diffusion term for interior surfaces
        dc_linear = np.zeros((nz, ny, nx, ng, 6))
        dc_linear[:  , :  , 1: , :, 0] = 2 * dc_nbr[:  , :  , 1: , :, 0] * dc[:  , :  , 1: , :] / (dc_nbr[:  , :  , 1: , :, 0] * dx + dc[:  , :  , 1: , :] * dx)
        dc_linear[:  , :  , :-1, :, 1] = 2 * dc_nbr[:  , :  , :-1, :, 1] * dc[:  , :  , :-1, :] / (dc_nbr[:  , :  , :-1, :, 1] * dx + dc[:  , :  , :-1, :] * dx)
        dc_linear[:  , 1: , :  , :, 2] = 2 * dc_nbr[:  , 1: , :  , :, 2] * dc[:  , 1: , :  , :] / (dc_nbr[:  , 1: , :  , :, 2] * dy + dc[:  , 1: , :  , :] * dy)
        dc_linear[:  , :-1, :  , :, 3] = 2 * dc_nbr[:  , :-1, :  , :, 3] * dc[:  , :-1, :  , :] / (dc_nbr[:  , :-1, :  , :, 3] * dy + dc[:  , :-1, :  , :] * dy)
        dc_linear[1: , :  , :  , :, 4] = 2 * dc_nbr[1: , :  , :  , :, 4] * dc[1: , :  , :  , :] / (dc_nbr[1: , :  , :  , :, 4] * dz + dc[1: , :  , :  , :] * dz)
        dc_linear[:-1, :  , :  , :, 5] = 2 * dc_nbr[:-1, :  , :  , :, 5] * dc[:-1, :  , :  , :] / (dc_nbr[:-1, :  , :  , :, 5] * dz + dc[:-1, :  , :  , :] * dz)

        # Make any cells that have no dif coef or flux tally highly diffusive
        dc_linear[np.isnan(dc_linear)] = 1.e-10
        dc_linear[dc_linear == 0.] = 1.e-10

        # Compute the non-linear finite difference diffusion term for interior surfaces
        dc_nonlinear = np.zeros((nz, ny, nx, ng, 6))
        dc_nonlinear[..., 0] = (-dc_linear[..., 0] * (-flux_nbr[..., 0] + flux) - net_current[..., 0]) / (flux_nbr[..., 0] + flux)
        dc_nonlinear[..., 1] = (-dc_linear[..., 1] * ( flux_nbr[..., 1] - flux) - net_current[..., 1]) / (flux_nbr[..., 1] + flux)
        dc_nonlinear[..., 2] = (-dc_linear[..., 2] * (-flux_nbr[..., 2] + flux) - net_current[..., 2]) / (flux_nbr[..., 2] + flux)
        dc_nonlinear[..., 3] = (-dc_linear[..., 3] * ( flux_nbr[..., 3] - flux) - net_current[..., 3]) / (flux_nbr[..., 3] + flux)
        dc_nonlinear[..., 4] = (-dc_linear[..., 4] * (-flux_nbr[..., 4] + flux) - net_current[..., 4]) / (flux_nbr[..., 4] + flux)
        dc_nonlinear[..., 5] = (-dc_linear[..., 5] * ( flux_nbr[..., 5] - flux) - net_current[..., 5]) / (flux_nbr[..., 5] + flux)

        # Ensure there are no nans
        dc_nonlinear[np.isnan(dc_nonlinear)] = 0.

        flux_array = np.repeat(flux, 6)
        flux_array.shape = self.flux_zyxg + (6,)

        # Multiply by the surface are to make the terms surface integrated
        dc_linear[..., 0:2]  = dc_linear[..., 0:2] * dy*dz
        dc_linear[..., 2:4]  = dc_linear[..., 2:4] * dx*dz
        dc_linear[..., 4:6]  = dc_linear[..., 4:6] * dx*dy
        dc_nonlinear[..., 0:2] = dc_nonlinear[..., 0:2] * dy*dz
        dc_nonlinear[..., 2:4] = dc_nonlinear[..., 2:4] * dx*dz
        dc_nonlinear[..., 4:6] = dc_nonlinear[..., 4:6] * dx*dy

        # Reshape the diffusion coefficient array
        dc_linear.shape    = (nx*ny*nz*ng, 6)
        dc_nonlinear.shape = (nx*ny*nz*ng, 6)

        # Set the diagonal
        dc_linear_diag    =  dc_linear   [:, 1:6:2].sum(axis=1) + dc_linear   [:, 0:6:2].sum(axis=1)
        dc_nonlinear_diag = -dc_nonlinear[:, 1:6:2].sum(axis=1) + dc_nonlinear[:, 0:6:2].sum(axis=1)
        dc_linear_data    = [dc_linear_diag]
        dc_nonlinear_data = [dc_nonlinear_diag]
        diags             = [0]

        dc_nonlinear_copy = np.copy(dc_nonlinear)

        # Zero boundary dc_nonlinear
        dc_nonlinear_copy.shape = self.flux_zyxg + (6,)
        dc_nonlinear_copy[:  ,  :,  0, :, 0] = 0.
        dc_nonlinear_copy[:  ,  :, -1, :, 1] = 0.
        dc_nonlinear_copy[:  ,  0,  :, :, 2] = 0.
        dc_nonlinear_copy[:  , -1,  :, :, 3] = 0.
        dc_nonlinear_copy[0  ,  :,  :, :, 4] = 0.
        dc_nonlinear_copy[-1 ,  :,  :, :, 5] = 0.
        dc_nonlinear_copy.shape = (nz*ny*nx*ng, 6)

        # Set the off-diagonals
        if nx > 1:
            dc_linear_data.append(-dc_linear[ng: , 0])
            dc_linear_data.append(-dc_linear[:-ng, 1])
            dc_nonlinear_data.append( dc_nonlinear_copy[ng: , 0])
            dc_nonlinear_data.append(-dc_nonlinear_copy[:-ng, 1])
            diags.append(-ng)
            diags.append(ng)
        if ny > 1:
            dc_linear_data.append(-dc_linear[nx*ng: , 2])
            dc_linear_data.append(-dc_linear[:-nx*ng, 3])
            dc_nonlinear_data.append( dc_nonlinear_copy[nx*ng: , 2])
            dc_nonlinear_data.append(-dc_nonlinear_copy[:-nx*ng, 3])
            diags.append(-nx*ng)
            diags.append(nx*ng)
        if nz > 1:
            dc_linear_data.append(-dc_linear[nx*ny*ng: , 4])
            dc_linear_data.append(-dc_linear[:-nx*ny*ng, 5])
            dc_nonlinear_data.append( dc_nonlinear_copy[nx*ny*ng: , 4])
            dc_nonlinear_data.append(-dc_nonlinear_copy[:-nx*ny*ng, 5])
            diags.append(-nx*ny*ng)
            diags.append(nx*ny*ng)

        return diags, dc_linear_data, dc_nonlinear_data


class InnerState(State):

    def __init__(self, states):
        super(InnerState, self).__init__(states)

        # Initialize Solver class attributes
        self.fwd_state = states['FORWARD_OUTER']
        self.pre_state = states['PREVIOUS_OUTER']

    @property
    def fwd_state(self):
        return self._fwd_state

    @property
    def pre_state(self):
        return self._pre_state

    @fwd_state.setter
    def fwd_state(self, fwd_state):
        self._fwd_state = fwd_state

    @pre_state.setter
    def pre_state(self, pre_state):
        self._pre_state = pre_state

    @property
    def weight(self):
        time_point = self.clock.times[self.time_point]
        fwd_time = self.clock.times['FORWARD_OUTER']
        weight = 1 - (fwd_time - time_point) / self.clock.dt_outer
        return weight

    @property
    def inscatter(self):
        wgt = self.weight
        inscatter_fwd  = self.fwd_state.inscatter
        inscatter_prev = self.pre_state.inscatter
        inscatter = inscatter_fwd * wgt + inscatter_prev * (1 - wgt)
        inscatter[inscatter < 0.] = 0.
        return inscatter

    @property
    def outscatter(self):
        return self.inscatter.sum(axis=1)

    @property
    def absorption(self):
        wgt = self.weight
        absorption_fwd  = self.fwd_state.absorption
        absorption_prev = self.pre_state.absorption
        absorption = absorption_fwd * wgt + absorption_prev * (1 - wgt)
        absorption[absorption < 0.] = 0.
        return absorption

    @property
    def kappa_fission(self):
        wgt = self.weight
        kappa_fission_fwd  = self.fwd_state.kappa_fission
        kappa_fission_prev = self.pre_state.kappa_fission
        kappa_fission = kappa_fission_fwd * wgt + kappa_fission_prev * (1 - wgt)
        kappa_fission[kappa_fission < 0.] = 0.
        return kappa_fission

    @property
    def pin_kappa_fission(self):
        wgt = self.weight
        kappa_fission_fwd  = self.fwd_state.pin_mesh_kappa_fission
        kappa_fission_prev = self.pre_state.pin_mesh_kappa_fission
        kappa_fission = kappa_fission_fwd * wgt + kappa_fission_prev * (1 - wgt)
        kappa_fission[kappa_fission < 0.] = 0.
        return kappa_fission

    @property
    def chi_prompt(self):
        wgt = self.weight
        chi_prompt_fwd  = self.fwd_state.chi_prompt
        chi_prompt_prev = self.pre_state.chi_prompt
        chi_prompt = chi_prompt_fwd * wgt + chi_prompt_prev * (1 - wgt)
        chi_prompt[chi_prompt < 0.] = 0.
        return chi_prompt

    @property
    def prompt_nu_fission(self):
        wgt = self.weight
        prompt_nu_fission_fwd  = self.fwd_state.prompt_nu_fission
        prompt_nu_fission_prev = self.pre_state.prompt_nu_fission
        prompt_nu_fission = prompt_nu_fission_fwd * wgt + prompt_nu_fission_prev * (1 - wgt)
        prompt_nu_fission[prompt_nu_fission < 0.] = 0.
        return prompt_nu_fission

    @property
    def chi_delayed(self):
        wgt = self.weight
        chi_delayed_fwd  = self.fwd_state.chi_delayed
        chi_delayed_prev = self.pre_state.chi_delayed
        chi_delayed = chi_delayed_fwd * wgt + chi_delayed_prev * (1 - wgt)
        chi_delayed[chi_delayed < 0.] = 0.
        return chi_delayed

    @property
    def delayed_nu_fission(self):
        wgt = self.weight
        delayed_nu_fission_fwd  = self.fwd_state.delayed_nu_fission
        delayed_nu_fission_prev = self.pre_state.delayed_nu_fission
        delayed_nu_fission = delayed_nu_fission_fwd * wgt + delayed_nu_fission_prev * (1 - wgt)
        delayed_nu_fission[delayed_nu_fission < 0.] = 0.
        return delayed_nu_fission

    @property
    def inverse_velocity(self):
        wgt = self.weight
        inverse_velocity_fwd  = self.fwd_state.inverse_velocity
        inverse_velocity_prev = self.pre_state.inverse_velocity
        inverse_velocity = inverse_velocity_fwd * wgt + inverse_velocity_prev * (1 - wgt)
        inverse_velocity[inverse_velocity < 0.] = 0.
        return inverse_velocity

    @property
    def decay_rate(self):
        wgt = self.weight
        decay_rate_fwd  = self.fwd_state.decay_rate
        decay_rate_prev = self.pre_state.decay_rate
        decay_rate = decay_rate_fwd * wgt + decay_rate_prev * (1 - wgt)
        decay_rate[decay_rate < 0.] = 0.
        decay_rate[decay_rate < 1.e-5] = 0.
        return decay_rate

    @property
    def pin_shape(self):
        wgt = self.weight
        shape_fwd  = self.fwd_state.pin_shape
        shape_prev = self.pre_state.pin_shape
        shape = shape_fwd * wgt + shape_prev * (1 - wgt)
        shape[shape < 0.] = 0.
        return shape

    @property
    def coupling_terms(self):
        wgt = self.weight
        diag, dc_lin_fwd, dc_nonlin_fwd = self.fwd_state.coupling_terms
        diag, dc_lin_pre, dc_nonlin_pre = self.pre_state.coupling_terms
        dc_lin = []
        dc_nonlin = []
        for i in range(len(dc_lin_fwd)):
            dc_lin   .append(dc_lin_fwd[i]    * wgt + dc_lin_pre[i]    * (1 - wgt))
            dc_nonlin.append(dc_nonlin_fwd[i] * wgt + dc_nonlin_pre[i] * (1 - wgt))
        return diag, dc_lin, dc_nonlin

    @property
    def time_removal_source(self):
        return self.time_removal_matrix * self.states['PREVIOUS_INNER'].flux.flatten()

    @property
    def time_removal_matrix(self):
        time_removal = self.inverse_velocity / self.dt_inner * self.flux_dxyz
        return sps.diags([time_removal.flatten()], [0])

    @property
    def transient_matrix(self):
        return self.time_removal_matrix - self.prompt_production_matrix \
            + self.destruction_matrix - self.k2_source_matrix

    @property
    def k1(self):
        return np.exp(- self.dt_inner * self.decay_rate)

    @property
    def k2(self):
        # Compute k2 / (lambda * k_crit)
        k2 = 1. - (1. - self.k1) / (self.dt_inner * self.decay_rate)
        return openmc.kinetics.nan_inf_to_zero(k2 / (self.decay_rate * self.k_crit))

    @property
    def k3(self):
        # Compute k3 / (lambda * k_crit)
        k3 = self.k1 - (1. - self.k1) / (self.dt_inner * self.decay_rate)
        return openmc.kinetics.nan_inf_to_zero(k3 / (self.decay_rate * self.k_crit))

    @property
    def k1_source(self):

        state = self.states['PREVIOUS_INNER']
        source = np.repeat(self.decay_rate * state.k1 * state.precursors, self.ng)
        source.shape = (self.flux_nxyz, self.nd, self.ng)
        return (source * self.chi_delayed).sum(axis=1)

    @property
    def k2_source_matrix(self):

        k2 = np.repeat(self.decay_rate * self.k2, self.ng * self.ng)
        k2.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)

        chi = np.repeat(self.chi_delayed, self.ng)
        chi.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)

        del_fis_rate = np.tile(self.delayed_nu_fission, self.ng)
        del_fis_rate.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)

        term_k2 = chi * k2 * del_fis_rate * self.flux_dxyz

        return openmc.kinetics.block_diag(term_k2.sum(axis=1))

    @property
    def k3_source_matrix(self):

        state = self.states['PREVIOUS_INNER']

        k3 = np.repeat(self.decay_rate * state.k3, self.ng * self.ng)
        k3.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)

        chi = np.repeat(self.chi_delayed, self.ng)
        chi.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)

        del_fis_rate = np.tile(self.delayed_nu_fission, self.ng)
        del_fis_rate.shape = (self.flux_nxyz, self.nd, self.ng, self.ng)

        term_k3 = chi * k3 * del_fis_rate * self.flux_dxyz

        return openmc.kinetics.block_diag(term_k3.sum(axis=1))

    @property
    def propagate_precursors(self):

        state = self.states['PREVIOUS_INNER']

        # Contribution from current precursors
        term_k1 = state.k1 * state.precursors

        # Contribution from generation at current time point
        flux = np.tile(self.flux, self.nd)
        flux.shape = (self.flux_nxyz, self.nd, self.ng)
        term_k2 = self.k2 * (self.delayed_nu_fission * flux).sum(axis=2) * self.flux_dxyz

        # Contribution from generation at previous time step
        flux = np.tile(state.flux, state.nd)
        flux.shape = (state.flux_nxyz, state.nd, state.ng)
        term_k3 = state.k3 * (state.delayed_nu_fission * flux).sum(axis=2) * self.flux_dxyz

        self._precursors = term_k1 + term_k2 - term_k3

    @property
    def dump_to_log_file(self):

        time_point = str(self.clock.times[self.time_point])
        f = h5py.File(self._log_file, 'a')
        if time_point not in f['INNER_STEPS'].keys():
            f['INNER_STEPS'].require_group(time_point)

        if 'flux' not in f['INNER_STEPS'][time_point].keys():
            f['INNER_STEPS'][time_point].create_dataset('flux', data=self.flux)
        else:
            flux = f['INNER_STEPS'][time_point]['flux']
            flux[...] = self.flux

        f.close()
