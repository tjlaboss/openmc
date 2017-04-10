from collections import OrderedDict
from numbers import Integral
import warnings
import copy
import itertools
import sys

import numpy as np
np.set_printoptions(precision=6)
import scipy.sparse as sps

import openmc
import openmc.checkvalue as cv
import openmc.mgxs
import openmc.kinetics
from openmc.kinetics.clock import TIME_POINTS
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.version_info[0] >= 3:
    basestring = str


class State(object):
    """State to store all the variables that describe a specific state of the system.

    Attributes
    ----------
    shape_mesh : openmc.mesh.Mesh
        Mesh by which shape is computed on.

    amplitude_mesh : openmc.mesh.Mesh
        Mesh by which amplitude is computed on.

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

    def __init__(self):

        # Initialize Solver class attributes
        self._shape_mesh = None
        self._amplitude_mesh = None
        self._unity_mesh = None
        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._shape = None
        self._amplitude = None
        self._adjoint_flux = None
        self._precursors = None
        self._mgxs_lib = None
        self._k_crit = 1.0
        self._chi_delayed_by_delayed_group = False
        self._chi_delayed_by_mesh = False
        self._num_delayed_groups = 6
        self._time_point = None
        self._clock = None
        self._core_volume = 1.
        self._log_file = None
        self._multi_group = True
        self._method = 'ADIABATIC'
        self._mgxs_loaded = False

    def __deepcopy__(self, memo):

        clone = type(self).__new__(type(self))
        clone._shape_mesh = self._shape_mesh
        clone._amplitude_mesh = self._amplitude_mesh
        clone._unity_mesh = self._unity_mesh
        clone._one_group = self.one_group
        clone._energy_groups = self.energy_groups
        clone._fine_groups = self.fine_groups
        clone._shape = copy.deepcopy(self._shape)
        clone._amplitude = copy.deepcopy(self._amplitude)
        clone._adjoint_flux = copy.deepcopy(self._adjoint_flux)
        clone._precursors = copy.deepcopy(self._precursors)
        clone._mgxs_lib = copy.deepcopy(self.mgxs_lib)
        clone._k_crit = self.k_crit
        clone._chi_delayed_by_delayed_group = self._chi_delayed_by_delayed_group
        clone._chi_delayed_by_mesh = self._chi_delayed_by_mesh
        clone._num_delayed_groups = self.num_delayed_groups
        clone._time_point = self.time_point
        clone._clock = self.clock
        clone._core_volume = self.core_volume
        clone._log_file = self._log_file
        clone._multi_group = self._multi_group
        clone._method = self._method
        clone._mgxs_loaded = self._mgxs_loaded

        return clone

    @property
    def shape_mesh(self):
        return self._shape_mesh

    @property
    def amplitude_mesh(self):
        return self._amplitude_mesh

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
    def shape(self):
        return copy.deepcopy(self._shape)

    @property
    def amplitude(self):
        nxyz = np.prod(self.amplitude_mesh.dimension)
        self._amplitude.shape = (nxyz, self.ng)
        return self._amplitude

    @property
    def adjoint_flux(self):
        self._adjoint_flux.shape = (self.nxyz, self.ng)
        return self._adjoint_flux

    @property
    def precursors(self):
        self._precursors.shape = (self.nxyz, self.ng)
        return self._precursors

    @property
    def mgxs_lib(self):
        return self._mgxs_lib

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

    @property
    def method(self):
        return self._method

    @property
    def mgxs_loaded(self):
        return self._mgxs_loaded

    @shape_mesh.setter
    def shape_mesh(self, mesh):
        self._shape_mesh = mesh

    @amplitude_mesh.setter
    def amplitude_mesh(self, mesh):
        self._amplitude_mesh = mesh

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

    @shape.setter
    def shape(self, shape):
        self._shape = copy.deepcopy(shape)
        self._shape.shape = (self.nxyz, self.ng)

    @amplitude.setter
    def amplitude(self, amplitude):
        self._amplitude = copy.deepcopy(amplitude)
        nxyz = np.prod(self.amplitude_mesh.dimension)
        self._amplitude.shape = (nxyz, self.ng)

    @adjoint_flux.setter
    def adjoint_flux(self, adjoint_flux):
        self._adjoint_flux = copy.deepcopy(adjoint_flux)
        self._adjoint_flux.shape = (self.nxyz, self.ng)

    @precursors.setter
    def precursors(self, precursors):
        self._precursors = copy.deepcopy(precursors)
        self._precursors.shape = (self.nxyz, self.nd)

    @mgxs_lib.setter
    def mgxs_lib(self, mgxs_lib):
        self._mgxs_lib = mgxs_lib

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

    @method.setter
    def method(self, method):
        self._method = method

    @mgxs_loaded.setter
    def mgxs_loaded(self, mgxs_loaded):
        self._mgxs_loaded = mgxs_loaded

    @property
    def shape_dimension(self):
        return tuple(self.shape_mesh.dimension[::-1])

    @property
    def amplitude_dimension(self):
        return tuple(self.amplitude_mesh.dimension[::-1])

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
    def nxyz(self):
        return np.prod(self.shape_dimension)

    @property
    def dxyz(self):
        return np.prod(self.shape_mesh.width)

    def delayed_source(self, collapse=True):

        decay_rate = np.repeat(self.decay_rate, self.ng)
        precursors = np.repeat(self.precursors, self.ng)
        decay_rate.shape = (self.nxyz, self.nd, self.ng)
        precursors.shape = (self.nxyz, self.nd, self.ng)
        delayed_source = (self.chi_delayed * decay_rate * precursors).sum(axis=1)

        if collapse:
            coarse_shape = self.amplitude_dimension + (self.ng,)
            fine_shape = self.shape_dimension + (self.ng,)
            delayed_source.shape = fine_shape
            delayed_source = openmc.kinetics.map_array(delayed_source, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            delayed_source.shape = (nxyz, self.ng)

        return delayed_source

    def time_source(self, state_prev, collapse=True):

        flux_prev = state_prev.flux
        amp = self.amplitude
        coarse_shape = self.amplitude_dimension + (self.ng,)
        fine_shape = self.shape_dimension + (self.ng,)
        amp.shape = coarse_shape
        time_source = openmc.kinetics.map_array(amp, fine_shape, normalize=True)
        time_source.shape = (self.nxyz, self.ng)
        time_source *= flux_prev * self.inverse_velocity / self.dt_outer

        if collapse:
            time_source.shape = fine_shape
            time_source = openmc.kinetics.map_array(time_source, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            time_source.shape = (nxyz, self.ng)

        return time_source

    def amplitude_derivative_matrix(self, amp_prev, collapse=True):

        amp = self.amplitude
        amp_deriv = (amp - amp_prev) / amp / self.dt_inner
        coarse_shape = self.amplitude_dimension + (self.ng,)
        fine_shape = self.shape_dimension + (self.ng,)
        amp_deriv.shape = coarse_shape
        amp_deriv = openmc.kinetics.map_array(amp_deriv, fine_shape, normalize=True)
        amp_deriv.shape = (self.nxyz, self.ng)
        amp_deriv *= self.inverse_velocity * self.dxyz

        if collapse:
            amp_deriv.shape = fine_shape
            amp_deriv = openmc.kinetics.map_array(amp_deriv, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            amp_deriv.shape = (nxyz, self.ng)

        return sps.diags(amp_deriv.flatten())

    @property
    def reactivity(self):
        production = self.production_matrix(False) * self.flux.flatten()
        destruction = self.destruction_matrix(False) * self.flux.flatten()
        balance = production - destruction
        balance = balance * self.adjoint_flux.flatten()
        production = production * self.adjoint_flux.flatten()
        return balance.sum() / production.sum()

    @property
    def beta_eff(self):
        flux = np.tile(self.flux, self.nd)
        adjoint_flux = np.tile(self.adjoint_flux, self.nd)

        delayed_production = self.delayed_production * self.dxyz
        delayed_production.shape = (self.nxyz * self.nd, self.ng, self.ng)
        delayed_production = openmc.kinetics.diagonal_matrix(delayed_production)
        delayed_production /= self.k_crit

        delayed_production *= flux.flatten()
        delayed_production *= adjoint_flux.flatten()
        delayed_production.shape = (self.nxyz, self.nd, self.ng)
        delayed_production = delayed_production.sum(axis=(0,2))

        production = self.production_matrix(False) * self.flux.flatten()
        production *= self.adjoint_flux.flatten()
        production.shape = (self.nxyz, self.ng)
        production = production.sum(axis=(0,1))
        production = np.repeat(production, self.nd)

        return (delayed_production / production).sum()

    @property
    def beta(self):
        flux = np.tile(self.flux, self.nd)

        delayed_production = self.delayed_production * self.dxyz
        delayed_production.shape = (self.nxyz * self.nd, self.ng, self.ng)
        delayed_production = openmc.kinetics.diagonal_matrix(delayed_production)
        delayed_production /= self.k_crit

        delayed_production *= flux.flatten()
        delayed_production.shape = (self.nxyz, self.nd, self.ng)
        delayed_production = delayed_production.sum(axis=(0,2))

        production = self.production_matrix(False) * self.flux.flatten()
        production.shape = (self.nxyz, self.ng)
        production = production.sum(axis=(0,1))
        production = np.repeat(production, self.nd)

        return (delayed_production / production).sum()

    @property
    def pnl(self):
        inv_velocity = self.adjoint_flux * self.inverse_velocity * self.flux * self.dxyz
        production = self.production_matrix(False) * self.flux.flatten()
        production = production * self.adjoint_flux.flatten()
        return inv_velocity.sum() / production.sum()

    @property
    def inscatter(self):
        if not self.mgxs_loaded:
            if self.multi_group:
                self._inscatter = self.mgxs_lib['nu-scatter matrix'].get_xs(row_column='outin')
            else:
                self._inscatter = self.mgxs_lib['consistent nu-scatter matrix'].get_xs(row_column='outin')

        self._inscatter.shape = (self.nxyz, self.ng, self.ng)
        return self._inscatter

    @property
    def outscatter(self):
        return self.inscatter.sum(axis=1)

    @property
    def absorption(self):
        if not self.mgxs_loaded:
            self._absorption = self.mgxs_lib['absorption'].get_xs()

        self._absorption.shape = (self.nxyz, self.ng)
        return self._absorption

    @property
    def kappa_fission(self):
        if not self.mgxs_loaded:
            self._kappa_fission = self.mgxs_lib['kappa-fission'].get_xs()

        self._kappa_fission.shape = (self.nxyz, self.ng)
        return self._kappa_fission

    def flux_frequency(self, state_prev):

        freq = (1. / self.dt_outer - state_prev.flux / self.flux / self.dt_outer) * self.dxyz

        freq[freq == -np.inf] = 0.
        freq[freq ==  np.inf] = 0.
        freq = np.nan_to_num(freq)

        freq.shape = self.shape_dimension + (self.ng,)
        coarse_shape = (1,1,1,self.ng)
        freq = openmc.kinetics.map_array(freq, coarse_shape, normalize=True)
        return freq

    def precursor_frequency(self):

        flux = np.tile(self.flux, self.nd).flatten()
        del_fis_rate = self.delayed_nu_fission.flatten() * flux
        del_fis_rate.shape = (self.nxyz, self.nd, self.ng)
        freq = del_fis_rate.sum(axis=2) / self.precursors / self.k_crit - self.decay_rate

        freq = self.decay_rate / (freq + self.decay_rate) * self.dxyz

        freq[freq == -np.inf] = 0.
        freq[freq ==  np.inf] = 0.
        freq = np.nan_to_num(freq)

        return freq

    def destruction_matrix(self, collapse=True, omega=False):

        linear, non_linear = self.coupling(collapse)
        inscatter       = self.inscatter * self.dxyz
        absorb_outscat  = self.outscatter + self.absorption
        absorb_outscat *= self.dxyz

        if collapse:
            inscatter      *= np.tile(self.shape, self.ng)
            absorb_outscat *= self.shape
            inscatter.shape = self.shape_dimension + (self.ng, self.ng)
            absorb_outscat.shape = self.shape_dimension + (self.ng,)
            coarse_shape = self.amplitude_dimension + (self.ng, self.ng)
            inscatter = openmc.kinetics.map_array(inscatter, coarse_shape, normalize=False)
            coarse_shape = self.amplitude_dimension + (self.ng,)
            absorb_outscat = openmc.kinetics.map_array(absorb_outscat, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            inscatter.shape = (nxyz, self.ng, self.ng)
        else:
            inscatter.shape = (self.nxyz, self.ng, self.ng)

        total = sps.diags(absorb_outscat.flatten()) - openmc.kinetics.diagonal_matrix(inscatter)

        if omega:
            total -= sps.diags(self.flux_frequency(self.states['PREVIOUS_OUT']).flatten())

        return total + linear + non_linear

    @property
    def chi_prompt(self):
        if not self.mgxs_loaded:
            self._chi_prompt = self.mgxs_lib['chi-prompt'].get_xs()

        self._chi_prompt.shape = (self.nxyz, self.ng)
        return self._chi_prompt

    @property
    def prompt_nu_fission(self):
        if not self.mgxs_loaded:
            self._prompt_nu_fission = self.mgxs_lib['prompt-nu-fission'].get_xs()

        self._prompt_nu_fission.shape = (self.nxyz, self.ng)
        return self._prompt_nu_fission

    @property
    def chi_delayed(self):

        if not self.mgxs_loaded:
            self._chi_delayed = self.mgxs_lib['chi-delayed'].get_xs()

            if self.chi_delayed_by_mesh:
                if not self.chi_delayed_by_delayed_group:
                    self._chi_delayed.shape = (self.nxyz, self.ng)
                    self._chi_delayed = np.tile(self._chi_delayed, self.nd)
            else:
                if self.chi_delayed_by_delayed_group:
                    self._chi_delayed = np.tile(self._chi_delayed.flatten(), self.nxyz)
                else:
                    self._chi_delayed = np.tile(self._chi_delayed.flatten(), self.nxyz)
                    self._chi_delayed.shape = (self.nxyz, self.ng)
                    self._chi_delayed = np.tile(self._chi_delayed, self.nd)

        self._chi_delayed.shape = (self.nxyz, self.nd, self.ng)
        return self._chi_delayed

    @property
    def delayed_nu_fission(self):
        if not self.mgxs_loaded:
            self._delayed_nu_fission = self.mgxs_lib['delayed-nu-fission'].get_xs()

        self._delayed_nu_fission.shape = (self.nxyz, self.nd, self.ng)
        return self._delayed_nu_fission

    @property
    def delayed_production(self):
        chi_delayed = np.repeat(self.chi_delayed, self.ng)
        chi_delayed.shape = (self.nxyz, self.nd, self.ng, self.ng)
        delayed_nu_fission = np.tile(self.delayed_nu_fission, self.ng)
        delayed_nu_fission.shape = (self.nxyz, self.nd, self.ng, self.ng)
        return (chi_delayed * delayed_nu_fission)

    def delayed_production_matrix(self, collapse=True, omega=False):

        if omega:
            chi_delayed = self.chi_delayed * self.precursor_frequency()
            chi_delayed = np.repeat(chi_delayed, self.ng)
            chi_delayed.shape = (self.nxyz, self.nd, self.ng, self.ng)
            delayed_nu_fission = np.tile(self.delayed_nu_fission, self.ng)
            delayed_nu_fission.shape = (self.nxyz, self.nd, self.ng, self.ng)
            delayed_production = (chi_delayed * delayed_nu_fission).sum(axis=1)
        else:
            delayed_production = self.delayed_production.sum(axis=1) * self.dxyz

        if collapse:
            shape = np.tile(self.shape, self.ng).flatten()
            delayed_production = delayed_production.flatten() * shape
            delayed_production.shape = self.shape_dimension + (self.ng, self.ng)
            coarse_shape = self.amplitude_dimension + (self.ng, self.ng)
            delayed_production = openmc.kinetics.map_array(delayed_production, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            delayed_production.shape = (nxyz, self.ng, self.ng)

        delayed_production = openmc.kinetics.diagonal_matrix(delayed_production)
        return delayed_production / self.k_crit

    def production_matrix(self, collapse=True, omega=False):
        return self.prompt_production_matrix(collapse) + self.delayed_production_matrix(collapse, omega)

    @property
    def prompt_production(self):
        chi_prompt = np.repeat(self.chi_prompt, self.ng)
        chi_prompt.shape = (self.nxyz, self.ng, self.ng)
        prompt_nu_fission = np.tile(self.prompt_nu_fission, self.ng)
        prompt_nu_fission.shape = (self.nxyz, self.ng, self.ng)
        return (chi_prompt * prompt_nu_fission)

    def prompt_production_matrix(self, collapse=True):
        prompt_production = self.prompt_production * self.dxyz

        if collapse:
            shape = np.tile(self.shape, self.ng).reshape((self.nxyz, self.ng, self.ng))
            prompt_production = prompt_production * shape
            prompt_production.shape = self.shape_dimension + (self.ng, self.ng)
            coarse_shape = self.amplitude_dimension + (self.ng, self.ng)
            prompt_production = openmc.kinetics.map_array(prompt_production, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            prompt_production.shape = (nxyz, self.ng, self.ng)

        prompt_production = openmc.kinetics.diagonal_matrix(prompt_production)
        return prompt_production / self.k_crit

    @property
    def core_power_density(self):
        mesh_volume = self.dxyz * self.nxyz
        return self.power.sum() * mesh_volume / self.core_volume

    def time_removal_matrix(self, collapse=True, inner=True):
        if inner:
            time_removal = self.inverse_velocity / self.dt_inner * self.dxyz
        else:
            time_removal = self.inverse_velocity / self.dt_outer * self.dxyz

        if collapse:
            time_removal *= self.shape
            time_removal.shape = self.shape_dimension + (self.ng,)
            coarse_shape = self.amplitude_dimension + (self.ng,)
            time_removal = openmc.kinetics.map_array(time_removal, coarse_shape, normalize=False)

        return sps.diags(time_removal.flatten(), 0)

    @property
    def inverse_velocity(self):
        if not self.mgxs_loaded:
            self._inverse_velocity = self.mgxs_lib['inverse-velocity'].get_xs()

        self._inverse_velocity.shape = (self.nxyz, self.ng)
        return self._inverse_velocity

    @property
    def decay_rate(self):
        if not self.mgxs_loaded:
            self._decay_rate = self.mgxs_lib['decay-rate'].get_xs()
            self._decay_rate[self._decay_rate < 1.e-5] = 0.

        self._decay_rate.shape = (self.nxyz, self.nd)
        return self._decay_rate

    @property
    def flux_tallied(self):
        if not self.mgxs_loaded:
            self._flux_tallied = self.mgxs_lib['kappa-fission'].tallies['flux'].get_values()
            self._flux_tallied.shape = (self.nxyz, self.ng)
            self._flux_tallied = self._flux_tallied[:, ::-1]

        self._flux_tallied.shape = (self.nxyz, self.ng)
        return self._flux_tallied

    @property
    def current_tallied(self):
        if not self.mgxs_loaded:
            self._current_tallied = self.mgxs_lib['current'].get_xs()

        return self._current_tallied

    @property
    def flux(self):
        amp = self.amplitude
        fine_shape = self.shape_dimension + (self.ng,)
        amp.shape = self.amplitude_dimension + (self.ng,)
        fine_amp = openmc.kinetics.map_array(amp, fine_shape, normalize=True)
        flux = fine_amp.flatten() * self.shape.flatten()
        flux.shape = (self.nxyz, self.ng)
        return flux

    @property
    def power(self):
        return (self.dxyz * self.kappa_fission * self.flux / self.k_crit).sum(axis=1)

    def dump_inner_to_log_file(self):

        time_point = str(self.clock.times[self.time_point])
        f = h5py.File(self._log_file, 'a')
        if time_point not in f['time_steps'].keys():
            f['time_steps'].require_group(time_point)
            f['time_steps'][time_point]['dump_type'] = 'inner'

        f['time_steps'][time_point].attrs['reactivity'] = self.reactivity
        f['time_steps'][time_point].attrs['beta_eff'] = self.beta_eff
        f['time_steps'][time_point].attrs['pnl'] = self.pnl
        f['time_steps'][time_point].attrs['core_power_density'] = self.core_power_density
        f.close()

    def dump_outer_to_log_file(self):

        time_point = str(self.clock.times[self.time_point])
        f = h5py.File(self._log_file, 'a')
        if time_point not in f['time_steps'].keys():
            f['time_steps'].require_group(time_point)

        f['time_steps'][time_point]['dump_type'] = 'outer'
        f['time_steps'][time_point]['amplitude'] = self.amplitude
        f['time_steps'][time_point]['shape'] = self.shape
        f['time_steps'][time_point]['flux'] = self.flux
        f['time_steps'][time_point]['precursors'] = self.precursors
        f['time_steps'][time_point].attrs['reactivity'] = self.reactivity
        f['time_steps'][time_point].attrs['beta_eff'] = self.beta_eff
        f['time_steps'][time_point].attrs['pnl'] = self.pnl
        f['time_steps'][time_point].attrs['core_power_density'] = self.core_power_density
        f['time_steps'][time_point]['kappa_fission'] = self.kappa_fission
        f['time_steps'][time_point]['power'] = self.power

        f.close()

    def compute_initial_precursor_concentration(self):
        flux = np.tile(self.flux, self.nd).flatten()
        del_fis_rate = self.delayed_nu_fission.flatten() * flux
        del_fis_rate.shape = (self.nxyz, self.nd, self.ng)
        precursors = del_fis_rate.sum(axis=2) / self.decay_rate / self.k_crit * self.dxyz
        precursors[precursors == -np.inf] = 0.
        precursors[precursors ==  np.inf] = 0.
        self.precursors = np.nan_to_num(precursors)

    def load_mgxs(self):
        self.mgxs_loaded = False
        self.inscatter
        self.absorption
        self.chi_prompt
        self.prompt_nu_fission
        self.chi_delayed
        self.delayed_nu_fission
        self.kappa_fission
        self.inverse_velocity
        self.decay_rate
        self.flux_tallied
        self.current_tallied
        self.diffusion_coefficient
        self.mgxs_loaded = True

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Instantiate a list of the delayed groups
        delayed_groups = list(range(1,self.num_delayed_groups + 1))

        # Create elements and ordered dicts and initialize to None
        self._mgxs_lib = OrderedDict()

        mgxs_types = ['absorption', 'diffusion-coefficient', 'decay-rate',
                      'kappa-fission', 'chi-prompt', 'chi-delayed', 'inverse-velocity',
                      'prompt-nu-fission', 'current', 'delayed-nu-fission']

        if self.multi_group:
            mgxs_types.append('nu-scatter matrix')
        else:
            mgxs_types.append('consistent nu-scatter matrix')

        # Populate the MGXS in the MGXS lib
        for mgxs_type in mgxs_types:
            mesh = self.shape_mesh
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
    def diffusion_coefficient(self):
        if not self.mgxs_loaded:
            self._diffusion_coefficient = self.mgxs_lib['diffusion-coefficient']
            self._diffusion_coefficient = self._diffusion_coefficient.get_condensed_xs(self.energy_groups).get_xs()

        self._diffusion_coefficient.shape = (self.nxyz, self.ng)
        return self._diffusion_coefficient

    def extract_shape(self, flux, power):
        coarse_shape    = self.amplitude_dimension + (self.ng,)
        fine_shape      = self.shape_dimension + (self.ng,)
        amplitude       = self.amplitude
        amplitude.shape = coarse_shape
        flux.shape      = fine_shape
        fine_amp        = openmc.kinetics.map_array(amplitude, fine_shape, normalize=True)
        fine_amp.shape  = fine_shape

        self.shape     = flux / fine_amp

        # Normalize shape
        if self.method == 'ADIABATIC':
            self.shape *= power / self.core_power_density

    def coupling(self, collapse=True, check_for_diag_dominance=False):

        # Get the dimensions of the mesh
        nz , ny , nx  = self.shape_dimension
        nza, nya, nxa = self.amplitude_dimension
        dx , dy , dz  = self.shape_mesh.width
        ng            = self.ng

        # Get the array of the surface-integrated surface net currents
        partial_current = copy.deepcopy(self.current_tallied)
        partial_current = partial_current.reshape(np.prod(partial_current.shape) / 12, 12)
        net_current = partial_current[:, range(0,12,2)] - partial_current[:, range(1,13,2)]
        net_current[:, 0:6:2] = -net_current[:, 0:6:2]
        net_current.shape = (nz, ny, nx, ng, 6)

        # Convert from surface-integrated to surface-averaged net current
        net_current[..., 0:2] /= (dy * dz)
        net_current[..., 2:4] /= (dx * dz)
        net_current[..., 4:6] /= (dx * dy)

        # Get the flux
        flux = copy.deepcopy(self.flux_tallied)
        flux.shape = (nz, ny, nx, ng)

        # Convert from volume-integrated to volume-averaged flux
        flux /= (dx * dy * dz)

        # Create an array of the neighbor cell fluxes
        flux_nbr = np.zeros((nz, ny, nx, ng, 6))
        flux_nbr[:  , :  , 1: , :, 0] = flux[:  , :  , :-1, :]
        flux_nbr[:  , :  , :-1, :, 1] = flux[:  , :  , 1: , :]
        flux_nbr[:  , 1: , :  , :, 2] = flux[:  , :-1, :  , :]
        flux_nbr[:  , :-1, :  , :, 3] = flux[:  , 1: , :  , :]
        flux_nbr[1: , :  , :  , :, 4] = flux[:-1, :  , :  , :]
        flux_nbr[:-1, :  , :  , :, 5] = flux[1: , :  , :  , :]

        if collapse:

            # Get the shape
            shape       = self.shape
            shape.shape = (nz, ny, nx, ng)

            # Create an array of the neighbor cell shapes
            shape_nbr = np.zeros((nz, ny, nx, ng, 6))
            shape_nbr[:  , :  , 1: , :, 0] = shape[:  , :  , :-1, :]
            shape_nbr[:  , :  , :-1, :, 1] = shape[:  , :  , 1: , :]
            shape_nbr[:  , 1: , :  , :, 2] = shape[:  , :-1, :  , :]
            shape_nbr[:  , :-1, :  , :, 3] = shape[:  , 1: , :  , :]
            shape_nbr[1: , :  , :  , :, 4] = shape[:-1, :  , :  , :]
            shape_nbr[:-1, :  , :  , :, 5] = shape[1: , :  , :  , :]

        # Get the diffusion coefficients tally
        dc       = self.diffusion_coefficient
        dc.shape = (nz, ny, nx, ng)
        dc_nbr   = np.zeros((nz, ny, nx, ng, 6))

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
        flux_array.shape = (nz, ny, nx, ng, 6)

        # Check for diagonal dominance
        if check_for_diag_dominance:

            # Make a mask of the location of all terms that need to be corrected (dd_mask) and
            # terms that don't need to be corrected (nd_mask)
            dd_mask = (np.abs(dc_nonlinear) > dc_linear)
            nd_mask = (dd_mask == False)

            # Save arrays as to whether the correction term is positive or negative.
            sign = np.abs(dc_nonlinear) / dc_nonlinear
            sign_pos  = (dc_nonlinear > 0.)
            sense_pos = np.zeros((nz, ny, nx, ng, 6))
            sense_pos[..., 0:6:2] = False
            sense_pos[..., 1:6:2] = True
            sign_sense = (sign_pos == sense_pos)
            not_sign_sense = (sign_sense == False)

            # Correct dc_linear
            dc_linear[:  , :  , 1: , :, 0] = nd_mask[:  , :  , 1: , :, 0] * dc_linear[:  , :  , 1: , :, 0] + dd_mask[:  , :  , 1: , :, 0] * (sign_sense[:  , :  , 1: , :, 0] * np.abs(net_current[:  , :  , 1: , :, 0] / (2 * flux_nbr[:  , :  , 1: , :, 0])) + not_sign_sense[:  , :  , 1: , :, 0] * np.abs(net_current[:  , :  , 1: , :, 0] / (2 * flux[:  , :  , 1: , :])))
            dc_linear[:  , :  , :-1, :, 1] = nd_mask[:  , :  , :-1, :, 1] * dc_linear[:  , :  , :-1, :, 1] + dd_mask[:  , :  , :-1, :, 1] * (sign_sense[:  , :  , :-1, :, 1] * np.abs(net_current[:  , :  , :-1, :, 1] / (2 * flux_nbr[:  , :  , :-1, :, 1])) + not_sign_sense[:  , :  , :-1, :, 1] * np.abs(net_current[:  , :  , :-1, :, 1] / (2 * flux[:  , :  , :-1, :])))
            dc_linear[:  , 1: , :  , :, 2] = nd_mask[:  , 1: , :  , :, 2] * dc_linear[:  , 1: , :  , :, 2] + dd_mask[:  , 1: , :  , :, 2] * (sign_sense[:  , 1: , :  , :, 2] * np.abs(net_current[:  , 1: , :  , :, 2] / (2 * flux_nbr[:  , 1: , :  , :, 2])) + not_sign_sense[:  , 1: , :  , :, 2] * np.abs(net_current[:  , 1: , :  , :, 2] / (2 * flux[:  , 1: , :  , :])))
            dc_linear[:  , :-1, :  , :, 3] = nd_mask[:  , :-1, :  , :, 3] * dc_linear[:  , :-1, :  , :, 3] + dd_mask[:  , :-1, :  , :, 3] * (sign_sense[:  , :-1, :  , :, 3] * np.abs(net_current[:  , :-1, :  , :, 3] / (2 * flux_nbr[:  , :-1, :  , :, 3])) + not_sign_sense[:  , :-1, :  , :, 3] * np.abs(net_current[:  , :-1, :  , :, 3] / (2 * flux[:  , :-1, :  , :])))
            dc_linear[1: , :  , :  , :, 4] = nd_mask[1: , :  , :  , :, 4] * dc_linear[1: , :  , :  , :, 4] + dd_mask[1: , :  , :  , :, 4] * (sign_sense[1: , :  , :  , :, 4] * np.abs(net_current[1: , :  , :  , :, 4] / (2 * flux_nbr[1: , :  , :  , :, 4])) + not_sign_sense[1: , :  , :  , :, 4] * np.abs(net_current[1: , :  , :  , :, 4] / (2 * flux[1: , :  , :  , :])))
            dc_linear[:-1, :  , :  , :, 5] = nd_mask[:-1, :  , :  , :, 5] * dc_linear[:-1, :  , :  , :, 5] + dd_mask[:-1, :  , :  , :, 5] * (sign_sense[:-1, :  , :  , :, 5] * np.abs(net_current[:-1, :  , :  , :, 5] / (2 * flux_nbr[:-1, :  , :  , :, 5])) + not_sign_sense[:-1, :  , :  , :, 5] * np.abs(net_current[:-1, :  , :  , :, 5] / (2 * flux[:-1, :  , :  , :])))

            dc_nonlinear[:  , :  , 1: , :, 0] = nd_mask[:  , :  , 1: , :, 0] * dc_nonlinear[:  , :  , 1: , :, 0] + dd_mask[:  , :  , 1: , :, 0] * sign[:  , :  , 1: , :, 0] * dc_linear[:  , :  , 1: , :, 0]
            dc_nonlinear[:  , :  , :-1, :, 1] = nd_mask[:  , :  , :-1, :, 1] * dc_nonlinear[:  , :  , :-1, :, 1] + dd_mask[:  , :  , :-1, :, 1] * sign[:  , :  , :-1, :, 1] * dc_linear[:  , :  , :-1, :, 1]
            dc_nonlinear[:  , 1: , :  , :, 2] = nd_mask[:  , 1: , :  , :, 2] * dc_nonlinear[:  , 1: , :  , :, 2] + dd_mask[:  , 1: , :  , :, 2] * sign[:  , 1: , :  , :, 2] * dc_linear[:  , 1: , :  , :, 2]
            dc_nonlinear[:  , :-1, :  , :, 3] = nd_mask[:  , :-1, :  , :, 3] * dc_nonlinear[:  , :-1, :  , :, 3] + dd_mask[:  , :-1, :  , :, 3] * sign[:  , :-1, :  , :, 3] * dc_linear[:  , :-1, :  , :, 3]
            dc_nonlinear[1: , :  , :  , :, 4] = nd_mask[1: , :  , :  , :, 4] * dc_nonlinear[1: , :  , :  , :, 4] + dd_mask[1: , :  , :  , :, 4] * sign[1: , :  , :  , :, 4] * dc_linear[1: , :  , :  , :, 4]
            dc_nonlinear[:-1, :  , :  , :, 5] = nd_mask[:-1, :  , :  , :, 5] * dc_nonlinear[:-1, :  , :  , :, 5] + dd_mask[:-1, :  , :  , :, 5] * sign[:-1, :  , :  , :, 5] * dc_linear[:-1, :  , :  , :, 5]

        # Multiply by the surface are to make the terms surface integrated
        dc_linear[..., 0:2] *= dy*dz
        dc_linear[..., 2:4] *= dx*dz
        dc_linear[..., 4:6] *= dx*dy
        dc_nonlinear[..., 0:2] *= dy*dz
        dc_nonlinear[..., 2:4] *= dx*dz
        dc_nonlinear[..., 4:6] *= dx*dy

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
        dc_nonlinear_copy.shape = (nz, ny, nx, ng, 6)
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

        # Collapse the shape weighted coefficients
        if collapse:

            # reshape the flux shape
            shape.shape = (nx*ny*nz*ng,)
            shape_nbr.shape = (nx*ny*nz*ng, 6)
            coarse_shape = (nza, nya, nxa, ng)

            # copy arrays
            dc_linear_data_copy = copy.deepcopy(dc_linear_data)
            dc_nonlinear_data_copy = copy.deepcopy(dc_nonlinear_data)
            diags_copy = copy.deepcopy(diags)

            # reinitialize the arrays
            dc_linear_data = []
            dc_nonlinear_data = []
            diags = []

            # shape weight and spatially collapse the dc_linear and dc_nonlinear
            for i,diag in enumerate(diags_copy):
                if diag == 0:
                    dc_linear_data     .append(dc_linear_data_copy[i]      * shape)
                    dc_nonlinear_data.append(dc_nonlinear_data_copy[i] * shape)
                    dc_linear_data[-1].shape      = (nz, ny, nx, ng)
                    dc_nonlinear_data[-1].shape = (nz, ny, nx, ng)
                    dc_linear_data[-1]            = openmc.kinetics.map_array(dc_linear_data[-i], coarse_shape, normalize=False).flatten()
                    dc_nonlinear_data[-1]       = openmc.kinetics.map_array(dc_nonlinear_data[-i], coarse_shape, normalize=False).flatten()
                    diags = [0]
                elif diag == -ng:

                    # Shape weight
                    dc_linear      = dc_linear_data_copy[i]      * shape_nbr[ng:, 0]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[ng:, 0]

                    # Elongate
                    dc_linear      = np.append(np.zeros(ng), dc_linear)
                    dc_nonlinear = np.append(np.zeros(ng), dc_nonlinear)

                    # Reshape
                    dc_linear.shape      = (nz, ny, nx, ng)
                    dc_nonlinear.shape = (nz, ny, nx, ng)

                    # Extract
                    ind_diag     = sum([range(i*nx/nxa + 1, (i+1)*nx/nxa) for i in range(nxa)], [])
                    ind_off_diag = sum([range(i*nx/nxa    , i*nx/nxa + 1) for i in range(nxa)], [])
                    dc_linear_diag          = dc_linear     [:, :, ind_diag    , :]
                    dc_linear_off_diag      = dc_linear     [:, :, ind_off_diag, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, :, ind_diag    , :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, :, ind_off_diag, :]

                    # Condense and add values to diag
                    if nx != nxa:
                        dc_linear_diag          = openmc.kinetics.map_array(dc_linear_diag         , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag    , coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]      += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nxa > 1:
                        dc_linear_off_diag      = openmc.kinetics.map_array(dc_linear_off_diag     , coarse_shape, normalize=False).flatten()[ng:]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[ng:]
                        dc_linear_data     .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags        .append(-ng)

                elif diag == ng:

                    # Shape weight
                    dc_linear      = dc_linear_data_copy[i]      * shape_nbr[:-ng, 1]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[:-ng, 1]

                    # Elongate
                    dc_linear      = np.append(dc_linear     , np.zeros(ng))
                    dc_nonlinear = np.append(dc_nonlinear, np.zeros(ng))

                    # Reshape
                    dc_linear.shape      = (nz, ny, nx, ng)
                    dc_nonlinear.shape = (nz, ny, nx, ng)

                    # Extract
                    ind_diag     = sum([range(i*nx/nxa      , (i+1)*nx/nxa-1) for i in range(nxa)], [])
                    ind_off_diag = sum([range((i+1)*nx/nxa-1, (i+1)*nx/nxa  ) for i in range(nxa)], [])
                    dc_linear_diag          = dc_linear     [:, :, ind_diag    , :]
                    dc_linear_off_diag      = dc_linear     [:, :, ind_off_diag, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, :, ind_diag    , :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, :, ind_off_diag, :]

                    # Condense and add values to diag
                    if nx != nxa:
                        dc_linear_diag          = openmc.kinetics.map_array(dc_linear_diag         , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag    , coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]      += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nxa > 1:
                        dc_linear_off_diag      = openmc.kinetics.map_array(dc_linear_off_diag     , coarse_shape, normalize=False).flatten()[:-ng]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[:-ng]
                        dc_linear_data     .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags        .append(ng)

                elif diag == -ng*nx:

                    # Shape weight
                    dc_linear      = dc_linear_data_copy[i]      * shape_nbr[nx*ng:, 2]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[nx*ng:, 2]

                    # Elongate
                    dc_linear      = np.append(np.zeros(nx*ng), dc_linear)
                    dc_nonlinear = np.append(np.zeros(nx*ng), dc_nonlinear)

                    # Reshape
                    dc_linear.shape      = (nz, ny, nx, ng)
                    dc_nonlinear.shape = (nz, ny, nx, ng)

                    # Extract
                    ind_diag     = sum([range(i*ny/nya + 1, (i+1)*ny/nya) for i in range(nya)], [])
                    ind_off_diag = sum([range(i*ny/nya    , i*ny/nya + 1) for i in range(nya)], [])
                    dc_linear_diag          = dc_linear     [:, ind_diag    , :, :]
                    dc_linear_off_diag      = dc_linear     [:, ind_off_diag, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, ind_diag    , :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, ind_off_diag, :, :]

                    # Condense and add values to diag
                    if ny != nya:
                        dc_linear_diag          = openmc.kinetics.map_array(dc_linear_diag         , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag    , coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]      += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nya > 1:
                        dc_linear_off_diag      = openmc.kinetics.map_array(dc_linear_off_diag     , coarse_shape, normalize=False).flatten()[nxa*ng:]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[nxa*ng:]
                        dc_linear_data     .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags        .append(-nxa*ng)

                elif diag == ng*nx:

                    # Shape weight
                    dc_linear      = dc_linear_data_copy[i]      * shape_nbr[:-nx*ng, 3]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[:-nx*ng, 3]

                    # Elongate
                    dc_linear      = np.append(dc_linear     , np.zeros(nx*ng))
                    dc_nonlinear = np.append(dc_nonlinear, np.zeros(nx*ng))

                    # Reshape
                    dc_linear.shape      = (nz, ny, nx, ng)
                    dc_nonlinear.shape = (nz, ny, nx, ng)

                    # Extract
                    ind_diag     = sum([range(i*ny/nya      , (i+1)*ny/nya-1) for i in range(nya)], [])
                    ind_off_diag = sum([range((i+1)*ny/nya-1, (i+1)*ny/nya  ) for i in range(nya)], [])
                    dc_linear_diag          = dc_linear     [:, ind_diag    , :, :]
                    dc_linear_off_diag      = dc_linear     [:, ind_off_diag, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, ind_diag    , :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, ind_off_diag, :, :]

                    # Condense and add values to diag
                    if ny != nya:
                        dc_linear_diag          = openmc.kinetics.map_array(dc_linear_diag         , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag    , coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]      += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nya > 1:
                        dc_linear_off_diag      = openmc.kinetics.map_array(dc_linear_off_diag     , coarse_shape, normalize=False).flatten()[:-nxa*ng]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[:-nxa*ng]
                        dc_linear_data     .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags        .append(nxa*ng)

                elif diag == -ng*nx*ny:

                    # Shape weight
                    dc_linear      = dc_linear_data_copy[i]      * shape_nbr[ny*nx*ng:, 4]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[ny*nx*ng:, 4]

                    # Elongate
                    dc_linear      = np.append(np.zeros(ny*nx*ng), dc_linear)
                    dc_nonlinear = np.append(np.zeros(ny*nx*ng), dc_nonlinear)

                    # Reshape
                    dc_linear.shape      = (nz, ny, nx, ng)
                    dc_nonlinear.shape = (nz, ny, nx, ng)

                    # Extract
                    ind_diag     = sum([range(i*nz/nza + 1, (i+1)*nz/nza) for i in range(nza)], [])
                    ind_off_diag = sum([range(i*nz/nza    , i*nz/nza + 1) for i in range(nza)], [])
                    dc_linear_diag          = dc_linear     [ind_diag    , :, :, :]
                    dc_linear_off_diag      = dc_linear     [ind_off_diag, :, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[ind_diag    , :, :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[ind_off_diag, :, :, :]

                    # Condense and add values to diag
                    if nz != nza:
                        dc_linear_diag          = openmc.kinetics.map_array(dc_linear_diag         , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag    , coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]      += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nza > 1:
                        dc_linear_off_diag      = openmc.kinetics.map_array(dc_linear_off_diag     , coarse_shape, normalize=False).flatten()[nya*nxa*ng:]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[nya*nxa*ng:]
                        dc_linear_data     .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags        .append(-nxa*ng)

                elif diag == ng*nx*ny:

                    # Shape weight
                    dc_linear      = dc_linear_data_copy[i]      * shape_nbr[:-ny*nx*ng, 5]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[:-ny*nx*ng, 5]

                    # Elongate
                    dc_linear      = np.append(dc_linear     , np.zeros(ny*nx*ng))
                    dc_nonlinear = np.append(dc_nonlinear, np.zeros(ny*nx*ng))

                    # Reshape
                    dc_linear.shape      = (nz, ny, nx, ng)
                    dc_nonlinear.shape = (nz, ny, nx, ng)

                    # Extract
                    ind_diag     = sum([range(i*nz/nza      , (i+1)*nz/nza-1) for i in range(nza)], [])
                    ind_off_diag = sum([range((i+1)*nz/nza-1, (i+1)*nz/nza  ) for i in range(nza)], [])
                    dc_linear_diag          = dc_linear     [ind_diag    , :, :, :]
                    dc_linear_off_diag      = dc_linear     [ind_off_diag, :, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[ind_diag    , :, :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[ind_off_diag, :, :, :]

                    # Condense and add values to diag
                    if nz != nza:
                        dc_linear_diag          = openmc.kinetics.map_array(dc_linear_diag         , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag    , coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]      += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nza > 1:
                        dc_linear_off_diag      = openmc.kinetics.map_array(dc_linear_off_diag     , coarse_shape, normalize=False).flatten()[:-nya*nxa*ng]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[:-nya*nxa*ng]
                        dc_linear_data     .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags        .append(nya*nxa*ng)


        # Form a matrix of the surface diffusion coefficients corrections
        dc_linear_matrix = sps.diags(dc_linear_data, diags)
        dc_nonlinear_matrix = sps.diags(dc_nonlinear_data, diags)

        return dc_linear_matrix, dc_nonlinear_matrix

    def create_sources(self, settings_file):

        nz, ny, nx = self.shape_dimension
        dx, dy, dz = [i/j for i,j in zip(self.shape_mesh.width, self.shape_mesh.dimension)]
        ll = self.shape_mesh.lower_left
        ge = self.energy_groups.group_edges
        ng = self.ng
        nd = self.nd
        sources = []

        #state_fwd_in = self.states['FORWARD_IN']
        #state_pre_in = self.states['PREVIOUS_IN']
        #time_source = self.inverse_velocity * self.shape * \
        #    state_fwd_in.frequency_source(state_pre_in, False)
        #time_source.shape = (nz, ny, nx, ng)

        if self.multi_group:
            delayed_source = np.repeat(self.decay_rate * self.precursors, self.ng)
            delayed_source.shape = (self.nxyz, self.nd, self.ng)
            delayed_source = (self.chi_delayed * delayed_source).sum(axis=1)
            delayed_source.shape = (nz, ny, nx, ng)
        else:
            delayed_source = self.decay_rate * self.precursors
            delayed_source.shape = (nz, ny, nx, nd)

        library = openmc.data.DataLibrary.from_xml()
        u235_lib = library.get_by_material('U235')
        u235 = openmc.data.IncidentNeutron.from_hdf5(u235_lib['path'])

        # Create list of sources
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):

                    bounds = [ll[0] + x*dx, ll[1] + y*dy, ll[2] + z*dz,
                              ll[0] + (x+1)*dx, ll[1] + (y+1)*dy, ll[2] + (z+1)*dz]

                    if self.method in ['QUASI-STATIC', 'STATIC-FLUX']:

                        # Create delayed source
                        space = openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
                        if self.multi_group:
                            for g in range(ng):
                                if delayed_source[z,y,x,g] > 0.:
                                    energy = openmc.stats.Uniform(ge[ng-g-1], ge[ng-g])
                                    source = openmc.source.Source(space=space, energy=energy, strength=delayed_source[z,y,x,g])
                                    sources.append(source)
                        else:
                            for d in range(nd):
                                energy = u235.reactions[18].products[d+1].distribution[0].energy.energy_out[0]
                                source = openmc.source.Source(space=space, energy=energy, strength=delayed_source[z,y,x,d])
                                sources.append(source)


                    if self.method == 'QUASI-STATIC':

                        # Create time absorption source
                        for g in range(ng):
                            energy = openmc.stats.Uniform(ge[ng-g-1], ge[ng-g])
                            #if time_source[z,y,x,g] > 0.0:
                            #    source = openmc.source.Source(space=space, energy=energy, strength=time_source[z,y,x,g])
                            #    sources.append(source)

        settings_file.source = sources


class DerivedState(State):
    """State to store all the variables that describe a specific state of the system.

    Attributes
    ----------
    mesh : openmc.mesh.Mesh
        Mesh which specifies the dimensions of coarse mesh.

    unity_mesh : openmc.mesh.Mesh
        Mesh which specifies contains only one cell.

    pin_cell_mesh : openmc.mesh.Mesh
        Mesh over the pin cells.

    assembly_mesh : openmc.mesh.Mesh
        Mesh over the assemblies.

    one_group : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the a one-energy-group structure.

    energy_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the energy groups structure.

    fine_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups used to tally the transport cross section that will be
        condensed to get the diffusion coefficients in the coarse group
        structure.

    shape : np.ndarray
        Numpy array used to store the shape.

    amplitude : np.ndarray
        Numpy array used to store the amplitude.

    adjoint_flux : np.ndarray
        Numpy array used to store the adjoint flux.

    precursors : OrderedDict of np.ndarray
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

    states : dict of openmc.kinetics.State
        A dictionary of all states.

    forward_state : str
        The forward state to interpolate with.

    previous_state : str
        The previous state to interpolate with.

    """

    def __init__(self, states):
        super(DerivedState, self).__init__()

        # Initialize Solver class attributes
        self.states = states
        self.forward_state = 'FORWARD_OUT'
        self.previous_state = 'PREVIOUS_OUT'

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        clone._shape_mesh = self._shape_mesh
        clone._amplitude_mesh = self._amplitude_mesh
        clone._unity_mesh = self._unity_mesh
        clone._one_group = self.one_group
        clone._energy_groups = self.energy_groups
        clone._shape = copy.deepcopy(self._shape)
        clone._amplitude = copy.deepcopy(self._amplitude)
        clone._adjoint_flux = copy.deepcopy(self._adjoint_flux)
        clone._precursors = copy.deepcopy(self._precursors)
        clone._mgxs_lib = copy.deepcopy(self.mgxs_lib)
        clone._k_crit = self.k_crit
        clone._chi_delayed_by_delayed_group = self._chi_delayed_by_delayed_group
        clone._chi_delayed_by_mesh = self._chi_delayed_by_mesh
        clone._num_delayed_groups = self.num_delayed_groups
        clone._time_point = self.time_point
        clone._clock = self.clock
        clone._core_volume = self.core_volume
        clone._log_file = self._log_file
        clone._multi_group = self._multi_group
        clone._states = self.states
        clone._forward_state = self.forward_state
        clone._previous_state = self.previous_state

        return clone

    @property
    def states(self):
        return self._states

    @property
    def forward_state(self):
        return self._forward_state

    @property
    def previous_state(self):
        return self._previous_state

    @states.setter
    def states(self, states):
        self._states = states

    @forward_state.setter
    def forward_state(self, forward_state):
        self._forward_state = forward_state

    @previous_state.setter
    def previous_state(self, previous_state):
        self._previous_state = previous_state

    @property
    def weight(self):
        time_point = self.clock.times[self.time_point]
        fwd_time = self.clock.times[self.forward_state]
        weight = 1 - (fwd_time - time_point) / self.clock.dt_outer
        return weight

    @property
    def inscatter(self):
        wgt = self.weight
        inscatter_fwd  = self.states[self.forward_state].inscatter
        inscatter_prev = self.states[self.previous_state].inscatter
        inscatter = inscatter_fwd * wgt + inscatter_prev * (1 - wgt)
        inscatter[inscatter < 0.] = 0.
        return inscatter

    @property
    def absorption(self):
        wgt = self.weight
        absorption_fwd  = self.states[self.forward_state].absorption
        absorption_prev = self.states[self.previous_state].absorption
        absorption = absorption_fwd * wgt + absorption_prev * (1 - wgt)
        absorption[absorption < 0.] = 0.
        return absorption


    def destruction_matrix(self, collapse=True, omega=False):
        wgt = self.weight
        state_fwd = self.states[self.forward_state]
        state_prev = self.states[self.previous_state]
        linear_fwd,  non_linear_fwd  = state_fwd.coupling(collapse)
        linear_prev, non_linear_prev = state_prev.coupling(collapse)
        linear = linear_fwd * wgt + linear_prev * (1 - wgt)
        non_linear = non_linear_fwd * wgt + non_linear_prev * (1 - wgt)
        inscatter       = self.inscatter.flatten() * self.dxyz
        absorb_outscat  = self.outscatter.flatten() + self.absorption.flatten()
        absorb_outscat *= self.dxyz

        if collapse:
            inscatter      *= np.tile(self.shape, self.ng).flatten()
            absorb_outscat *= self.shape.flatten()
            inscatter.shape      = self.shape_dimension + (self.ng, self.ng)
            absorb_outscat.shape = self.shape_dimension + (self.ng,)
            coarse_shape   = self.amplitude_dimension + (self.ng, self.ng)
            inscatter      = openmc.kinetics.map_array(inscatter, coarse_shape, normalize=False)
            coarse_shape   = self.amplitude_dimension + (self.ng,)
            absorb_outscat = openmc.kinetics.map_array(absorb_outscat, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            inscatter.shape = (nxyz, self.ng, self.ng)
        else:
            inscatter.shape = (self.nxyz, self.ng, self.ng)

        total = sps.diags(absorb_outscat.flatten()) - openmc.kinetics.diagonal_matrix(inscatter)
        return total + linear + non_linear

    @property
    def chi_prompt(self):
        wgt = self.weight
        chi_prompt_fwd  = self.states[self.forward_state].chi_prompt
        chi_prompt_prev = self.states[self.previous_state].chi_prompt
        chi_prompt = chi_prompt_fwd * wgt + chi_prompt_prev * (1 - wgt)
        chi_prompt[chi_prompt < 0.] = 0.
        return chi_prompt

    @property
    def prompt_nu_fission(self):
        wgt = self.weight
        prompt_nu_fission_fwd  = self.states[self.forward_state].prompt_nu_fission
        prompt_nu_fission_prev = self.states[self.previous_state].prompt_nu_fission
        prompt_nu_fission = prompt_nu_fission_fwd * wgt + prompt_nu_fission_prev * (1 - wgt)
        prompt_nu_fission[prompt_nu_fission < 0.] = 0.
        return prompt_nu_fission

    @property
    def chi_delayed(self):
        wgt = self.weight
        chi_delayed_fwd  = self.states[self.forward_state].chi_delayed
        chi_delayed_prev = self.states[self.previous_state].chi_delayed
        chi_delayed = chi_delayed_fwd * wgt + chi_delayed_prev * (1 - wgt)
        chi_delayed[chi_delayed < 0.] = 0.
        return chi_delayed

    @property
    def delayed_nu_fission(self):
        wgt = self.weight
        delayed_nu_fission_fwd  = self.states[self.forward_state].delayed_nu_fission
        delayed_nu_fission_prev = self.states[self.previous_state].delayed_nu_fission
        delayed_nu_fission = delayed_nu_fission_fwd * wgt + delayed_nu_fission_prev * (1 - wgt)
        delayed_nu_fission[delayed_nu_fission < 0.] = 0.
        return delayed_nu_fission

    @property
    def decay_rate(self):
        wgt = self.weight
        decay_rate_fwd  = self.states[self.forward_state].decay_rate
        decay_rate_prev = self.states[self.previous_state].decay_rate
        decay_rate = decay_rate_fwd * wgt + decay_rate_prev * (1 - wgt)
        decay_rate[decay_rate < 0.] = 0.
        decay_rate[decay_rate < 1.e-5] = 0.
        return decay_rate

    @property
    def inverse_velocity(self):
        wgt = self.weight
        inverse_velocity_fwd  = self.states[self.forward_state].inverse_velocity
        inverse_velocity_prev = self.states[self.previous_state].inverse_velocity
        inverse_velocity = inverse_velocity_fwd * wgt + inverse_velocity_prev * (1 - wgt)
        inverse_velocity[inverse_velocity < 0.] = 0.
        return inverse_velocity

    @property
    def diffusion_coefficients(self):
        wgt = self.weight
        diffusion_coefficients_fwd  = self.states[self.forward_state].diffusion_coefficients
        diffusion_coefficients_prev = self.states[self.previous_state].diffusion_coefficients
        diffusion_coefficients = diffusion_coefficients_fwd * wgt + diffusion_coefficients_prev * (1 - wgt)
        diffusion_coefficients[diffusion_coefficients < 0.] = 0.
        return diffusion_coefficients

    @property
    def shape(self):
        wgt = self.weight
        shape_fwd  = self.states[self.forward_state].shape
        shape_prev = self.states[self.previous_state].shape
        shape = shape_fwd * wgt + shape_prev * (1 - wgt)
        shape[shape < 0.] = 0.
        return shape

    @property
    def kappa_fission(self):
        wgt = self.weight
        kappa_fission_fwd  = self.states[self.forward_state].kappa_fission
        kappa_fission_prev = self.states[self.previous_state].kappa_fission
        kappa_fission = kappa_fission_fwd * wgt + kappa_fission_prev * (1 - wgt)
        kappa_fission[kappa_fission < 0.] = 0.
        return kappa_fission

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Create elements and ordered dicts and initialize to None
        self._mgxs_lib = None

    def coupling(self, check_for_diag_dominance=False):

        msg = 'Cannot compute the surface diffusion coefficients ' \
              'for a DerivedState'
        raise ValueError(msg)

    def transient_matrix(self, collapse=True):
        return self.time_removal_matrix(collapse, True) + self.shape_deriv_matrix(collapse) \
            + (- self.prompt_production_matrix(collapse) + self.destruction_matrix(collapse) \
                     - self.k2_source_matrix)

    def shape_deriv_matrix(self, collapse=True):
        state_fwd = self.states[self.forward_state]
        state_prev = self.states[self.previous_state]
        shape_deriv = (state_fwd.shape - state_prev.shape) / self.dt_outer
        shape_deriv.shape = (self.nxyz, self.ng)
        shape_deriv *= self.inverse_velocity * self.dxyz

        if collapse:
            shape_deriv.shape = self.shape_dimension + (self.ng,)
            coarse_shape = self.amplitude_dimension + (self.ng,)
            shape_deriv = openmc.kinetics.map_array(shape_deriv, coarse_shape, normalize=False)

        return sps.diags(shape_deriv.flatten(), 0)

    @property
    def k1(self):
        return np.exp(- self.dt_inner * self.decay_rate)

    @property
    def k2(self):

        # Compute k2 / (lambda * k_crit)
        k2 = 1. - (1. - self.k1) / (self.dt_inner * self.decay_rate)
        k2 /= self.decay_rate * self.k_crit

        # Convert -inf, inf, and nan to zero
        k2[k2 == -np.inf] = 0.
        k2[k2 ==  np.inf] = 0.
        k2 = np.nan_to_num(k2)

        return k2

    @property
    def k3(self):
        k3 = self.k1 - (1. - self.k1) / (self.dt_inner * self.decay_rate)
        k3 /= self.decay_rate * self.k_crit

        # Convert -inf, inf, and nan to zero
        k3[k3 == -np.inf] = 0.
        k3[k3 ==  np.inf] = 0.
        k3 = np.nan_to_num(k3)

        return k3

    def k1_source(self, state, collapse=True):

        source = np.repeat(self.decay_rate * state.k1 * state.precursors, self.ng)
        source.shape = (self.nxyz, self.nd, self.ng)
        source *= self.chi_delayed
        source = source.sum(axis=1)

        if collapse:
            source.shape = self.shape_dimension     + (self.ng,)
            coarse_shape = self.amplitude_dimension + (self.ng,)
            source = openmc.kinetics.map_array(source, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            source.shape = (nxyz, self.ng)

        return source

    @property
    def k2_source_matrix(self):

        k2 = np.repeat(self.decay_rate * self.k2, self.ng * self.ng)
        k2.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        chi = np.repeat(self.chi_delayed, self.ng)
        chi.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        shape = np.tile(self.shape, self.nd).flatten()
        del_fis_rate = (self.delayed_nu_fission.flatten() * shape).reshape((self.nxyz, self.nd, self.ng))
        del_fis_rate = np.tile(del_fis_rate, self.ng)
        del_fis_rate.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        term_k2 = chi * k2 * del_fis_rate * self.dxyz
        coarse_shape = self.amplitude_dimension + (self.nd, self.ng, self.ng)
        term_k2 = openmc.kinetics.map_array(term_k2, coarse_shape, normalize=False)
        nxyz = np.prod(self.amplitude_mesh.dimension)
        term_k2.shape = (nxyz, self.nd, self.ng, self.ng)

        return openmc.kinetics.diagonal_matrix(term_k2.sum(axis=1))

    def k3_source_matrix(self, state):

        k3 = np.repeat(self.decay_rate * state.k3, self.ng * self.ng)
        k3.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        chi = np.repeat(self.chi_delayed, self.ng)
        chi.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        shape = np.tile(self.shape, self.nd).flatten()
        del_fis_rate = (self.delayed_nu_fission.flatten() * shape).reshape((self.nxyz, self.nd, self.ng))
        del_fis_rate = np.tile(del_fis_rate, self.ng)
        del_fis_rate.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        term_k3 = chi * k3 * del_fis_rate * self.dxyz
        coarse_shape = self.amplitude_dimension + (self.nd, self.ng, self.ng)
        term_k3 = openmc.kinetics.map_array(term_k3, coarse_shape, normalize=False)
        nxyz = np.prod(self.amplitude_mesh.dimension)
        term_k3.shape = (nxyz, self.nd, self.ng, self.ng)

        return openmc.kinetics.diagonal_matrix(term_k3.sum(axis=1))

    def propagate_precursors(self, state):

        # Contribution from current precursors
        term_k1 = state.k1 * state.precursors

        # Contribution from generation at current time point
        flux = np.tile(self.flux, self.nd)
        flux.shape = (self.nxyz, self.nd, self.ng)
        term_k2 = self.k2 * (self.delayed_nu_fission * flux).sum(axis=2) * self.dxyz

        # Contribution from generation at previous time step
        flux = np.tile(state.flux, state.nd)
        flux.shape = (state.nxyz, state.nd, state.ng)
        term_k3 = state.k3 * (state.delayed_nu_fission * flux).sum(axis=2) * self.dxyz

        self._precursors = term_k1 + term_k2 - term_k3

    @property
    def frequency_source(self, state_prev, collapse=True):

        # 1 / amp * (d amp / dt) * dxyz

        amp_deriv = (self.amplitude - state_prev.amplitude) / self.dt_inner
        fine_shape = self.shape_dimension + (self.ng,)
        amp_deriv.shape = self.amplitude_dimension + (self.ng,)
        fine_amp_deriv = openmc.kinetics.map_array(amp_deriv, fine_shape, normalize=True)
        fine_amp_deriv.shape = (self.nxyz, self.ng)
        frequency = fine_amp_deriv / self.amplitude * self.dxyz

        if collapse:
            coarse_shape = self.amplitude_dimension + (self.ng,)
            frequency.shape = fine_shape
            frequency = openmc.kinetics.map_array(frequency, coarse_shape, normalize=False)
            nxyz = np.prod(self.amplitude_mesh.dimension)
            frequency.shape = (nxyz, self.ng)

        return frequency
