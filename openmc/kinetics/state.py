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

    flux : OrderedDict of np.ndarray
        Numpy array used to store the flux.

    adjoint_flux : OrderedDict of np.ndarray
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

    nxyz : int
        The number of mesh cells.

    dxyz : float
        The volume of a mesh cell.

    num_delayed_groups : int
        The number of delayed neutron precursor groups.

    nd : int
        The number of delayed neutron precursor groups.

    ng : int
        The number of energy groups.

    time_point : str
        The time point of this state.

    clock : openmc.kinetics.Clock
        A clock object to indicate the current simulation time.

    """

    def __init__(self):

        # Initialize Solver class attributes
        self._mesh = None
        self._unity_mesh = None
        self._pin_cell_mesh = None
        self._assembly_mesh = None
        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._flux = None
        self._adjoint_flux = None
        self._precursors = None
        self._mgxs_lib = None
        self._k_crit = 1.0
        self._num_delayed_groups = 6
        self._time_point = None
        self._clock = None
        self._core_volume = 1.
        self._chi_delayed_by_delayed_group = False
        self._chi_delayed_by_mesh = False
        self._log_file = None

    def __deepcopy__(self, memo):

        clone = type(self).__new__(type(self))
        clone._mesh = self.mesh
        clone._unity_mesh = self.unity_mesh
        clone._pin_cell_mesh = self.pin_cell_mesh
        clone._assembly_mesh = self.assembly_mesh
        clone._one_group = self.one_group
        clone._energy_groups = self.energy_groups
        clone._flux = copy.deepcopy(self._flux)
        clone._adjoint_flux = copy.deepcopy(self._adjoint_flux)
        clone._precursors = copy.deepcopy(self.precursors)
        clone._mgxs_lib = copy.deepcopy(self.mgxs_lib)
        clone._k_crit = self.k_crit
        clone._num_delayed_groups = self.num_delayed_groups
        clone._clock = self.clock
        clone._time_point = self.time_point
        clone._core_volume = self.core_volume
        clone._chi_delayed_by_delayed_group = self._chi_delayed_by_delayed_group
        clone._chi_delayed_by_mesh = self._chi_delayed_by_mesh
        clone._log_file = self._log_file

        return clone

    @property
    def log_file(self):
        return self._log_file

    @property
    def core_volume(self):
        return self._core_volume

    @property
    def time_point(self):
        return self._time_point

    @property
    def clock(self):
        return self._clock

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
        return self._flux

    @property
    def adjoint_flux(self):
        return self._adjoint_flux

    @property
    def precursors(self):
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
    def nxyz(self):
        return np.prod(self.mesh.dimension)

    @property
    def dxyz(self):
        return np.prod(self.mesh.width)

    @property
    def ng(self):
        return self.energy_groups.num_groups

    @property
    def ngp(self):
        return self.energy_groups.num_groups

    @property
    def nd(self):
        return self.num_delayed_groups

    @property
    def dt(self):
        return self.clock.dt_inner

    @log_file.setter
    def log_file(self, log_file):
        self._log_file = log_file

    @core_volume.setter
    def core_volume(self, core_volume):
        self._core_volume = core_volume

    @time_point.setter
    def time_point(self, time_point):
        self._time_point = time_point

    @clock.setter
    def clock(self, clock):
        self._clock = clock

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

    @unity_mesh.setter
    def unity_mesh(self, unity_mesh):
        self._unity_mesh = unity_mesh

    @pin_cell_mesh.setter
    def pin_cell_mesh(self, pin_cell_mesh):
        self._pin_cell_mesh = pin_cell_mesh

    @assembly_mesh.setter
    def assembly_mesh(self, assembly_mesh):
        self._assembly_mesh = assembly_mesh

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
        self._flux.shape = (self.nxyz, self.ng)

    @adjoint_flux.setter
    def adjoint_flux(self, adjoint_flux):
        self._adjoint_flux = copy.deepcopy(adjoint_flux)
        self._adjoint_flux.shape = (self.nxyz, self.ng)

    @precursors.setter
    def precursors(self, precursors):
        self._precursors = precursors

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

    @property
    def reactivity(self):
        production = self.production_matrix * self.flux.flatten()
        destruction = self.destruction_matrix * self.flux.flatten()
        balance = production - destruction
        balance = balance * self.adjoint_flux.flatten()
        production = production * self.adjoint_flux.flatten()
        return balance.sum() / production.sum()

    def get_diagonal_matrix(self, array):

        ni, ng, ng = array.shape
        diags = np.zeros((ng*2-1, ni * ng))
        ndiag = [0] + [g for g in range(1,ng)] + [-g for g in range(1,ng)]

        for i in range(ni):
            for r in range(ng):
                for c in range(ng):
                    diags[c-r][i*ng+r] = array[i, r, c]

        diags2 = [diags[0]]
        for g in range(1,ng):
            diags2.append(diags[g][:-g])
        for g in range(1,ng):
            diags2.append(diags[-g][g:])

        return sps.diags(diags2, ndiag)

    @property
    def beta_eff(self):
        flux = np.tile(self.flux.flatten(), self.nd)
        adjoint_flux = np.tile(self.adjoint_flux.flatten(), self.nd)
        delayed_production = self.delayed_production
        delayed_production.shape = (self.nxyz * self.nd, self.ng, self.ng)

        delayed_production = self.get_diagonal_matrix(delayed_production)
        delayed_production *= self.dxyz / self.k_crit

        delayed_production = delayed_production * flux
        delayed_production = delayed_production * adjoint_flux
        delayed_production.shape = (self.nxyz, self.nd, self.ng)
        delayed_production = delayed_production.sum(axis=2).sum(axis=0)

        production = self.production_matrix * self.flux.flatten()
        production = production * self.adjoint_flux.flatten()
        production.shape = (self.nxyz, self.ng)
        production = production.sum(axis=1).sum(axis=0)
        production = np.tile(production, self.nd)

        return (delayed_production / production).sum()

    @property
    def beta(self):
        flux = np.tile(self.flux.flatten(), self.nd)
        delayed_production = self.delayed_production
        delayed_production.shape = (self.nxyz * self.nd, self.ng, self.ng)

        delayed_production = self.get_diagonal_matrix(delayed_production)
        delayed_production *= self.dxyz / self.k_crit

        delayed_production = delayed_production * flux
        delayed_production.shape = (self.nxyz, self.nd, self.ng)
        delayed_production = delayed_production.sum(axis=2).sum(axis=0)

        production = self.production_matrix * self.flux.flatten()
        production.shape = (self.nxyz, self.ng)
        production = production.sum(axis=1).sum(axis=0)
        production = np.tile(production, self.nd)

        return (delayed_production / production).sum()

    @property
    def pnl(self):
        inv_velocity = self.dxyz * self.adjoint_flux * self.inverse_velocity * self.flux
        production = self.production_matrix * self.flux.flatten()
        production = production * self.adjoint_flux.flatten()
        return inv_velocity.sum() / production.sum()

    @property
    def inscatter(self):
        inscatter = self.mgxs_lib['consistent nu-scatter matrix'].get_xs(row_column='outin')
        inscatter.shape = (self.nxyz, self.ngp, self.ng)
        return inscatter

    @property
    def outscatter(self):
        return self.inscatter.sum(axis=1)

    @property
    def absorption(self):
        absorption = self.mgxs_lib['absorption'].get_xs()
        absorption.shape = (self.nxyz, self.ng)
        return absorption

    @property
    def destruction_matrix(self):
        stream, stream_corr = self.compute_surface_dif_coefs()
        inscatter = self.get_diagonal_matrix(self.inscatter)
        outscatter = sps.diags(self.outscatter.flatten(), 0)
        absorb = sps.diags(self.absorption.flatten(), 0)

        return self.dxyz * (absorb + outscatter - inscatter) + \
            stream + stream_corr

    @property
    def adjoint_destruction_matrix(self):
        stream, stream_corr = self.compute_surface_dif_coefs()
        inscatter = self.get_diagonal_matrix(self.inscatter)
        outscatter = sps.diags(self.outscatter.flatten(), 0)
        absorb = sps.diags(self.absorption.flatten(), 0)
        matrix = self.dxyz * (absorb + outscatter - inscatter)

        return matrix.transpose() + stream.transpose() + stream_corr.transpose()

    @property
    def chi_prompt(self):
        chi_prompt = self.mgxs_lib['chi-prompt'].get_xs()
        chi_prompt.shape = (self.nxyz, self.ngp)
        return chi_prompt

    @property
    def prompt_nu_fission(self):
        prompt_nu_fission = self.mgxs_lib['prompt-nu-fission'].get_xs()
        prompt_nu_fission.shape = (self.nxyz, self.ng)
        return prompt_nu_fission

    @property
    def chi_delayed(self):
        chi_delayed = self.mgxs_lib['chi-delayed'].get_xs()
        if not self.chi_delayed_by_mesh:
            chi_delayed = np.tile(chi_delayed, self.nxyz)
        if not self.chi_delayed_by_delayed_group:
            chi_delayed = np.tile(chi_delayed, self.nd)
        chi_delayed.shape = (self.nxyz, self.nd, self.ngp)
        return chi_delayed

    @property
    def delayed_nu_fission(self):
        delayed_nu_fission = self.mgxs_lib['delayed-nu-fission'].get_xs()
        delayed_nu_fission.shape = (self.nxyz, self.nd, self.ng)
        return delayed_nu_fission

    @property
    def delayed_production(self):
        chi_delayed = np.repeat(self.chi_delayed, self.ng)
        chi_delayed.shape = (self.nxyz, self.nd, self.ngp, self.ng)
        delayed_nu_fission = np.tile(self.delayed_nu_fission, self.ngp)
        delayed_nu_fission.shape = (self.nxyz, self.nd, self.ngp, self.ng)
        return (chi_delayed * delayed_nu_fission)

    @property
    def delayed_production_matrix(self):
        delayed_production = self.get_diagonal_matrix(self.delayed_production.sum(axis=1))
        return delayed_production * self.dxyz / self.k_crit

    @property
    def production_matrix(self):
        return self.prompt_production_matrix + self.delayed_production_matrix

    @property
    def prompt_production(self):
        chi_prompt = np.repeat(self.chi_prompt, self.ng)
        chi_prompt.shape = (self.nxyz, self.ngp, self.ng)
        prompt_nu_fission = np.tile(self.prompt_nu_fission, self.ngp)
        prompt_nu_fission.shape = (self.nxyz, self.ngp, self.ng)
        return (chi_prompt * prompt_nu_fission)

    @property
    def prompt_production_matrix(self):
        prompt_production = self.get_diagonal_matrix(self.prompt_production)
        return prompt_production * self.dxyz / self.k_crit

    @property
    def kappa_fission(self):
        kappa_fission = self.mgxs_lib['kappa-fission'].get_xs()
        kappa_fission.shape = (self.nxyz, self.ng)
        return kappa_fission

    @property
    def pin_cell_kappa_fission(self):
        kappa_fission = self.mgxs_lib['PIN-CELL kappa-fission'].get_xs()
        kappa_fission.shape = (np.prod(self.pin_cell_mesh.dimension), self.ng)
        return kappa_fission

    @property
    def assembly_kappa_fission(self):
        kappa_fission = self.mgxs_lib['ASSEMBLY kappa-fission'].get_xs()
        kappa_fission.shape = (np.prod(self.assembly_mesh.dimension), self.ng)
        return kappa_fission

    @property
    def pin_cell_shape(self):
        shape = self.mgxs_lib['PIN-CELL kappa-fission'].tallies['flux'].get_values()
        shape.shape = (np.prod(self.pin_cell_mesh.dimension), self.ng)
        shape = shape[:, ::-1]
        return shape

    @property
    def assembly_shape(self):
        shape = self.mgxs_lib['ASSEMBLY kappa-fission'].tallies['flux'].get_values()
        shape.shape = (np.prod(self.assembly_mesh.dimension), self.ng)
        shape = shape[:, ::-1]
        return shape

    @property
    def core_power_density(self):
        return self.mesh_powers.sum() / self.core_volume

    @property
    def mesh_powers(self):
        return self.dxyz * (self.kappa_fission * self.flux).sum(axis=1)

    @property
    def pin_powers(self):

        # Get the pin mesh shape
        pin_powers = (self.pin_cell_kappa_fission * self.pin_cell_shape).sum(axis=1)
        pin_powers.shape = self.pin_cell_mesh.dimension[::-1]

        # Get the finest mesh shape
        finest_mesh_shape = [max(i,j) for i,j in zip(self.pin_cell_mesh.dimension[::-1], self.mesh.dimension[::-1])]
        pin_shape_ratio = [i/j for i,j in zip(finest_mesh_shape,self.pin_cell_mesh.dimension[::-1])]

        # Get the finest mesh powers
        finest_pin_powers = np.copy(pin_powers)
        for i in range(3):
            finest_pin_powers = finest_pin_powers.repeat(pin_shape_ratio[i], axis=i)

        finest_pin_powers.shape = finest_mesh_shape
        finest_pin_ratios = np.zeros(finest_mesh_shape)

        # Get the ratio of the finest_powers to the pin_powers
        for k in range(finest_mesh_shape[0]):
            for j in range(finest_mesh_shape[1]):
                for i in range(finest_mesh_shape[2]):
                    finest_pin_ratios[k,j,i] = finest_pin_powers[k,j,i] / \
                        pin_powers[k/pin_shape_ratio[0], j/pin_shape_ratio[1], i/pin_shape_ratio[2]] / \
                        np.prod(pin_shape_ratio)

        # Get the power density on the finest mesh
        mesh_powers = self.mesh_powers
        mesh_powers.shape = self.mesh.dimension[::-1]
        mesh_shape_ratio = [i/j for i,j in zip(finest_mesh_shape,self.mesh.dimension[::-1])]
        finest_mesh_powers = np.copy(mesh_powers)
        for i in range(3):
            finest_mesh_powers = finest_mesh_powers.repeat(mesh_shape_ratio[i], axis=i)

        finest_mesh_powers.shape = finest_mesh_shape

        # Get the coarse mesh powers
        pin_powers = np.zeros(self.pin_cell_mesh.dimension[::-1])
        for i in range(finest_mesh_shape[2]):
            for j in range(finest_mesh_shape[1]):
                for k in range(finest_mesh_shape[0]):
                    pin_powers[k/pin_shape_ratio[0], j/pin_shape_ratio[1], i/pin_shape_ratio[2]] += \
                        finest_pin_ratios[k,j,i] * finest_mesh_powers[k,j,i]

        pin_powers = np.nan_to_num(pin_powers)

        return pin_powers

    @property
    def assembly_powers(self):

        # Get the assembly mesh shape
        assembly_powers = (self.assembly_kappa_fission * self.assembly_shape).sum(axis=1)
        assembly_powers.shape = self.assembly_mesh.dimension[::-1]

        # Get the finest mesh shape
        finest_mesh_shape = [max(i,j) for i,j in zip(self.assembly_mesh.dimension[::-1], self.mesh.dimension[::-1])]
        assembly_shape_ratio = [i/j for i,j in zip(finest_mesh_shape,self.assembly_mesh.dimension[::-1])]

        # Get the finest mesh powers
        finest_assembly_powers = assembly_powers
        for i in range(3):
            finest_assembly_powers = finest_assembly_powers.repeat(assembly_shape_ratio[i], axis=i)

        finest_assembly_powers.shape = finest_mesh_shape
        finest_assembly_ratios = np.zeros(finest_mesh_shape)

        # Get the ratio of the finest_powers to the assembly_powers
        for i in range(finest_mesh_shape[2]):
            for j in range(finest_mesh_shape[1]):
                for k in range(finest_mesh_shape[0]):
                    finest_assembly_ratios[k,j,i] = finest_assembly_powers[k,j,i] / \
                        assembly_powers[k/assembly_shape_ratio[0], j/assembly_shape_ratio[1], i/assembly_shape_ratio[2]] / \
                        np.prod(assembly_shape_ratio)

        # Get the power density on the finest mesh
        mesh_powers = self.mesh_powers
        mesh_powers.shape = self.mesh.dimension[::-1]
        mesh_shape_ratio = [i/j for i,j in zip(finest_mesh_shape,self.mesh.dimension[::-1])]
        finest_mesh_powers = mesh_powers
        for i in range(3):
            finest_mesh_powers = finest_mesh_powers.repeat(mesh_shape_ratio[i], axis=i)

        finest_mesh_powers.shape = finest_mesh_shape

        # Get the coarse mesh powers
        assembly_powers = np.zeros(self.assembly_mesh.dimension[::-1])
        for i in range(finest_mesh_shape[2]):
            for j in range(finest_mesh_shape[1]):
                for k in range(finest_mesh_shape[0]):
                    assembly_powers[k/assembly_shape_ratio[0], j/assembly_shape_ratio[1], i/assembly_shape_ratio[2]] += \
                        finest_assembly_ratios[k,j,i] * finest_mesh_powers[k,j,i]

        return assembly_powers

    @property
    def time_source_matrix(self):
        time_source = self.dxyz * self.inverse_velocity / self.dt
        return sps.diags(time_source.flatten(), 0)

    def decay_source(self, state):
        decay_source = self.decay_rate * state.precursors
        decay_source /= 1. + self.decay_rate * self.dt
        decay_source = np.repeat(decay_source, self.ngp)
        decay_source.shape = (self.nxyz, self.nd, self.ngp)
        return self.dxyz * (self.chi_delayed * decay_source).sum(axis=1)

    @property
    def decay_rate(self):
        decay_rate = self.mgxs_lib['decay-rate'].get_xs()
        decay_rate.shape = (self.nxyz, self.nd)
        return decay_rate

    @property
    def inverse_velocity(self):
        inverse_velocity = self.mgxs_lib['inverse-velocity'].get_xs()
        inverse_velocity.shape = (self.nxyz, self.ng)
        return inverse_velocity

    @property
    def transient_matrix(self):
        return self.time_source_matrix - self.prompt_production_matrix \
            + self.destruction_matrix - self.decay_production_matrix

    @property
    def decay_production_matrix(self):
        decay_term = self.decay_rate * self.dt / \
            (1. + self.dt * self.decay_rate) / self.k_crit
        decay_term = np.repeat(decay_term, self.ngp * self.ng)
        decay_term.shape = (self.nxyz, self.nd, self.ngp, self.ng)
        decay_term *= self.dxyz * self.delayed_production
        return self.get_diagonal_matrix(decay_term.sum(axis=1))

    @property
    def delayed_fission_rate(self):
        flux = np.tile(self.flux, self.nd)
        flux.shape = (self.nxyz, self.nd, self.ng)
        return (self.delayed_nu_fission * flux).sum(axis=2)

    def dump_inner_to_log_file(self):

        time_point = str(self.clock.times[self.time_point])
        f = h5py.File(self._log_file, 'a')
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
        f['time_steps'].require_group(time_point)
        f['time_steps'][time_point]['dump_type'] = 'outer'
        f['time_steps'][time_point]['flux'] = self.flux
        f['time_steps'][time_point]['current_balance'] = self.current_balance
        f['time_steps'][time_point]['current_computed'] = self.current_computed
        f['time_steps'][time_point]['adjoint_flux'] = self.adjoint_flux
        f['time_steps'][time_point]['precursors'] = self.precursors
        f['time_steps'][time_point].attrs['reactivity'] = self.reactivity
        f['time_steps'][time_point].attrs['beta_eff'] = self.beta_eff
        f['time_steps'][time_point].attrs['pnl'] = self.pnl
        f['time_steps'][time_point].attrs['core_power_density'] = self.core_power_density
        f['time_steps'][time_point]['kappa_fission'] = self.kappa_fission
        f['time_steps'][time_point]['pin_cell_kappa_fission'] = self.pin_cell_kappa_fission
        f['time_steps'][time_point]['assembly_kappa_fission'] = self.assembly_kappa_fission
        f['time_steps'][time_point]['pin_cell_shape'] = self.pin_cell_shape
        f['time_steps'][time_point]['assembly_shape'] = self.assembly_shape
        f['time_steps'][time_point]['pin_powers'] = self.pin_powers
        f['time_steps'][time_point]['assembly_powers'] = self.assembly_powers
        f.close()

    def propagate_precursors(self, state):

        # Get the flux and repeat to cover all delayed groups
        self.precursors = (self.delayed_fission_rate * self.dt / self.k_crit
                           + state.precursors) / (1 + self.dt * self.decay_rate)

    def compute_initial_precursor_concentration(self):
        self.precursors = self.delayed_fission_rate / self.decay_rate / self.k_crit
        self.precursors = np.nan_to_num(self.precursors)
        self.precursors[self.precursors == np.inf] = 0.

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Instantiate a list of the delayed groups
        delayed_groups = list(range(1,self.num_delayed_groups + 1))

        # Create elements and ordered dicts and initialize to None
        self._mgxs_lib = OrderedDict()

        # Create kappa-fission MGXS objects for the pin cell and assembly meshes
        self._mgxs_lib['PIN-CELL kappa-fission'] = openmc.mgxs.MGXS.get_mgxs(
            'kappa-fission', domain=self.pin_cell_mesh, domain_type='mesh',
            energy_groups=self.energy_groups, by_nuclide=False,
            name= self.time_point + ' - PIN-CELL - kappa-fission')

        self._mgxs_lib['ASSEMBLY kappa-fission'] = openmc.mgxs.MGXS.get_mgxs(
            'kappa-fission', domain=self.assembly_mesh, domain_type='mesh',
            energy_groups=self.energy_groups, by_nuclide=False,
            name= self.time_point + ' - ASSEMBLY - kappa-fission')

        mgxs_types = ['absorption', 'diffusion-coefficient', 'decay-rate',
                      'kappa-fission', 'consistent nu-scatter matrix', 'chi-prompt',
                      'chi-delayed', 'inverse-velocity', 'prompt-nu-fission',
                      'current', 'delayed-nu-fission']

        # Populate the MGXS in the MGXS lib
        for mgxs_type in mgxs_types:
            if mgxs_type == 'diffusion-coefficient':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.fine_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type == 'consistent nu-scatter matrix':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.energy_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
                self._mgxs_lib[mgxs_type].correction = None
            elif mgxs_type == 'decay-rate':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.one_group,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type in openmc.mgxs.MGXS_TYPES:
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.energy_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type == 'chi-delayed':
                if self.chi_delayed_by_delayed_group:
                    if self.chi_delayed_by_mesh:
                        self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                            mgxs_type, domain=self.mesh, domain_type='mesh',
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
                            mgxs_type, domain=self.mesh, domain_type='mesh',
                            energy_groups=self.energy_groups, by_nuclide=False,
                            name= self.time_point + ' - ' + mgxs_type)
                    else:
                        self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                            mgxs_type, domain=self.unity_mesh, domain_type='mesh',
                            energy_groups=self.energy_groups, by_nuclide=False,
                            name= self.time_point + ' - ' + mgxs_type)
            elif mgxs_type in openmc.mgxs.MDGXS_TYPES:
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.energy_groups,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name= self.time_point + ' - ' + mgxs_type)

    @property
    def diffusion_coefficients(self):
        dc_mgxs = self.mgxs_lib['diffusion-coefficient']
        dc = dc_mgxs.get_condensed_xs(self.energy_groups).get_xs()
        dc.shape = (self.nxyz, self.ng)
        return dc

    @property
    def current_balance(self):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        ng = self.ng

        # Get the array of the surface-integrated surface net currents
        partial_current = self.mgxs_lib['current'].get_xs()
        partial_current = partial_current.reshape(np.prod(partial_current.shape) / 12, 12)
        net_current = partial_current[:, range(0,12,2)] - partial_current[:, range(1,13,2)]
        net_current = net_current.sum(axis=1)
        net_current.shape = (nz*ny*nx, ng)
        return net_current

    @property
    def current_computed(self):

        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.ng

        stream, stream_corr = self.compute_surface_dif_coefs()
        flux = self.mgxs_lib['absorption'].tallies['flux'].get_values()
        flux.shape = (nz, ny, nx, ng)
        flux = flux[:, :, :, ::-1]
        flux /= (dx * dy * dz)
        current = (stream + stream_corr) * flux.flatten()
        current.shape = (nx*ny*nz, ng)
        return current

    def compute_surface_dif_coefs(self, check_for_diag_dominance=False):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.ng

        # Get the array of the surface-integrated surface net currents
        partial_current = self.mgxs_lib['current'].get_xs()
        partial_current = partial_current.reshape(np.prod(partial_current.shape) / 12, 12)
        net_current = partial_current[:, range(0,12,2)] - partial_current[:, range(1,13,2)]
        net_current[:, 0:6:2] = -net_current[:, 0:6:2]
        net_current.shape = (nz, ny, nx, ng, 6)

        # Convert from surface-integrated to surface-averaged net current
        net_current[..., 0:2] /= (dy * dz)
        net_current[..., 2:4] /= (dx * dz)
        net_current[..., 4:6] /= (dx * dy)

        # Get the flux
        flux = self.mgxs_lib['absorption'].tallies['flux'].get_values()
        flux.shape = (nz, ny, nx, ng)
        flux = flux[:, :, :, ::-1]

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

        # Get the diffusion coefficients tally
        dc_mgxs = self.mgxs_lib['diffusion-coefficient']
        dc = dc_mgxs.get_condensed_xs(self.energy_groups).get_xs()
        dc.shape = (nz, ny, nx, ng)
        dc_nbr = np.zeros((nz, ny, nx, ng, 6))

        # Create array of neighbor cell diffusion coefficients
        dc_nbr[:  , :  , 1: , :, 0] = dc[:  , :  , :-1, :]
        dc_nbr[:  , :  , :-1, :, 1] = dc[:  , :  , 1: , :]
        dc_nbr[:  , 1: , :  , :, 2] = dc[:  , :-1, :  , :]
        dc_nbr[:  , :-1, :  , :, 3] = dc[:  , 1: , :  , :]
        dc_nbr[1: , :  , :  , :, 4] = dc[:-1, :  , :  , :]
        dc_nbr[:-1, :  , :  , :, 5] = dc[1: , :  , :  , :]

        # Compute the surface diffusion coefficients for interior surfaces
        sdc = np.zeros((nz, ny, nx, ng, 6))
        sdc[:  , :  , 1: , :, 0] = 2 * dc_nbr[:  , :  , 1: , :, 0] * dc[:  , :  , 1: , :] / (dc_nbr[:  , :  , 1: , :, 0] * dx + dc[:  , :  , 1: , :] * dx)
        sdc[:  , :  , :-1, :, 1] = 2 * dc_nbr[:  , :  , :-1, :, 1] * dc[:  , :  , :-1, :] / (dc_nbr[:  , :  , :-1, :, 1] * dx + dc[:  , :  , :-1, :] * dx)
        sdc[:  , 1: , :  , :, 2] = 2 * dc_nbr[:  , 1: , :  , :, 2] * dc[:  , 1: , :  , :] / (dc_nbr[:  , 1: , :  , :, 2] * dy + dc[:  , 1: , :  , :] * dy)
        sdc[:  , :-1, :  , :, 3] = 2 * dc_nbr[:  , :-1, :  , :, 3] * dc[:  , :-1, :  , :] / (dc_nbr[:  , :-1, :  , :, 3] * dy + dc[:  , :-1, :  , :] * dy)
        sdc[1: , :  , :  , :, 4] = 2 * dc_nbr[1: , :  , :  , :, 4] * dc[1: , :  , :  , :] / (dc_nbr[1: , :  , :  , :, 4] * dz + dc[1: , :  , :  , :] * dz)
        sdc[:-1, :  , :  , :, 5] = 2 * dc_nbr[:-1, :  , :  , :, 5] * dc[:-1, :  , :  , :] / (dc_nbr[:-1, :  , :  , :, 5] * dz + dc[:-1, :  , :  , :] * dz)

        # Compute the surface diffusion coefficient correction terms
        sdc_corr = np.zeros((nz, ny, nx, ng, 6))
        sdc_corr[..., 0] = (-sdc[..., 0] * (-flux_nbr[..., 0] + flux) - net_current[..., 0]) / (flux_nbr[..., 0] + flux)
        sdc_corr[..., 1] = (-sdc[..., 1] * ( flux_nbr[..., 1] - flux) - net_current[..., 1]) / (flux_nbr[..., 1] + flux)
        sdc_corr[..., 2] = (-sdc[..., 2] * (-flux_nbr[..., 2] + flux) - net_current[..., 2]) / (flux_nbr[..., 2] + flux)
        sdc_corr[..., 3] = (-sdc[..., 3] * ( flux_nbr[..., 3] - flux) - net_current[..., 3]) / (flux_nbr[..., 3] + flux)
        sdc_corr[..., 4] = (-sdc[..., 4] * (-flux_nbr[..., 4] + flux) - net_current[..., 4]) / (flux_nbr[..., 4] + flux)
        sdc_corr[..., 5] = (-sdc[..., 5] * ( flux_nbr[..., 5] - flux) - net_current[..., 5]) / (flux_nbr[..., 5] + flux)

        flux_array = np.repeat(flux, 6)
        flux_array.shape = (nz, ny, nx, ng, 6)

        # Check for diagonal dominance
        if check_for_diag_dominance:

            # Make a mask of the location of all terms that need to be corrected (dd_mask) and
            # terms that don't need to be corrected (nd_mask)
            dd_mask = (np.abs(sdc_corr) > sdc)
            nd_mask = (dd_mask == False)

            # Save arrays as to whether the correction term is positive or negative.
            sign = np.abs(sdc_corr) / sdc_corr
            sign_pos  = (sdc_corr > 0.)
            sense_pos = np.zeros((nz, ny, nx, ng, 6))
            sense_pos[..., 0:6:2] = False
            sense_pos[..., 1:6:2] = True
            sign_sense = (sign_pos == sense_pos)
            not_sign_sense = (sign_sense == False)

            # Correct sdc
            sdc[:  , :  , 1: , :, 0] = nd_mask[:  , :  , 1: , :, 0] * sdc[:  , :  , 1: , :, 0] + dd_mask[:  , :  , 1: , :, 0] * (sign_sense[:  , :  , 1: , :, 0] * np.abs(net_current[:  , :  , 1: , :, 0] / (2 * flux_nbr[:  , :  , 1: , :, 0])) + not_sign_sense[:  , :  , 1: , :, 0] * np.abs(net_current[:  , :  , 1: , :, 0] / (2 * flux[:  , :  , 1: , :])))
            sdc[:  , :  , :-1, :, 1] = nd_mask[:  , :  , :-1, :, 1] * sdc[:  , :  , :-1, :, 1] + dd_mask[:  , :  , :-1, :, 1] * (sign_sense[:  , :  , :-1, :, 1] * np.abs(net_current[:  , :  , :-1, :, 1] / (2 * flux_nbr[:  , :  , :-1, :, 1])) + not_sign_sense[:  , :  , :-1, :, 1] * np.abs(net_current[:  , :  , :-1, :, 1] / (2 * flux[:  , :  , :-1, :])))
            sdc[:  , 1: , :  , :, 2] = nd_mask[:  , 1: , :  , :, 2] * sdc[:  , 1: , :  , :, 2] + dd_mask[:  , 1: , :  , :, 2] * (sign_sense[:  , 1: , :  , :, 2] * np.abs(net_current[:  , 1: , :  , :, 2] / (2 * flux_nbr[:  , 1: , :  , :, 2])) + not_sign_sense[:  , 1: , :  , :, 2] * np.abs(net_current[:  , 1: , :  , :, 2] / (2 * flux[:  , 1: , :  , :])))
            sdc[:  , :-1, :  , :, 3] = nd_mask[:  , :-1, :  , :, 3] * sdc[:  , :-1, :  , :, 3] + dd_mask[:  , :-1, :  , :, 3] * (sign_sense[:  , :-1, :  , :, 3] * np.abs(net_current[:  , :-1, :  , :, 3] / (2 * flux_nbr[:  , :-1, :  , :, 3])) + not_sign_sense[:  , :-1, :  , :, 3] * np.abs(net_current[:  , :-1, :  , :, 3] / (2 * flux[:  , :-1, :  , :])))
            sdc[1: , :  , :  , :, 4] = nd_mask[1: , :  , :  , :, 4] * sdc[1: , :  , :  , :, 4] + dd_mask[1: , :  , :  , :, 4] * (sign_sense[1: , :  , :  , :, 4] * np.abs(net_current[1: , :  , :  , :, 4] / (2 * flux_nbr[1: , :  , :  , :, 4])) + not_sign_sense[1: , :  , :  , :, 4] * np.abs(net_current[1: , :  , :  , :, 4] / (2 * flux[1: , :  , :  , :])))
            sdc[:-1, :  , :  , :, 5] = nd_mask[:-1, :  , :  , :, 5] * sdc[:-1, :  , :  , :, 5] + dd_mask[:-1, :  , :  , :, 5] * (sign_sense[:-1, :  , :  , :, 5] * np.abs(net_current[:-1, :  , :  , :, 5] / (2 * flux_nbr[:-1, :  , :  , :, 5])) + not_sign_sense[:-1, :  , :  , :, 5] * np.abs(net_current[:-1, :  , :  , :, 5] / (2 * flux[:-1, :  , :  , :])))

            sdc_corr[:  , :  , 1: , :, 0] = nd_mask[:  , :  , 1: , :, 0] * sdc_corr[:  , :  , 1: , :, 0] + dd_mask[:  , :  , 1: , :, 0] * sign[:  , :  , 1: , :, 0] * sdc[:  , :  , 1: , :, 0]
            sdc_corr[:  , :  , :-1, :, 1] = nd_mask[:  , :  , :-1, :, 1] * sdc_corr[:  , :  , :-1, :, 1] + dd_mask[:  , :  , :-1, :, 1] * sign[:  , :  , :-1, :, 1] * sdc[:  , :  , :-1, :, 1]
            sdc_corr[:  , 1: , :  , :, 2] = nd_mask[:  , 1: , :  , :, 2] * sdc_corr[:  , 1: , :  , :, 2] + dd_mask[:  , 1: , :  , :, 2] * sign[:  , 1: , :  , :, 2] * sdc[:  , 1: , :  , :, 2]
            sdc_corr[:  , :-1, :  , :, 3] = nd_mask[:  , :-1, :  , :, 3] * sdc_corr[:  , :-1, :  , :, 3] + dd_mask[:  , :-1, :  , :, 3] * sign[:  , :-1, :  , :, 3] * sdc[:  , :-1, :  , :, 3]
            sdc_corr[1: , :  , :  , :, 4] = nd_mask[1: , :  , :  , :, 4] * sdc_corr[1: , :  , :  , :, 4] + dd_mask[1: , :  , :  , :, 4] * sign[1: , :  , :  , :, 4] * sdc[1: , :  , :  , :, 4]
            sdc_corr[:-1, :  , :  , :, 5] = nd_mask[:-1, :  , :  , :, 5] * sdc_corr[:-1, :  , :  , :, 5] + dd_mask[:-1, :  , :  , :, 5] * sign[:-1, :  , :  , :, 5] * sdc[:-1, :  , :  , :, 5]

        # Multiply by the surface area
        sdc[..., 0:2] *= dy * dz
        sdc[..., 2:4] *= dx * dz
        sdc[..., 4:6] *= dx * dy
        sdc_corr[..., 0:2] *= dy * dz
        sdc_corr[..., 2:4] *= dx * dz
        sdc_corr[..., 4:6] *= dx * dy

        # Reshape the diffusion coefficient array
        flux.shape        = (nx*ny*nz, ng)
        sdc.shape         = (nx*ny*nz*ng, 6)
        sdc_corr.shape    = (nx*ny*nz*ng, 6)

        # Set the diagonal
        sdc_diag      =  sdc     [:, 1:6:2].sum(axis=1) + sdc     [:, 0:6:2].sum(axis=1)
        sdc_corr_diag = -sdc_corr[:, 1:6:2].sum(axis=1) + sdc_corr[:, 0:6:2].sum(axis=1)
        sdc_data      = [sdc_diag]
        sdc_corr_data = [sdc_corr_diag]
        diags         = [0]

        sdc_corr_copy = np.copy(sdc_corr)

        # Zero boundary sdc_corr
        sdc_corr_copy.shape = (nz, ny, nx, ng, 6)
        sdc_corr_copy[:  ,  :,  0, :, 0] = 0.
        sdc_corr_copy[:  ,  :, -1, :, 1] = 0.
        sdc_corr_copy[:  ,  0,  :, :, 2] = 0.
        sdc_corr_copy[:  , -1,  :, :, 3] = 0.
        sdc_corr_copy[0  ,  :,  :, :, 4] = 0.
        sdc_corr_copy[-1 ,  :,  :, :, 5] = 0.
        sdc_corr_copy.shape = (nz*ny*nx*ng, 6)

        # Set the off-diagonals
        if nx > 1:
            sdc_data.append(-sdc[ng: , 0])
            sdc_data.append(-sdc[:-ng, 1])
            sdc_corr_data.append( sdc_corr_copy[ng: , 0])
            sdc_corr_data.append(-sdc_corr_copy[:-ng, 1])
            diags.append(-ng)
            diags.append(ng)
        if ny > 1:
            sdc_data.append(-sdc[nx*ng: , 2])
            sdc_data.append(-sdc[:-nx*ng, 3])
            sdc_corr_data.append( sdc_corr_copy[nx*ng: , 2])
            sdc_corr_data.append(-sdc_corr_copy[:-nx*ng, 3])
            diags.append(-nx*ng)
            diags.append(nx*ng)
        if nz > 1:
            sdc_data.append(-sdc[nx*ny*ng: , 4])
            sdc_data.append(-sdc[:-nx*ny*ng, 5])
            sdc_corr_data.append( sdc_corr_copy[nx*ny*ng: , 4])
            sdc_corr_data.append(-sdc_corr_copy[:-nx*ny*ng, 5])
            diags.append(-nx*ny*ng)
            diags.append(nx*ny*ng)

        # Form a matrix of the surface diffusion coefficients corrections
        sdc_data = np.nan_to_num(sdc_data)
        sdc_corr_data = np.nan_to_num(sdc_corr_data)
        sdc_matrix = sps.diags(sdc_data, diags)
        sdc_corr_matrix = sps.diags(sdc_corr_data, diags)

        return sdc_matrix, sdc_corr_matrix


class DerivedState(State):
    """State to store all the variables that describe a specific state of the system.

    Attributes
    ----------
    mesh : openmc.mesh.Mesh
        Mesh which specifies the dimensions of coarse mesh.

    unity_mesh : openmc.mesh.Mesh
        Mesh which specifies contains only one cell.

    one_group : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the a one-energy-group structure.

    energy_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the energy groups structure.

    fine_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups used to tally the transport cross section that will be
        condensed to get the diffusion coefficients in the coarse group
        structure.

    flux : OrderedDict of np.ndarray
        Numpy array used to store the flux.

    adjoint_flux : OrderedDict of np.ndarray
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

    nxyz : int
        The number of mesh cells.

    dxyz : float
        The volume of a mesh cell.

    num_delayed_groups : int
        The number of delayed neutron precursor groups.

    nd : int
        The number of delayed neutron precursor groups.

    ng : int
        The number of energy groups.

    time_point : str
        The time point of this state.

    clock : openmc.kinetics.Clock
        A clock object to indicate the current simulation time.

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
        clone._mesh = self.mesh
        clone._unity_mesh = self.unity_mesh
        clone._one_group = self.one_group
        clone._energy_groups = self.energy_groups
        clone._flux = copy.deepcopy(self._flux)
        clone._adjoint_flux = copy.deepcopy(self._adjoint_flux)
        clone._precursors = copy.deepcopy(self.precursors)
        clone._k_crit = self.k_crit
        clone._num_delayed_groups = self.num_delayed_groups
        clone._clock = self.clock
        clone._time_point = self.time_point
        clone._core_volume = self.core_volume
        clone._chi_delayed_by_delayed_group = self._chi_delayed_by_delayed_group
        clone._chi_delayed_by_mesh = self._chi_delayed_by_mesh
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
        prev_time = self.clock.times[self.previous_state]
        weight = (time_point - prev_time) / (fwd_time - prev_time)
        return weight

    @property
    def inscatter(self):
        wgt = self.weight
        inscatter_fwd  = self.states[self.forward_state].inscatter
        inscatter_prev = self.states[self.previous_state].inscatter
        inscatter = inscatter_fwd * wgt + inscatter_prev * (1 - wgt)
        return inscatter

    @property
    def absorption(self):
        wgt = self.weight
        absorption_fwd  = self.states[self.forward_state].absorption
        absorption_prev = self.states[self.previous_state].absorption
        absorption = absorption_fwd * wgt + absorption_prev * (1 - wgt)
        return absorption

    @property
    def destruction_matrix(self):
        wgt = self.weight
        state_fwd = self.states[self.forward_state]
        state_prev = self.states[self.previous_state]
        stream_fwd,  stream_corr_fwd  = state_fwd.compute_surface_dif_coefs()
        stream_prev, stream_corr_prev = state_prev.compute_surface_dif_coefs()
        stream = stream_fwd * wgt + stream_prev * (1 - wgt)
        stream_corr = stream_corr_prev * wgt + stream_corr_prev * (1 - wgt)
        inscatter = self.get_diagonal_matrix(self.inscatter)
        outscatter = sps.diags(self.outscatter.flatten(), 0)
        absorb = sps.diags(self.absorption.flatten(), 0)

        return self.dxyz * (absorb + outscatter - inscatter) + \
            stream + stream_corr

    @property
    def adjoint_destruction_matrix(self):
        wgt = self.weight
        state_fwd = self.states[self.forward_state]
        state_prev = self.states[self.previous_state]
        stream_fwd,  stream_corr_fwd  = state_fwd.compute_surface_dif_coefs()
        stream_prev, stream_corr_prev = state_prev.compute_surface_dif_coefs()
        stream = stream_fwd * wgt + stream_prev * (1 - wgt)
        stream_corr = stream_corr_prev * wgt + stream_corr_prev * (1 - wgt)
        inscatter = self.get_diagonal_matrix(self.inscatter)
        outscatter = sps.diags(self.outscatter.flatten(), 0)
        absorb = sps.diags(self.absorption.flatten(), 0)
        matrix = self.dxyz * (absorb + outscatter - inscatter)

        return matrix.transpose() + stream

    @property
    def chi_prompt(self):
        wgt = self.weight
        chi_prompt_fwd  = self.states[self.forward_state].chi_prompt
        chi_prompt_prev = self.states[self.previous_state].chi_prompt
        chi_prompt = chi_prompt_fwd * wgt + chi_prompt_prev * (1 - wgt)
        return chi_prompt

    @property
    def prompt_nu_fission(self):
        wgt = self.weight
        prompt_nu_fission_fwd  = self.states[self.forward_state].prompt_nu_fission
        prompt_nu_fission_prev = self.states[self.previous_state].prompt_nu_fission
        prompt_nu_fission = prompt_nu_fission_fwd * wgt + prompt_nu_fission_prev * (1 - wgt)
        return prompt_nu_fission

    @property
    def chi_delayed(self):
        wgt = self.weight
        chi_delayed_fwd  = self.states[self.forward_state].chi_delayed
        chi_delayed_prev = self.states[self.previous_state].chi_delayed
        chi_delayed = chi_delayed_fwd * wgt + chi_delayed_prev * (1 - wgt)
        return chi_delayed

    @property
    def delayed_nu_fission(self):
        wgt = self.weight
        delayed_nu_fission_fwd  = self.states[self.forward_state].delayed_nu_fission
        delayed_nu_fission_prev = self.states[self.previous_state].delayed_nu_fission
        delayed_nu_fission = delayed_nu_fission_fwd * wgt + delayed_nu_fission_prev * (1 - wgt)
        return delayed_nu_fission

    @property
    def kappa_fission(self):
        wgt = self.weight
        kappa_fission_fwd  = self.states[self.forward_state].kappa_fission
        kappa_fission_prev = self.states[self.previous_state].kappa_fission
        kappa_fission = kappa_fission_fwd * wgt + kappa_fission_prev * (1 - wgt)
        return kappa_fission

    @property
    def pin_cell_kappa_fission(self):
        wgt = self.weight
        kappa_fission_fwd  = self.states[self.forward_state].pin_cell_kappa_fission
        kappa_fission_prev = self.states[self.previous_state].pin_cell_kappa_fission
        kappa_fission = kappa_fission_fwd * wgt + kappa_fission_prev * (1 - wgt)
        return kappa_fission

    @property
    def assembly_kappa_fission(self):
        wgt = self.weight
        kappa_fission_fwd  = self.states[self.forward_state].assembly_kappa_fission
        kappa_fission_prev = self.states[self.previous_state].assembly_kappa_fission
        kappa_fission = kappa_fission_fwd * wgt + kappa_fission_prev * (1 - wgt)
        return kappa_fission

    @property
    def pin_cell_shape(self):
        wgt = self.weight
        shape_fwd  = self.states[self.forward_state].pin_cell_shape
        shape_prev = self.states[self.previous_state].pin_cell_shape
        shape = shape_fwd * wgt + shape_prev * (1 - wgt)
        return shape

    @property
    def assembly_shape(self):
        wgt = self.weight
        shape_fwd  = self.states[self.forward_state].assembly_shape
        shape_prev = self.states[self.previous_state].assembly_shape
        shape = shape_fwd * wgt + shape_prev * (1 - wgt)
        return shape

    @property
    def decay_rate(self):
        wgt = self.weight
        decay_rate_fwd  = self.states[self.forward_state].decay_rate
        decay_rate_prev = self.states[self.previous_state].decay_rate
        decay_rate = decay_rate_fwd * wgt + decay_rate_prev * (1 - wgt)
        return decay_rate

    @property
    def inverse_velocity(self):
        wgt = self.weight
        inverse_velocity_fwd  = self.states[self.forward_state].inverse_velocity
        inverse_velocity_prev = self.states[self.previous_state].inverse_velocity
        inverse_velocity = inverse_velocity_fwd * wgt + inverse_velocity_prev * (1 - wgt)
        return inverse_velocity

    @property
    def diffusion_coefficients(self):
        wgt = self.weight
        diffusion_coefficients_fwd  = self.states[self.forward_state].diffusion_coefficients
        diffusion_coefficients_prev = self.states[self.previous_state].diffusion_coefficients
        diffusion_coefficients = diffusion_coefficients_fwd * wgt + diffusion_coefficients_prev * (1 - wgt)
        return diffusion_coefficients

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Create elements and ordered dicts and initialize to None
        self._mgxs_lib = None

    def compute_surface_dif_coefs(self, check_for_diag_dominance=False):

        msg = 'Cannot compute the surface diffusion coefficients ' \
              'for a DerivedState'
        raise ValueError(msg)
