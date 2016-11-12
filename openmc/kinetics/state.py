from collections import OrderedDict
from numbers import Integral
import warnings
import copy
import itertools
import sys

import numpy as np
import scipy.sparse as sps

import openmc
import openmc.checkvalue as cv
import openmc.mgxs
import openmc.kinetics
from openmc.kinetics.clock import TIME_POINTS

if sys.version_info[0] >= 3:
    basestring = str


class State(object):
    """State to store all the variables that describe a specific state of the system.

    Attributes
    ----------
    mesh : openmc.mesh.Mesh
        Mesh which specifies the dimensions of coarse mesh.

    one_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the a one-energy-group structure.

    energy_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups which specifies the energy groups structure.

    fine_groups : openmc.mgxs.groups.EnergyGroups
        EnergyGroups used to tally the transport cross section that will be
        condensed to get the diffusion coefficients in the coarse group
        structure.

    flux : OrderedDict of np.ndarray
        Numpy array used to store the flux.

    precursors : OrderedDict of np.ndarray
        Numpy array used to store the precursor concentrations.

    mgxs_lib : OrderedDict of OrderedDict of openmc.tallies
        Dict of Dict of tallies. The first Dict is indexed by time point
        and the second Dict is indexed by rxn type.

    k_crit : float
        The initial eigenvalue.

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

    time : str
        The time point of this state.

    clock : openmc.kinetics.Clock
        A clock object to indicate the current simulation time.

    """

    def __init__(self):

        # Initialize Solver class attributes
        self._mesh = None
        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._flux = None
        self._precursors = None
        self._mgxs_lib = None
        self._k_crit = 1.0
        self._num_delayed_groups = 6
        self._time = None
        self._clock = None
        self._core_volume = 1.

    def __deepcopy__(self, memo):

        clone = type(self).__new__(type(self))
        clone._mesh = self.mesh
        clone._one_group = self.one_group
        clone._energy_groups = self.energy_groups
        clone._flux = copy.deepcopy(self._flux)
        clone._precursors = copy.deepcopy(self.precursors)
        clone._mgxs_lib = copy.deepcopy(self.mgxs_lib)
        clone._k_crit = self.k_crit
        clone._num_delayed_groups = self.num_delayed_groups
        clone._clock = self.clock
        clone._time = self.time
        clone._core_volume = self.core_volume

        return clone

    @property
    def core_volume(self):
        return self._core_volume

    @property
    def time(self):
        return self._time

    @property
    def clock(self):
        return self._clock

    @property
    def mesh(self):
        return self._mesh

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
    def precursors(self):
        return self._precursors

    @property
    def mgxs_lib(self):
        return self._mgxs_lib

    @property
    def k_crit(self):
        return self._k_crit

    @property
    def num_delayed_groups(self):
        return self._num_delayed_groups

    @property
    def nxzy(self):
        return np.prod(self.mesh.dimension)

    @property
    def dxzy(self):
        return np.prod(self.mesh.width)

    @property
    def ng(self):
        return self.energy_groups.num_groups

    @property
    def nd(self):
        return self.num_delayed_groups

    @property
    def dt(self):
        return self.clock.dt_inner


    @core_volume.setter
    def core_volume(self, core_volume):
        self._core_volume = core_volume

    @time.setter
    def time(self, time):
        self._time = time

    @clock.setter
    def clock(self, clock):
        self._clock = clock

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh

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
        self._flux = flux

    @precursors.setter
    def precursors(self, precursors):
        self._precursors = precursors

    @mgxs_lib.setter
    def mgxs_lib(self, mgxs_lib):
        self._mgxs_lib = mgxs_lib

    @k_crit.setter
    def k_crit(self, k_crit):
        self._k_crit = k_crit

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        self._num_delayed_groups = num_delayed_groups

    @property
    def inscatter(self):
        inscatter = self.mgxs_lib['nu-scatter matrix'].get_xs(row_column='outin')
        inscatter.shape = (self.nxyz, self.ng, self.ng)
        return inscatter

    @property
    def outscatter(self):
        return inscatter.sum(axis=1)

    @property
    def absorption(self):
        absorption = self.mgxs_lib['absorption'].get_xs()
        absorption.shape = (self.nxyz, self.ng)
        return absorption

    @property
    def destruction_matrix(self):
        stream, stream_corr = self.compute_surface_dif_coefs()
        inscatter = sps.block_diag(self.inscatter)
        outscatter = sps.diags(self.outscatter.flatten())
        absorb = sps.diags(self.absorption.flatten())

        return self.dxyz * (absorb + outscatter - inscatter) + \
            stream + stream_corr

    @property
    def chi_prompt(self):
        chi_prompt = self.mgxs_lib['chi-prompt'].get_xs()
        chi_prompt.shape = (self.nxyz, self.ng)
        return chi_prompt

    @property
    def prompt_nu_fission(self):
        prompt_nu_fission = self.mgxs_lib['prompt-nu-fission'].get_xs()
        prompt_nu_fission.shape = (self.nxyz, self.ng)
        return prompt_nu_fission

    @property
    def prompt_production(self):
        chi_prompt = np.repeat(self.chi_prompt, self.ng)
        chi_prompt.shape = (self.nxyz, self.ng, self.ng)
        prompt_nu_fission = np.tile(self.prompt_nu_fission, self.ng)
        prompt_nu_fission.shape = (self.nxyz, self.ng, self.ng)
        return self.dxyz * chi_prompt * prompt_nu_fission / self.k_crit

    @property
    def chi_delayed(self):
        chi_delayed = self.mgxs_lib['chi-delayed'].get_xs()
        chi_delayed.shape = (self.nxyz, self.nd, self.ng)
        return chi_delayed

    @property
    def delayed_nu_fission(self):
        delayed_nu_fission = self.mgxs_lib['delayed-nu-fission'].get_xs()
        delayed_nu_fission.shape = (self.nxyz, self.nd, self.ng)
        return delayed_nu_fission

    @property
    def delayed_production(self):
        chi_delayed = np.repeat(self.chi_delayed, self.ng)
        chi_delayed.shape = (self.nxyz, self.dg, self.ng, self.ng)
        delayed_nu_fission = np.tile(self.delayed_nu_fission, self.ng)
        delayed_nu_fission.shape = (self.nxyz, self.dg, self.ng, self.ng)
        return self.dxyz * chi_delayed * delayed_nu_fission / self.k_crit

    @property
    def production_matrix(self):
        return sps.block_diag(self.prompt_production +
                              self.delayed_production.sum(axis=1))

    @property
    def prompt_production_matrix(self):
        return sps.block_diag(self.prompt_production)

    @property
    def kappa_fission(self):
        kappa_fission = self.mgxs_lib['kappa-fission'].get_xs()
        kappa_fission.shape = (self.nxyz, self.ng)

    @property
    def core_power_density(self):
        return self.mesh_powers.sum() / self.core_volume

    @property
    def mesh_powers(self):
        return self.dxyz * (self.kappa_fission * self.flux).sum(axis=1)

    @property
    def time_source_matrix(self):
        time_source = self.dxyz * self.inverse_velocity / self.dt
        return sps.diags(time_source.flatten())

    @property
    def decay_source(self):
        decay_source = self.decay_rate * self.precursors
        decay_source = np.repeat(decay_source, self.ng)
        decay_source.shape = (self.nxyz, self.nd, self.ng)
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
        return self.time_source_matrix + self.prompt_production_matrix \
            - self.destruction_matrix + self.decay_production_matrix

    @property
    def decay_production_matrix(self):
        decay_term = self.decay_rate * self.dt / (1. + self.dt * self.decay_rate)
        decay_term = np.repeat(decay_term, self.ng * self.ng)
        decay_term.shape = (self.nxyz, self.nd, self.ng, self.ng)
        return sps.block_diag((self.delayed_production * decay_term).sum(axis=1))

    @property
    def delayed_fission_rate(self):
        flux = np.tile(self.flux, nd)
        flux.shape = (self.nxyz, self.nd, self.ng)
        return (self.delayed_nu_fission * flux).sum(axis=2)

    def propagate_precursors(self, state):

        # Get the flux and repeat to cover all delayed groups
        delayed_source = self.delayed_fission_rate * self.dt / \
                         (1 + self.dt * self.decay_rate) / self.k_crit

        self.precursors = delayed_source + state.precursors

    def compute_initial_precursor_concentration(self):
        self.precursors = self.delayed_fission_rate / self.decay_rate / self.k_crit

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Instantiate a list of the delayed groups
        delayed_groups = list(range(1,self.num_delayed_groups + 1))

        # Create elements and ordered dicts and initialize to None
        self._mgxs_lib = OrderedDict()

        mgxs_types = ['transport', 'absorption',
                      'kappa-fission', 'nu-scatter matrix', 'chi-prompt',
                      'chi-delayed', 'inverse-velocity', 'prompt-nu-fission',
                      'current', 'delayed-nu-fission', 'decay-rate', 'total']

        # Populate the MGXS in the MGXS lib
        for mgxs_type in mgxs_types:
            if mgxs_type == 'diffusion-coefficient':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.fine_groups, by_nuclide=False,
                    name= self.time + ' - ' + mgxs_type)
            elif mgxs_type == 'nu-scatter matrix':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.energy_groups, by_nuclide=False,
                    name= self.time + ' - ' + mgxs_type)
                self._mgxs_lib[mgxs_type].correction = None
            elif mgxs_type == 'decay-rate':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.one_group,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name= self.time + ' - ' + mgxs_type)
            elif mgxs_type in openmc.mgxs.MGXS_TYPES:
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.energy_groups, by_nuclide=False,
                    name= self.time + ' - ' + mgxs_type)
            elif mgxs_type in openmc.mgxs.MDGXS_TYPES:
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.energy_groups,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name= self.time + ' - ' + mgxs_type)

    def compute_surface_dif_coefs(self):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.ng

        # Compute the net current
        partial_current = self.mgxs_lib['current'].get_xs()
        partial_current.shape = (partial_current.shape[0] / 12, 12)
        net_current = partial_current[:, range(6)] - partial_current[:, range(6,12)]
        net_current[:, 0:6:2] = -net_current[:, 0:6:2]
        net_current.shape = (nz, ny, nx, ng, 6)
        net_current[..., 0:2] /= (dy * dz)
        net_current[..., 2:4] /= (dx * dz)
        net_current[..., 4:6] /= (dx * dy)

        # Get the flux
        flux = copy.deepcopy(self.flux)
        flux.shape = (nz, ny, nx, ng)
        flux_array = np.zeros((nz, ny, nx, ng, 6))

        # Create a 2D array of the diffusion coefficients
        flux_array[:  , :  , 1: , :, 0] = flux[:  , :  , :-1, :]
        flux_array[:  , :  , :-1, :, 1] = flux[:  , :  , 1: , :]
        flux_array[:  , 1: , :  , :, 2] = flux[:  , :-1, :  , :]
        flux_array[:  , :-1, :  , :, 3] = flux[:  , 1: , :  , :]
        flux_array[1: , :  , :  , :, 4] = flux[:-1, :  , :  , :]
        flux_array[:-1, :  , :  , :, 5] = flux[1: , :  , :  , :]

        # Get the diffusion coefficients tally
        #dc_mgxs = self.mgxs_lib['diffusion-coefficient']
        #dc_mgxs = dc_mgxs.get_condensed_xs(self.energy_groups)
        dc_mgxs = self.mgxs_lib['transport']
        dc = 1.0 / (3.0 * dc_mgxs.get_xs())
        dc.shape = (nz, ny, nx, ng)
        dc_array = np.zeros((nz, ny, nx, ng, 6))

        # Create a 2D array of the diffusion coefficients
        dc_array[:  , :  , 1: , :, 0] = dc[:  , :  , :-1, :]
        dc_array[:  , :  , :-1, :, 1] = dc[:  , :  , 1: , :]
        dc_array[:  , 1: , :  , :, 2] = dc[:  , :-1, :  , :]
        dc_array[:  , :-1, :  , :, 3] = dc[:  , 1: , :  , :]
        dc_array[1: , :  , :  , :, 4] = dc[:-1, :  , :  , :]
        dc_array[:-1, :  , :  , :, 5] = dc[1: , :  , :  , :]

        # Compute the surface diffusion coefficients for interior surfaces
        sdc = np.zeros((nz, ny, nx, ng, 6))
        sdc[..., 0] = 2 * dc_array[..., 0] * dc / (dc_array[..., 0] * dx + dc * dx)
        sdc[..., 1] = 2 * dc_array[..., 1] * dc / (dc_array[..., 1] * dx + dc * dx)
        sdc[..., 2] = 2 * dc_array[..., 2] * dc / (dc_array[..., 2] * dy + dc * dy)
        sdc[..., 3] = 2 * dc_array[..., 3] * dc / (dc_array[..., 3] * dy + dc * dy)
        sdc[..., 4] = 2 * dc_array[..., 4] * dc / (dc_array[..., 4] * dz + dc * dz)
        sdc[..., 5] = 2 * dc_array[..., 5] * dc / (dc_array[..., 5] * dz + dc * dz)

        # net_current, flux_array, surf_dif_coef
        sdc_corr = np.zeros((nz, ny, nx, ng, 6))
        sdc_corr[..., 0] = (-sdc[..., 0] * (-flux_array[..., 0] + flux) - net_current[..., 0]) / (flux_array[..., 0] + flux)
        sdc_corr[..., 1] = (-sdc[..., 1] * ( flux_array[..., 1] - flux) - net_current[..., 1]) / (flux_array[..., 1] + flux)
        sdc_corr[..., 2] = (-sdc[..., 2] * (-flux_array[..., 2] + flux) - net_current[..., 2]) / (flux_array[..., 2] + flux)
        sdc_corr[..., 3] = (-sdc[..., 3] * ( flux_array[..., 3] - flux) - net_current[..., 3]) / (flux_array[..., 3] + flux)
        sdc_corr[..., 4] = (-sdc[..., 4] * (-flux_array[..., 4] + flux) - net_current[..., 4]) / (flux_array[..., 4] + flux)
        sdc_corr[..., 5] = (-sdc[..., 5] * ( flux_array[..., 5] - flux) - net_current[..., 5]) / (flux_array[..., 5] + flux)

        # net_current, flux_array, surf_dif_coef
        sdc_corr_od = np.zeros((nz, ny, nx, ng, 6))
        sdc_corr_od[:  ,:  ,1: ,:,0] = sdc_corr[:  ,:  ,1: ,:,0]
        sdc_corr_od[:  ,:  ,:-1,:,1] = sdc_corr[:  ,:  ,:-1,:,1]
        sdc_corr_od[:  ,1: ,:  ,:,2] = sdc_corr[:  ,1: ,:  ,:,2]
        sdc_corr_od[:  ,:-1,:  ,:,3] = sdc_corr[:  ,:-1,:  ,:,3]
        sdc_corr_od[1: ,:  ,:  ,:,4] = sdc_corr[1: ,:  ,:  ,:,4]
        sdc_corr_od[:-1,:  ,:  ,:,5] = sdc_corr[:-1,:  ,:  ,:,5]

        # Check for diagonal dominance
        dd_mask = (np.abs(sdc_corr_od) > sdc)
        nd_mask = (dd_mask == False)
        pos = (sdc_corr_od > 0.)
        neg = (sdc_corr_od < 0.)

        # Correct sdc for diagonal dominance
        sdc[:  ,:  ,1: ,:,0] = nd_mask[:  ,:  ,1: ,:,0] * sdc[:  ,:  ,1: ,:,0] + dd_mask[:  ,:  ,1: ,:,0] * (neg[:  ,:  ,1: ,:,0] * np.abs(net_current[:  ,:  ,1: ,:,0] / (2 * flux_array[:  ,:  ,1: ,:,0])) + pos[:  ,:  ,1: ,:,0] * np.abs(net_current[:  ,:  ,1: ,:,0] / (2 * flux[:  ,:  ,1: ,:])))
        sdc[:  ,:  ,:-1,:,1] = nd_mask[:  ,:  ,:-1,:,1] * sdc[:  ,:  ,:-1,:,1] + dd_mask[:  ,:  ,:-1,:,1] * (pos[:  ,:  ,:-1,:,1] * np.abs(net_current[:  ,:  ,:-1,:,1] / (2 * flux_array[:  ,:  ,:-1,:,1])) + neg[:  ,:  ,:-1,:,1] * np.abs(net_current[:  ,:  ,:-1,:,1] / (2 * flux[:  ,:  ,:-1,:])))
        sdc[:  ,1: ,:  ,:,2] = nd_mask[:  ,1: ,:  ,:,2] * sdc[:  ,1: ,:  ,:,2] + dd_mask[:  ,1: ,:  ,:,2] * (neg[:  ,1: ,:  ,:,2] * np.abs(net_current[:  ,1: ,:  ,:,2] / (2 * flux_array[:  ,1: ,:  ,:,2])) + pos[:  ,1: ,:  ,:,2] * np.abs(net_current[:  ,1: ,:  ,:,2] / (2 * flux[:  ,1: ,:  ,:])))
        sdc[:  ,:-1,:  ,:,3] = nd_mask[:  ,:-1,:  ,:,3] * sdc[:  ,:-1,:  ,:,3] + dd_mask[:  ,:-1,:  ,:,3] * (pos[:  ,:-1,:  ,:,3] * np.abs(net_current[:  ,:-1,:  ,:,3] / (2 * flux_array[:  ,:-1,:  ,:,3])) + neg[:  ,:-1,:  ,:,3] * np.abs(net_current[:  ,:-1,:  ,:,3] / (2 * flux[:  ,:-1,:  ,:])))
        sdc[1: ,:  ,:  ,:,4] = nd_mask[1: ,:  ,:  ,:,4] * sdc[1: ,:  ,:  ,:,4] + dd_mask[1: ,:  ,:  ,:,4] * (neg[1: ,:  ,:  ,:,4] * np.abs(net_current[1: ,:  ,:  ,:,4] / (2 * flux_array[1: ,:  ,:  ,:,4])) + pos[1: ,:  ,:  ,:,4] * np.abs(net_current[1: ,:  ,:  ,:,4] / (2 * flux[1: ,:  ,:  ,:])))
        sdc[:-1,:  ,:  ,:,5] = nd_mask[:-1,:  ,:  ,:,5] * sdc[:-1,:  ,:  ,:,5] + dd_mask[:-1,:  ,:  ,:,5] * (pos[:-1,:  ,:  ,:,5] * np.abs(net_current[:-1,:  ,:  ,:,5] / (2 * flux_array[:-1,:  ,:  ,:,5])) + neg[:-1,:  ,:  ,:,5] * np.abs(net_current[:-1,:  ,:  ,:,5] / (2 * flux[:-1,:  ,:  ,:])))

        # Correct sdc correct for diagonal dominance
        sdc_corr_od = nd_mask * sdc_corr_od + dd_mask * (pos * sdc - neg * sdc)

        # Multiply by the surface area
        sdc[..., 0:2] *= dy * dz
        sdc[..., 2:4] *= dx * dz
        sdc[..., 4:6] *= dx * dy
        sdc_corr[..., 0:2] *= dy * dz
        sdc_corr[..., 2:4] *= dx * dz
        sdc_corr[..., 4:6] *= dx * dy

        # Reshape the diffusion coefficient array
        flux.shape = (nx*ny*nz*ng)
        sdc.shape = (nx*ny*nz*ng, 6)
        sdc_corr.shape = (nx*ny*nz*ng, 6)
        sdc_corr_od.shape = (nx*ny*nz*ng, 6)

        # Set the diagonal
        sdc_diag = sdc.sum(axis=1)
        sdc_corr_diag = sdc_corr[:, 0:6:2].sum(axis=1) - sdc_corr[:, 1:6:2].sum(axis=1)
        sdc_data  = [sdc_diag]
        sdc_corr_data  = [sdc_corr_diag]
        diags = [0]

        # Set the off-diagonals
        if nx > 1:
            sdc_data.append(-sdc[ng:, 0])
            sdc_data.append(-sdc[:-ng, 1])
            sdc_corr_data.append( sdc_corr_od[ng:, 0])
            sdc_corr_data.append(-sdc_corr_od[:-ng, 1])
            diags.append(-ng)
            diags.append(ng)
        if ny > 1:
            sdc_data.append(-sdc[nx*ng:, 2])
            sdc_data.append(-sdc[:-nx*ng   , 3])
            sdc_corr_data.append( sdc_corr_od[nx*ng:, 2])
            sdc_corr_data.append(-sdc_corr_od[:-nx*ng   , 3])
            diags.append(-nx*ng)
            diags.append(nx*ng)
        if nz > 1:
            sdc_data.append(-sdc[nx*ny*ng:, 4])
            sdc_data.append(-sdc[:-nx*ny*ng, 5])
            sdc_corr_data.append( sdc_corr_od[nx*ny*ng:, 4])
            sdc_corr_data.append(-sdc_corr_od[:-nx*ny*ng, 5])
            diags.append(-nx*ny*ng)
            diags.append(nx*ny*ng)

        # Form a matrix of the surface diffusion coefficients corrections
        sdc_matrix = sps.diags(sdc_data, diags)
        sdc_corr_matrix = sps.diags(sdc_corr_data, diags)

        return sdc_matrix, sdc_corr_matrix
