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

    precursor_conc : OrderedDict of np.ndarray
        Numpy array used to store the precursor concentrations.

    mgxs_lib : OrderedDict of OrderedDict of openmc.tallies
        Dict of Dict of tallies. The first Dict is indexed by time point
        and the second Dict is indexed by rxn type.

    k_crit : float
        The initial eigenvalue.

    num_delayed_groups : int
        The number of delayed neutron precursor groups.

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
        self._precursor_conc = None
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
        clone._precursor_conc = copy.deepcopy(self.precursor_conc)
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
    def precursor_conc(self):
        return self._precursor_conc

    @property
    def mgxs_lib(self):
        return self._mgxs_lib

    @property
    def k_crit(self):
        return self._k_crit

    @property
    def num_delayed_groups(self):
        return self._num_delayed_groups

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

    @precursor_conc.setter
    def precursor_conc(self, precursor_conc):
        self._precursor_conc = precursor_conc

    @mgxs_lib.setter
    def mgxs_lib(self, mgxs_lib):
        self._mgxs_lib = mgxs_lib

    @k_crit.setter
    def k_crit(self, k_crit):
        self._k_crit = k_crit

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        self._num_delayed_groups = num_delayed_groups

    def get_destruction_matrix(self):

        # Get the volume of a mesh cell
        dxyz = np.prod(self.mesh.width)

        # Extract the necessary cross sections
        inscatter = self.mgxs_lib['nu-scatter matrix']\
                        .get_xs(representation='matrix')
        absorb = self.mgxs_lib['absorption'].get_xs(reprensentation='matrix')
        stream, stream_corr = self.compute_surface_dif_coefs()
        outscatter = sps.diags(np.asarray(inscatter.sum(axis=0)).flatten(), 0)
        loss_matrix = dxyz * (absorb + outscatter - inscatter) \
                      + stream + stream_corr

        # Return the destruction matrix
        return loss_matrix

    def get_production_matrix(self):

        # Get the volume of a mesh cell
        dxyz = np.prod(self.mesh.width)

        # Extract the necessary cross sections
        chi_p = self.mgxs_lib['chi-prompt'].get_xs(representation='matrix')
        chi_d = self.mgxs_lib['chi-delayed'].get_xs(representation='matrix')
        nu_fis_p = self.mgxs_lib['prompt-nu-fission']\
                       .get_xs(representation='matrix')
        nu_fis_d = self.mgxs_lib['delayed-nu-fission']\
                       .get_xs(representation='matrix')

        # Return the production matrix
        return dxyz * (chi_p * nu_fis_p + chi_d * nu_fis_d)

    def get_kappa_fission_matrix(self):

        # Extract the necessary cross sections
        kappa_fission = self.mgxs_lib['kappa-fission']\
                            .get_xs(representation='matrix')

        # Return the destruction matrix
        return kappa_fission

    def get_core_power_density(self):

        # Get the volume of a mesh cell
        dxyz = np.prod(self.mesh.width)

        # Compute the power in each mesh cell
        powers = dxyz * self.get_kappa_fission_matrix() * self.flux

        # Return the core power density
        return powers.sum() / self.core_volume

    def get_mesh_powers(self):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.energy_groups.num_groups

        # Compute the power in each mesh cell
        powers = self.get_kappa_fission_matrix() * self.flux
        powers.shape = (ny, nz, nx, ng)
        powers = powers.sum(axis=3)

        # Return the core power density
        return powers

    def get_source(self):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        dxyz = dx * dy * dz
        ng = self.energy_groups.num_groups
        nd = self.num_delayed_groups

        # Extract the necessary cross sections
        chi_d = self.mgxs_lib['chi-delayed'].get_xs(representation='matrix')
        decay_rate = self.mgxs_lib['decay-rate'].get_xs(representation='array')
        inv_velocity = self.mgxs_lib['inverse-velocity']\
                           .get_xs(representation='matrix')
        dt = self.clock.dt_inner

        # Compute the delayed source
        delayed_source = decay_rate / (1.0 + dt * decay_rate)
        delayed_source = delayed_source * self.precursor_conc

        # Multiply the delayed source by the delayed spectrum
        if self.mgxs_lib['chi-delayed'].delayed_groups is None:
            delayed_source.shape = (nz, ny, nx, nd)
            delayed_source = delayed_source.sum(axis=3)
            delayed_source = delayed_source.flatten()
            delayed_source = np.repeat(delayed_source, ng)
            delayed_source = chi_d * delayed_source
        else:
            delayed_source = np.repeat(delayed_source, ng)
            delayed_source = chi_d * delayed_source
            delayed_source.shape = (nz, ny, nx, nd, ng)
            delayed_source = delayed_source.sum(axis=3)
            delayed_source = delayed_source.flatten()

        # Compute the time absorption source
        time_source = inv_velocity * self.flux / dt

        # Combine to get the total source
        return dxyz * (time_source + delayed_source)

    def get_transient_matrix(self):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.energy_groups.num_groups
        nd = self.num_delayed_groups
        dxyz = dx * dy * dz

        # Extract the necessary cross sections
        chi_p = self.mgxs_lib['chi-prompt'].get_xs(representation='matrix')
        chi_d = self.mgxs_lib['chi-delayed'].get_xs(representation='matrix')
        nu_fis_p = self.mgxs_lib['prompt-nu-fission'].get_xs(representation='matrix')
        nu_fis_d = self.mgxs_lib['delayed-nu-fission'].get_xs(representation='array')
        decay_rate = self.mgxs_lib['decay-rate'].get_xs(representation='array')
        inv_velocity = self.mgxs_lib['inverse-velocity'].get_xs(representation='matrix')
        inscatter = self.mgxs_lib['nu-scatter matrix'].get_xs(representation='matrix')
        absorb = self.mgxs_lib['absorption'].get_xs(representation='matrix')
        stream, stream_corr = self.compute_surface_dif_coefs()
        outscatter = sps.diags(np.asarray(inscatter.sum(axis=0)).flatten(), 0)
        dt = self.clock.dt_inner

        # Get the destruction matrix
        dest_matrix = dxyz * (absorb + outscatter - inscatter) \
                      + stream + stream_corr

        # Get the prompt production matrix
        prod_matrix_p = chi_p * nu_fis_p / self.k_crit

        # Compute the delayed production matrix
        prod_array_d = dt * decay_rate / (1.0 + dt * decay_rate)
        prod_array_d = np.repeat(prod_array_d, ng)
        prod_array_d = prod_array_d * nu_fis_d / self.k_crit

        if self.mgxs_lib['chi-delayed'].delayed_groups is None:
            prod_array_d.shape = (nz, ny, nx, nd, ng)
            prod_array_d = prod_array_d.sum(axis=3)
            prod_array_d = prod_array_d.flatten()
            prod_matrix_d = sps.diags(prod_array_d)
        else:
            print('This option is not currently supported')

        prod_matrix_d = chi_d * prod_matrix_d
        prod_matrix = dxyz * (prod_matrix_p + prod_matrix_d)

        # Compute the time absorption source
        time_source = dxyz * inv_velocity / dt

        # Combine to get the total source
        transient_matrix = time_source + dest_matrix - prod_matrix
        return transient_matrix

    def propagate_precursors(self, state, dt):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.energy_groups.num_groups
        nd = self.num_delayed_groups
        dt = self.clock.dt_inner

        # Extract the necessary cross sections
        nu_fis_d = self.mgxs_lib['delayed-nu-fission'].get_xs(representation='matrix')
        decay_rate = self.mgxs_lib['decay-rate'].get_xs(representation='array')
        old_conc = state.precursor_conc

        # Get the flux and repeat to cover all delayed groups
        flux = copy.deepcopy(self.flux)
        flux = np.tile(flux, nd)

        del_fis_rate = nu_fis_d * flux
        del_fis_rate.shape = (nz, ny, nx, nd, ng)
        del_fis_rate = del_fis_rate.sum(axis=4)
        del_fis_rate = del_fis_rate.flatten()
        source = dt / (1 + dt * decay_rate)
        source *= del_fis_rate / self.k_crit

        self.precursor_conc = old_conc / (1 + dt * decay_rate) + source

    def compute_initial_precursor_concentration(self):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.energy_groups.num_groups
        nd = self.num_delayed_groups

        # Extract the necessary cross sections
        nu_fis_d = self.mgxs_lib['delayed-nu-fission'].get_xs(representation='matrix')
        decay_rate = self.mgxs_lib['decay-rate'].get_xs(representation='array')

        # Get the flux and repeat to cover all delayed groups
        flux = copy.deepcopy(self.flux)
        flux = np.tile(flux, nd)

        del_fis_rate = nu_fis_d * flux
        del_fis_rate.shape = (nz, ny, nx, nd, ng)
        del_fis_rate = del_fis_rate.sum(axis=4)
        del_fis_rate = del_fis_rate.flatten()
        self.precursor_conc = del_fis_rate / decay_rate / self.k_crit

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Instantiate a list of the delayed groups
        delayed_groups = list(range(1,self.num_delayed_groups + 1))

        # Create elements and ordered dicts and initialize to None
        self._mgxs_lib         = OrderedDict()

        mgxs_types = ['transport', 'diffusion-coefficient', 'absorption',
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
            elif mgxs_type == 'chi-delayed':
                self._mgxs_lib[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=self.mesh, domain_type='mesh',
                    energy_groups=self.energy_groups,
                    by_nuclide=False,
                    name= self.time + ' - ' + mgxs_type)
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
        ng = self.energy_groups.num_groups

        # Compute the net current
        partial_current = self.mgxs_lib['current'].get_xs(representation='array')
        net_current = partial_current[:, range(6)] - partial_current[:, range(6,12)]
        net_current[:, 0:6:2] = -net_current[:, 0:6:2]
        net_current.shape = (nz, ny, nx, ng, 6)
        net_current[..., 0:2] /= (dy * dz)
        net_current[..., 2:4] /= (dx * dz)
        net_current[..., 4:6] /= (dx * dy)

        # Get the flux
        flux = self.mgxs_lib['absorption'].tallies['flux'].get_values().flatten()
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
        dc_mgxs = self.mgxs_lib['diffusion-coefficient']
        dc_mgxs = dc_mgxs.get_condensed_xs(self.energy_groups)
        dc = dc_mgxs.get_xs()
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
