from __future__ import division

from collections import OrderedDict
from numbers import Integral
import warnings
import os
import sys
import copy
import abc
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

    statepoint_files : OrderedDict of openmc.statepoint
        Statepoint files for each time point.

    summary_files : OrderedDict of openmc.summary
        Summary files for each time point.

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

    A : OrderedDict of sps.lil_matrix
        Numpy matrix used for storing the destruction terms.

    M : OrderedDict of sps.lil_matrix
        Numpy matrix used for storing the production terms.

    AM : sps.lil_matrix
        Numpy matrix used for storing the combined production/destruction terms.

    flux : OrderedDict of np.ndarray
        Numpy array used to store the flux.

    amplitude : Ordered dict of np.ndarray
        Numpy array used to store the amplitude.

    shape : OrderedDict of np.ndarray
        Numpy array used to store the shape.

    source : OrderedDict of np.ndarray
        Numpy array used to store the source.

    power : OrderedDict of np.ndarray
        Numpy array used to store the power.

    precursor_conc : OrderedDict of np.ndarray
        Numpy array used to store the precursor concentrations.

    mgxs_lib : OrderedDict of OrderedDict of openmc.tallies
        Dict of Dict of tallies. The first Dict is indexed by time point
        and the second Dict is indexed by rxn type.

    k_eff_0 : float
        The initial eigenvalue.

    num_delayed_groups : int
        The number of delayed neutron precursor groups.

    Methods
    -------
    - initialize_xs()
    take_outer_step()
    take_inner_step()
    solve()
    - extract_xs()
    2 normalize_flux()
    broadcast_to_all()
    broadcast_to_one()
    - compute_shape()
    integrate_precursor_conc()
    3 compute_initial_precursor_conc()
    1 compute_power()
    construct_A()
    construct_M()
    construct_AM()
    interpolate_xs()

    To Do
    -----
    1) Create getters and setters for all attributes
    2) Create method to generate initialize xs
    3) Create method to compute flux
    4) Create method to compute initial precursor concentrations
    5) Create method to compute the initial power

    """

    def __init__(self):

        # Initialize Solver class attributes
        self._mesh = None
        self._geometry = None
        self._settings_file = None
        self._materials_file = None
        self._statepoint_files = None
        self._summary_files = None
        self._clock = None
        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._A = None
        self._M = None
        self._AM = None
        self._flux = None
        self._amplitude = None
        self._shape = None
        self._source = None
        self._power = None
        self._precursor_conc = None
        self._mgxs_lib = None
        self._k_eff_0 = None
        self._num_delayed_groups = 6

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
    def statepoint_files(self):
        return self._statepoint_files

    @property
    def summary_files(self):
        return self._summary_files

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
    def A(self):
        return self._A

    @property
    def M(self):
        return self._M

    @property
    def AM(self):
        return self._AM

    @property
    def flux(self):
        return self._flux

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def shape(self):
        return self._shape

    @property
    def source(self):
        return self._source

    @property
    def power(self):
        return self._power

    @property
    def precursor_conc(self):
        return self._precursor_conc

    @property
    def mgxs_lib(self):
        return self._mgxs_lib

    @property
    def k_eff_0(self):
        return self._k_eff_0

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

    @statepoint_files.setter
    def statepoint_files(self, statepoint_files):
        self._statepoint_files = statepoint_files

    @summary_files.setter
    def summary_files(self, summary_files):
        self._summary_files = summary_files

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

    @A.setter
    def A(self):
        self._A = A

    @M.setter
    def M(self, M):
        self._M = M

    @AM.setter
    def AM(self, AM):
        self._AM = AM

    @flux.setter
    def flux(self, flux):
        self._flux = flux

    @amplitude.setter
    def amplitude(self, amplitude):
        self._amplitude = amplitude

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @source.setter
    def source(self, source):
        self._source = source

    @power.setter
    def power(self, power):
        self._power = power

    @precursor_conc.setter
    def precursor_conc(self, precursor_conc):
        self._precursor_conc = precursor_conc

    @mgxs_lib.setter
    def mgxs_lib(self, mgxs_lib):
        self._mgxs_lib = mgxs_lib

    @k_eff_0.setter
    def k_eff_0(self, k_eff_0):
        self._k_eff_0 = k_eff_0

    @num_delayed_groups.setter
    def num_delayed_groups(self, num_delayed_groups):
        self._num_delayed_groups = num_delayed_groups

    def initialize_mgxs(self):
        """Initialize all the tallies for the problem.

        """

        # Instantiate a list of the delayed groups
        delayed_groups = list(range(1,self.num_delayed_groups + 1))

        # Create ordered dicts
        self._A                = OrderedDict()
        self._M                = OrderedDict()
        self._AM               = OrderedDict()
        self._flux             = OrderedDict()
        self._amplitude        = OrderedDict()
        self._shape            = OrderedDict()
        self._source           = OrderedDict()
        self._power            = OrderedDict()
        self._precursor_conc   = OrderedDict()
        self._mgxs_lib         = OrderedDict()
        self._statepoint_files = OrderedDict()
        self._summary_files    = OrderedDict()

        # Create elements and ordered dicts and initialize to None
        for t in TIME_POINTS:
            self._A[t]                = None
            self._M[t]                = None
            self._AM[t]               = None
            self._flux[t]             = None
            self._amplitude[t]        = None
            self._shape[t]            = None
            self._source[t]           = None
            self._power[t]            = None
            self._statepoint_files[t] = None
            self._summary_files[t]    = None
            self._precursor_conc[t]   = None
            self._mgxs_lib[t]         = OrderedDict()

        mgxs_types = ['total', 'transport', 'diffusion-coefficient', 'absorption',
                      'kappa-fission', 'nu-scatter matrix', 'chi-prompt', 'chi',
                      'chi-delayed', 'inverse-velocity', 'prompt-nu-fission',
                      'current', 'delayed-nu-fission', 'chi-delayed', 'beta',
                      'nu-fission', 'nu-scatter', 'decay-rate']

        # Populate the MGXS in the MGXS lib
        for t in TIME_POINTS:
            for mgxs_type in mgxs_types:
                if mgxs_type == 'diffusion-coefficient':
                    self._mgxs_lib[t][mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                        mgxs_type, domain=self.mesh, domain_type='mesh',
                        energy_groups=self.fine_groups, by_nuclide=False,
                        name= t + ' - ' + mgxs_type)
                elif mgxs_type == 'nu-scatter matrix':
                    self._mgxs_lib[t][mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                        mgxs_type, domain=self.mesh, domain_type='mesh',
                        energy_groups=self.energy_groups, by_nuclide=False,
                        name= t + ' - ' + mgxs_type)
                    self._mgxs_lib[t][mgxs_type].correction = None
                elif mgxs_type == 'chi-delayed':
                    self._mgxs_lib[t][mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                        mgxs_type, domain=self.mesh, domain_type='mesh',
                        energy_groups=self.energy_groups,
                        by_nuclide=False,
                        name= t + ' - ' + mgxs_type)
                elif mgxs_type == 'decay-rate':
                    self._mgxs_lib[t][mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                        mgxs_type, domain=self.mesh, domain_type='mesh',
                        energy_groups=self.one_group,
                        delayed_groups=delayed_groups, by_nuclide=False,
                        name= t + ' - ' + mgxs_type)
                elif mgxs_type in openmc.mgxs.MGXS_TYPES:
                    self._mgxs_lib[t][mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
                        mgxs_type, domain=self.mesh, domain_type='mesh',
                        energy_groups=self.energy_groups, by_nuclide=False,
                        name= t + ' - ' + mgxs_type)
                elif mgxs_type in openmc.mgxs.MDGXS_TYPES:
                    self._mgxs_lib[t][mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                        mgxs_type, domain=self.mesh, domain_type='mesh',
                        energy_groups=self.energy_groups,
                        delayed_groups=delayed_groups, by_nuclide=False,
                        name= t + ' - ' + mgxs_type)

    def run_openmc(self, time):

        # Create the xml files
        self.geometry.export_to_xml()
        self._materials_file.export_to_xml()
        self._settings_file.export_to_xml()
        self.generate_tallies_file(time)

        # Run OpenMC
        openmc.run(mpi_procs=8)

        # Load MGXS from statepoint
        self.statepoint_files[time] = openmc.StatePoint(
            'statepoint.{}.h5'.format(self.settings_file.batches))

        for mgxs in self.mgxs_lib[time].values():
            mgxs.load_from_statepoint(self.statepoint_files[time])

    def compute_initial_shape(self):

        # Get the dimensions of the mesh
        dx, dy, dz = self.mesh.width
        dxyz = dx * dy * dz

        # Get the matrices needed to reproduce the initial eigenvalue solve
        inscatter = self.mgxs_lib['START']['nu-scatter matrix'].get_mean_matrix()
        absorb = self.mgxs_lib['START']['absorption'].get_mean_matrix()
        chi_p = self.mgxs_lib['START']['chi-prompt'].get_mean_matrix()
        chi_d = self.mgxs_lib['START']['chi-delayed'].get_mean_matrix()
        nu_fis_p = self.mgxs_lib['START']['prompt-nu-fission'].get_mean_matrix()
        nu_fis_d = self.mgxs_lib['START']['delayed-nu-fission'].get_mean_matrix()
        stream = self.compute_surface_dif_coefs('START')
        stream_corr = self.compute_surface_dif_coefs_corr('START')
        flux = self.mgxs_lib['START']['absorption'].tallies['flux'].get_values()
        outscatter = sps.diags(np.asarray(inscatter.sum(axis=0)).flatten(), 0)

        if isinstance(nu_fis_d, list):
            nu_fis_d = sum(nu_fis_d)

        # Form the A and M matrices
        self.A['START'] = dxyz * (absorb + outscatter - inscatter) + stream + stream_corr
        self.M['START'] = dxyz * (chi_p * nu_fis_p + chi_d * nu_fis_d)
        self.flux['START'] = flux.flatten()

        source = self.M['START'] * self.flux['START']
        sink = self.A['START'] * self.flux['START']
        k = source.sum() / sink.sum()
        print('balance k-eff: {0:1.6f}',format(k))

        # Compute the initial eigenvalue
        self.compute_eigenvalue('START')

    def compute_eigenvalue(self, time):

        # Get A, M, and flux
        A = self.A[time]
        M = self.M[time]
        flux = self.flux[time]
        k_eff = self.k_eff_0

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

            print('linear solver iter {0} resid {1:1.5e} k-eff {2:1.6f}'.format(i, residual, k_eff))

            if residual < 1.e-6 and i > 5:
                break

        print('k-eff MC {0:1.6f} -- DIFFUSION {1:1.6f}'.format(self.statepoint_files['START'].k_combined[0], k_eff))
        self.k_eff_0 = k_eff


    def compute_surface_dif_coefs(self, time):

        # Get the diffusion coefficients tally
        dc_mgxs = self.mgxs_lib[time]['diffusion-coefficient']
        dc_mgxs = dc_mgxs.get_condensed_xs(self.energy_groups)
        dc = dc_mgxs.get_xs()

        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.energy_groups.num_groups
        dc_array = np.zeros((nz, ny, nx, ng, 6))
        dc.shape = (nz, ny, nx, ng)
        dc_copy = copy.deepcopy(dc)

        # Create a 2D array of the diffusion coefficients
        dc_array[:  , :  , 1: , :, 0] = dc_copy[:  , :  , :-1, :]
        dc_array[:  , :  , :-1, :, 1] = dc_copy[:  , :  , 1: , :]
        dc_array[:  , 1: , :  , :, 2] = dc_copy[:  , :-1, :  , :]
        dc_array[:  , :-1, :  , :, 3] = dc_copy[:  , 1: , :  , :]
        dc_array[1: , :  , :  , :, 4] = dc_copy[:-1, :  , :  , :]
        dc_array[:-1, :  , :  , :, 5] = dc_copy[1: , :  , :  , :]

        # Compute the surface diffusion coefficients for interior surfaces
        sdc = np.zeros((nz, ny, nx, ng, 6))
        sdc[..., 0] = 2 * dc_array[..., 0] * dc / (dc_array[..., 0] * dx + dc * dx)
        sdc[..., 1] = 2 * dc_array[..., 1] * dc / (dc_array[..., 1] * dx + dc * dx)
        sdc[..., 2] = 2 * dc_array[..., 2] * dc / (dc_array[..., 2] * dy + dc * dy)
        sdc[..., 3] = 2 * dc_array[..., 3] * dc / (dc_array[..., 3] * dy + dc * dy)
        sdc[..., 4] = 2 * dc_array[..., 4] * dc / (dc_array[..., 4] * dz + dc * dz)
        sdc[..., 5] = 2 * dc_array[..., 5] * dc / (dc_array[..., 5] * dz + dc * dz)

        sdc[..., 0:2] *= dy * dz
        sdc[..., 2:4] *= dx * dz
        sdc[..., 4:6] *= dx * dy

        # Reshape the diffusion coefficient array
        sdc.shape = (nx*ny*nz*ng, 6)
        sdc_diag = sdc.sum(axis=1)
        data  = [sdc_diag]
        diags = [0]

        if nx > 1:
            data.append(-sdc[ng:, 0])
            data.append(-sdc[:-ng, 1])
            diags.append(-ng)
            diags.append(ng)
        if ny > 1:
            data.append(-sdc[nx*ng:, 2])
            data.append(-sdc[:-nx*ng   , 3])
            diags.append(-nx*ng)
            diags.append(nx*ng)
        if nz > 1:
            data.append(-sdc[nx*ny*ng:, 4])
            data.append(-sdc[:-nx*ny*ng, 5])
            diags.append(-nx*ny*ng)
            diags.append(nx*ny*ng)

        # Form a matrix of the surface diffusion coefficients
        sdc_matrix = sps.diags(data, diags)

        return sdc_matrix

    def compute_surface_dif_coefs_corr(self, time):

        # Get the dimensions of the mesh
        nx, ny, nz = self.mesh.dimension
        dx, dy, dz = self.mesh.width
        ng = self.energy_groups.num_groups

        # Compute the net current
        partial_current = self.mgxs_lib[time]['current'].get_mean_array()
        net_current = partial_current[:, range(6)] - partial_current[:, range(6,12)]
        net_current[:, 0:6:2] = -net_current[:, 0:6:2]
        net_current.shape = (nz, ny, nx, ng, 6)

        # Get the flux
        flux = self.mgxs_lib[time]['total'].tallies['flux'].get_values()
        flux.shape = (nz, ny, nx, ng)
        flux_array = np.zeros((nz, ny, nx, ng, 6))
        flux_copy = copy.deepcopy(flux)

        # Create a 2D array of the diffusion coefficients
        flux_array[:  , :  , 1: , :, 0] = flux_copy[:  , :  , :-1, :]
        flux_array[:  , :  , :-1, :, 1] = flux_copy[:  , :  , 1: , :]
        flux_array[:  , 1: , :  , :, 2] = flux_copy[:  , :-1, :  , :]
        flux_array[:  , :-1, :  , :, 3] = flux_copy[:  , 1: , :  , :]
        flux_array[1: , :  , :  , :, 4] = flux_copy[:-1, :  , :  , :]
        flux_array[:-1, :  , :  , :, 5] = flux_copy[1: , :  , :  , :]

        # Get the diffusion coefficients tally
        dc_mgxs = self.mgxs_lib[time]['diffusion-coefficient']
        dc_mgxs = dc_mgxs.get_condensed_xs(self.energy_groups)
        dc = dc_mgxs.get_xs()
        dc.shape = (nz, ny, nx, ng)

        dc_array = np.zeros((nz, ny, nx, ng, 6))
        dc_copy = copy.deepcopy(dc)

        # Create a 2D array of the diffusion coefficients
        dc_array[:  , :  , 1: , :, 0] = dc_copy[:  , :  , :-1, :]
        dc_array[:  , :  , :-1, :, 1] = dc_copy[:  , :  , 1: , :]
        dc_array[:  , 1: , :  , :, 2] = dc_copy[:  , :-1, :  , :]
        dc_array[:  , :-1, :  , :, 3] = dc_copy[:  , 1: , :  , :]
        dc_array[1: , :  , :  , :, 4] = dc_copy[:-1, :  , :  , :]
        dc_array[:-1, :  , :  , :, 5] = dc_copy[1: , :  , :  , :]

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
        sdc_corr[..., 0] = (-sdc[..., 0] * (-flux_array[..., 0] + flux) - net_current[..., 0] / (dy * dz)) / (flux_array[..., 0] + flux)
        sdc_corr[..., 1] = (-sdc[..., 1] * ( flux_array[..., 1] - flux) - net_current[..., 1] / (dy * dz)) / (flux_array[..., 1] + flux)
        sdc_corr[..., 2] = (-sdc[..., 2] * (-flux_array[..., 2] + flux) - net_current[..., 2] / (dx * dz)) / (flux_array[..., 2] + flux)
        sdc_corr[..., 3] = (-sdc[..., 3] * ( flux_array[..., 3] - flux) - net_current[..., 3] / (dx * dz)) / (flux_array[..., 3] + flux)
        sdc_corr[..., 4] = (-sdc[..., 4] * (-flux_array[..., 4] + flux) - net_current[..., 4] / (dx * dy)) / (flux_array[..., 4] + flux)
        sdc_corr[..., 5] = (-sdc[..., 5] * ( flux_array[..., 5] - flux) - net_current[..., 5] / (dx * dy)) / (flux_array[..., 5] + flux)

        # net_current, flux_array, surf_dif_coef
        sdc_corr_od = np.zeros((nz, ny, nx, ng, 6))
        sdc_corr_od[:  ,:  ,1: ,:,0] = (-sdc[:  ,:  ,1: ,:,0] * (-flux_array[:  ,:  ,1: ,:,0] + flux[:  ,:  ,1: ,:]) - net_current[:  ,:  ,1: ,:,0] / (dy * dz)) / (flux_array[:  ,:  ,1: ,:,0] + flux[:  ,:  ,1: ,:])
        sdc_corr_od[:  ,:  ,:-1,:,1] = (-sdc[:  ,:  ,:-1,:,1] * ( flux_array[:  ,:  ,:-1,:,1] - flux[:  ,:  ,:-1,:]) - net_current[:  ,:  ,:-1,:,1] / (dy * dz)) / (flux_array[:  ,:  ,:-1,:,1] + flux[:  ,:  ,:-1,:])
        sdc_corr_od[:  ,1: ,:  ,:,2] = (-sdc[:  ,1: ,:  ,:,2] * (-flux_array[:  ,1: ,:  ,:,2] + flux[:  ,1: ,:  ,:]) - net_current[:  ,1: ,:  ,:,2] / (dx * dz)) / (flux_array[:  ,1: ,:  ,:,2] + flux[:  ,1: ,:  ,:])
        sdc_corr_od[:  ,:-1,:  ,:,3] = (-sdc[:  ,:-1,:  ,:,3] * ( flux_array[:  ,:-1,:  ,:,3] - flux[:  ,:-1,:  ,:]) - net_current[:  ,:-1,:  ,:,3] / (dx * dz)) / (flux_array[:  ,:-1,:  ,:,3] + flux[:  ,:-1,:  ,:])
        sdc_corr_od[1: ,:  ,:  ,:,4] = (-sdc[1: ,:  ,:  ,:,4] * (-flux_array[1: ,:  ,:  ,:,4] + flux[1: ,:  ,:  ,:]) - net_current[1: ,:  ,:  ,:,4] / (dx * dy)) / (flux_array[1: ,:  ,:  ,:,4] + flux[1: ,:  ,:  ,:])
        sdc_corr_od[:-1,:  ,:  ,:,5] = (-sdc[:-1,:  ,:  ,:,5] * ( flux_array[:-1,:  ,:  ,:,5] - flux[:-1,:  ,:  ,:]) - net_current[:-1,:  ,:  ,:,5] / (dx * dy)) / (flux_array[:-1,:  ,:  ,:,5] + flux[:-1,:  ,:  ,:])

        # Recompute the currents to check that we computed the coeffs correctly
        compute_currents = np.zeros((nz, ny, nx, ng, 6))
        compute_currents[..., 0] = - sdc[..., 0] * (-flux_array[..., 0] + flux) - sdc_corr[..., 0] * (flux_array[..., 0] + flux)
        compute_currents[..., 1] = - sdc[..., 1] * ( flux_array[..., 1] - flux) - sdc_corr[..., 1] * (flux_array[..., 1] + flux)
        compute_currents[..., 2] = - sdc[..., 2] * (-flux_array[..., 2] + flux) - sdc_corr[..., 2] * (flux_array[..., 2] + flux)
        compute_currents[..., 3] = - sdc[..., 3] * ( flux_array[..., 3] - flux) - sdc_corr[..., 3] * (flux_array[..., 3] + flux)
        compute_currents[..., 4] = - sdc[..., 4] * (-flux_array[..., 4] + flux) - sdc_corr[..., 4] * (flux_array[..., 4] + flux)
        compute_currents[..., 5] = - sdc[..., 5] * ( flux_array[..., 5] - flux) - sdc_corr[..., 5] * (flux_array[..., 5] + flux)

        sdc_corr[..., 0:2] *= dy * dz
        sdc_corr[..., 2:4] *= dx * dz
        sdc_corr[..., 4:6] *= dx * dy
        sdc_corr_od[..., 0:2] *= dy * dz
        sdc_corr_od[..., 2:4] *= dx * dz
        sdc_corr_od[..., 4:6] *= dx * dy

        # Reshape the diffusion coefficient array
        sdc_corr.shape = (nx*ny*nz*ng, 6)
        sdc_corr_od.shape = (nx*ny*nz*ng, 6)

        sdc_corr_diag = sdc_corr[:, 0:6:2].sum(axis=1) - sdc_corr[:, 1:6:2].sum(axis=1)
        data  = [sdc_corr_diag]
        diags = [0]

        if nx > 1:
            data.append( sdc_corr_od[ng:, 0])
            data.append(-sdc_corr_od[:-ng, 1])
            diags.append(-ng)
            diags.append(ng)
        if ny > 1:
            data.append( sdc_corr_od[nx*ng:, 2])
            data.append(-sdc_corr_od[:-nx*ng   , 3])
            diags.append(-nx*ng)
            diags.append(nx*ng)
        if nz > 1:
            data.append( sdc_corr_od[nx*ny*ng:, 4])
            data.append(-sdc_corr_od[:-nx*ny*ng, 5])
            diags.append(-nx*ny*ng)
            diags.append(nx*ny*ng)

        # Form a matrix of the surface diffusion coefficients
        sdc_corr_matrix = sps.diags(data, diags)

        return sdc_corr_matrix

    def generate_tallies_file(self, time):

        # Generate a new tallies file
        tallies_file = openmc.Tallies()

        # Get the MGXS library
        mgxs_lib = self.mgxs_lib[time]

        # Add the tallies to the file
        for mgxs in mgxs_lib.values():
            tallies_file += mgxs.tallies.values()

        # Export the tallies file to xml
        tallies_file.export_to_xml()
