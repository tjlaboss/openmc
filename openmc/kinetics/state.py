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
    shape_mesh : openmc.mesh.Mesh
        Mesh by which shape is computed on.

    amplitude_mesh : openmc.mesh.Mesh
        Mesh by which amplitude is computed on.

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
        self._shape_mesh = None
        self._amplitude_mesh = None
        self._pin_mesh = None
        self._unity_mesh = None

        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._num_delayed_groups = 6

        self._amplitude = None
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
    def shape_mesh(self):
        return self._shape_mesh

    @property
    def amplitude_mesh(self):
        return self._amplitude_mesh

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
    def amplitude(self):
        self._amplitude.shape = (self.amplitude_nxyz, self.ng)
        return self._amplitude

    @property
    def adjoint_flux(self):
        self._adjoint_flux.shape = (self.shape_nxyz, self.ng)
        return self._adjoint_flux

    @property
    def precursors(self):
        self._precursors.shape = (self.shape_nxyz, self.nd)
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

    @shape_mesh.setter
    def shape_mesh(self, mesh):
        self._shape_mesh = mesh

    @amplitude_mesh.setter
    def amplitude_mesh(self, mesh):
        self._amplitude_mesh = mesh

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

    @amplitude.setter
    def amplitude(self, amplitude):
        self._amplitude = copy.deepcopy(amplitude)
        self._amplitude.shape = (self.amplitude_nxyz, self.ng)

    @adjoint_flux.setter
    def adjoint_flux(self, adjoint_flux):
        self._adjoint_flux = copy.deepcopy(adjoint_flux)
        self._adjoint_flux.shape = (self.shape_nxyz, self.ng)

    @precursors.setter
    def precursors(self, precursors):
        self._precursors = copy.deepcopy(precursors)
        self._precursors.shape = (self.shape_nxyz, self.nd)

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
    def shape_dimension(self):
        return tuple(self.shape_mesh.dimension[::-1])

    @property
    def pin_dimension(self):
        return tuple(self.pin_mesh.dimension[::-1])

    @property
    def amplitude_dimension(self):
        return tuple(self.amplitude_mesh.dimension[::-1])

    @property
    def shape_zyxg(self):
        return self.shape_dimension + (self.ng,)

    @property
    def pin_zyxg(self):
        return self.pin_dimension + (self.ng,)

    @property
    def amplitude_zyxg(self):
        return self.amplitude_dimension + (self.ng,)

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
    def dt_inter(self):
        return self.clock.dt_inter

    @property
    def dt_outer(self):
        return self.clock.dt_outer

    @property
    def shape_nxyz(self):
        return np.prod(self.shape_dimension)

    @property
    def amplitude_nxyz(self):
        return np.prod(self.amplitude_dimension)

    @property
    def pin_nxyz(self):
        return np.prod(self.pin_dimension)

    @property
    def shape_dxyz(self):
        return np.prod(self.shape_mesh.width)

    @property
    def amplitude_dxyz(self):
        return np.prod(self.amplitude_mesh.width)

    @property
    def pin_dxyz(self):
        return np.prod(self.pin_mesh.width)

    @property
    def flux(self):
        amplitude = self.amplitude
        amplitude.shape = self.amplitude_zyxg
        fine_amp = openmc.kinetics.map_array(amplitude, self.shape_zyxg, normalize=True)
        return fine_amp.reshape((self.shape_nxyz, self.ng)) * self.shape

    @property
    def power(self):
        return (self.shape_dxyz * self.kappa_fission * self.flux / self.k_crit).sum(axis=1)

    @property
    def core_power_density(self):
        mesh_volume = self.shape_dxyz * self.shape_nxyz
        return self.power.sum() * mesh_volume / self.core_volume

    @property
    def pin_flux(self):
        flux = self.flux
        flux.shape = self.shape_zyxg
        flux = openmc.kinetics.map_array(flux, self.pin_zyxg, normalize=True)
        return flux.reshape((self.pin_nxyz, self.ng)) * self.pin_shape

    @property
    def pin_power(self):
        power = (self.pin_dxyz * self.pin_kappa_fission * self.pin_flux / self.k_crit).sum(axis=1)
        pin_core_power = power.sum() * (self.pin_dxyz * self.pin_nxyz) / self.core_volume
        return power * self.core_power_density / pin_core_power


@add_metaclass(ABCMeta)
class ShapeState(State):

    def __init__(self, states):
        super(ShapeState, self).__init__(states)

        # Initialize Solver class attributes
        self._shape = None

    @property
    def shape(self):
        self._shape.shape = (self.shape_nxyz, self.ng)
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = copy.deepcopy(shape)
        self._shape.shape = (self.shape_nxyz, self.ng)

    @property
    def delayed_production(self):
        chi_delayed = np.repeat(self.chi_delayed, self.ng)
        chi_delayed.shape = (self.shape_nxyz, self.nd, self.ng, self.ng)
        delayed_nu_fission = np.tile(self.delayed_nu_fission, self.ng)
        delayed_nu_fission.shape = (self.shape_nxyz, self.nd, self.ng, self.ng)
        return (chi_delayed * delayed_nu_fission)

    @property
    def prompt_production(self):
        chi_prompt = np.repeat(self.chi_prompt, self.ng)
        chi_prompt.shape = (self.shape_nxyz, self.ng, self.ng)
        prompt_nu_fission = np.tile(self.prompt_nu_fission, self.ng)
        prompt_nu_fission.shape = (self.shape_nxyz, self.ng, self.ng)
        return (chi_prompt * prompt_nu_fission)

    def coupling_matrix(self, collapse=True):

        diags, dc_linear_data, dc_nonlinear_data = self.coupling_terms

        # Get the dimensions of the mesh
        nz , ny , nx  = self.shape_dimension
        nza, nya, nxa = self.amplitude_dimension
        dx , dy , dz  = self.shape_mesh.width
        ng            = self.ng

        if collapse:

            # Get the shape
            shape       = self.shape
            shape.shape = self.shape_zyxg

            # Create an array of the neighbor cell shapes
            shape_nbr = np.zeros(self.shape_zyxg + (6,))
            shape_nbr[:  , :  , 1: , :, 0] = shape[:  , :  , :-1, :]
            shape_nbr[:  , :  , :-1, :, 1] = shape[:  , :  , 1: , :]
            shape_nbr[:  , 1: , :  , :, 2] = shape[:  , :-1, :  , :]
            shape_nbr[:  , :-1, :  , :, 3] = shape[:  , 1: , :  , :]
            shape_nbr[1: , :  , :  , :, 4] = shape[:-1, :  , :  , :]
            shape_nbr[:-1, :  , :  , :, 5] = shape[1: , :  , :  , :]

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
                    dc_linear_data[-1].shape    = self.shape_zyxg
                    dc_nonlinear_data[-1].shape = self.shape_zyxg
                    dc_linear_data[-1]          = openmc.kinetics.map_array(dc_linear_data[-i], coarse_shape, normalize=False).flatten()
                    dc_nonlinear_data[-1]       = openmc.kinetics.map_array(dc_nonlinear_data[-i], coarse_shape, normalize=False).flatten()
                    diags = [0]
                elif diag == -ng:

                    # Shape weight
                    dc_linear    = dc_linear_data_copy[i]    * shape_nbr[ng:, 0]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[ng:, 0]

                    # Elongate
                    dc_linear    = np.append(np.zeros(ng), dc_linear)
                    dc_nonlinear = np.append(np.zeros(ng), dc_nonlinear)

                    # Reshape
                    dc_linear.shape    = self.shape_zyxg
                    dc_nonlinear.shape = self.shape_zyxg

                    # Extract
                    ind_diag     = sum([range(i*nx/nxa + 1, (i+1)*nx/nxa) for i in range(nxa)], [])
                    ind_off_diag = sum([range(i*nx/nxa    , i*nx/nxa + 1) for i in range(nxa)], [])
                    dc_linear_diag        = dc_linear     [:, :, ind_diag    , :]
                    dc_linear_off_diag    = dc_linear     [:, :, ind_off_diag, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, :, ind_diag    , :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, :, ind_off_diag, :]

                    # Condense and add values to diag
                    if nx != nxa:
                        dc_linear_diag        = openmc.kinetics.map_array(dc_linear_diag   , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag, coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]    += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nxa > 1:
                        dc_linear_off_diag    = openmc.kinetics.map_array(dc_linear_off_diag   , coarse_shape, normalize=False).flatten()[ng:]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[ng:]
                        dc_linear_data   .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags            .append(-ng)

                elif diag == ng:

                    # Shape weight
                    dc_linear    = dc_linear_data_copy[i]    * shape_nbr[:-ng, 1]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[:-ng, 1]

                    # Elongate
                    dc_linear    = np.append(dc_linear   , np.zeros(ng))
                    dc_nonlinear = np.append(dc_nonlinear, np.zeros(ng))

                    # Reshape
                    dc_linear.shape    = self.shape_zyxg
                    dc_nonlinear.shape = self.shape_zyxg

                    # Extract
                    ind_diag     = sum([range(i*nx/nxa      , (i+1)*nx/nxa-1) for i in range(nxa)], [])
                    ind_off_diag = sum([range((i+1)*nx/nxa-1, (i+1)*nx/nxa  ) for i in range(nxa)], [])
                    dc_linear_diag        = dc_linear   [:, :, ind_diag    , :]
                    dc_linear_off_diag    = dc_linear   [:, :, ind_off_diag, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, :, ind_diag    , :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, :, ind_off_diag, :]

                    # Condense and add values to diag
                    if nx != nxa:
                        dc_linear_diag        = openmc.kinetics.map_array(dc_linear_diag   , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag, coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]    += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nxa > 1:
                        dc_linear_off_diag    = openmc.kinetics.map_array(dc_linear_off_diag   , coarse_shape, normalize=False).flatten()[:-ng]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[:-ng]
                        dc_linear_data   .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags            .append(ng)

                elif diag == -ng*nx:

                    # Shape weight
                    dc_linear    = dc_linear_data_copy[i]    * shape_nbr[nx*ng:, 2]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[nx*ng:, 2]

                    # Elongate
                    dc_linear    = np.append(np.zeros(nx*ng), dc_linear)
                    dc_nonlinear = np.append(np.zeros(nx*ng), dc_nonlinear)

                    # Reshape
                    dc_linear.shape    = self.shape_zyxg
                    dc_nonlinear.shape = self.shape_zyxg

                    # Extract
                    ind_diag     = sum([range(i*ny/nya + 1, (i+1)*ny/nya) for i in range(nya)], [])
                    ind_off_diag = sum([range(i*ny/nya    , i*ny/nya + 1) for i in range(nya)], [])
                    dc_linear_diag        = dc_linear   [:, ind_diag    , :, :]
                    dc_linear_off_diag    = dc_linear   [:, ind_off_diag, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, ind_diag    , :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, ind_off_diag, :, :]

                    # Condense and add values to diag
                    if ny != nya:
                        dc_linear_diag        = openmc.kinetics.map_array(dc_linear_diag   , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag, coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]    += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nya > 1:
                        dc_linear_off_diag    = openmc.kinetics.map_array(dc_linear_off_diag   , coarse_shape, normalize=False).flatten()[nxa*ng:]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[nxa*ng:]
                        dc_linear_data   .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags            .append(-nxa*ng)

                elif diag == ng*nx:

                    # Shape weight
                    dc_linear    = dc_linear_data_copy[i]    * shape_nbr[:-nx*ng, 3]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[:-nx*ng, 3]

                    # Elongate
                    dc_linear    = np.append(dc_linear   , np.zeros(nx*ng))
                    dc_nonlinear = np.append(dc_nonlinear, np.zeros(nx*ng))

                    # Reshape
                    dc_linear.shape    = self.shape_zyxg
                    dc_nonlinear.shape = self.shape_zyxg

                    # Extract
                    ind_diag     = sum([range(i*ny/nya      , (i+1)*ny/nya-1) for i in range(nya)], [])
                    ind_off_diag = sum([range((i+1)*ny/nya-1, (i+1)*ny/nya  ) for i in range(nya)], [])
                    dc_linear_diag        = dc_linear   [:, ind_diag    , :, :]
                    dc_linear_off_diag    = dc_linear   [:, ind_off_diag, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[:, ind_diag    , :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[:, ind_off_diag, :, :]

                    # Condense and add values to diag
                    if ny != nya:
                        dc_linear_diag        = openmc.kinetics.map_array(dc_linear_diag   , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag, coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]    += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nya > 1:
                        dc_linear_off_diag    = openmc.kinetics.map_array(dc_linear_off_diag   , coarse_shape, normalize=False).flatten()[:-nxa*ng]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[:-nxa*ng]
                        dc_linear_data   .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags            .append(nxa*ng)

                elif diag == -ng*nx*ny:

                    # Shape weight
                    dc_linear    = dc_linear_data_copy[i]    * shape_nbr[ny*nx*ng:, 4]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[ny*nx*ng:, 4]

                    # Elongate
                    dc_linear    = np.append(np.zeros(ny*nx*ng), dc_linear)
                    dc_nonlinear = np.append(np.zeros(ny*nx*ng), dc_nonlinear)

                    # Reshape
                    dc_linear.shape    = self.shape_zyxg
                    dc_nonlinear.shape = self.shape_zyxg

                    # Extract
                    ind_diag     = sum([range(i*nz/nza + 1, (i+1)*nz/nza) for i in range(nza)], [])
                    ind_off_diag = sum([range(i*nz/nza    , i*nz/nza + 1) for i in range(nza)], [])
                    dc_linear_diag        = dc_linear   [ind_diag    , :, :, :]
                    dc_linear_off_diag    = dc_linear   [ind_off_diag, :, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[ind_diag    , :, :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[ind_off_diag, :, :, :]

                    # Condense and add values to diag
                    if nz != nza:
                        dc_linear_diag        = openmc.kinetics.map_array(dc_linear_diag   , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag, coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]    += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nza > 1:
                        dc_linear_off_diag    = openmc.kinetics.map_array(dc_linear_off_diag   , coarse_shape, normalize=False).flatten()[nya*nxa*ng:]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[nya*nxa*ng:]
                        dc_linear_data   .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags            .append(-nxa*ng)

                elif diag == ng*nx*ny:

                    # Shape weight
                    dc_linear    = dc_linear_data_copy[i]    * shape_nbr[:-ny*nx*ng, 5]
                    dc_nonlinear = dc_nonlinear_data_copy[i] * shape_nbr[:-ny*nx*ng, 5]

                    # Elongate
                    dc_linear    = np.append(dc_linear   , np.zeros(ny*nx*ng))
                    dc_nonlinear = np.append(dc_nonlinear, np.zeros(ny*nx*ng))

                    # Reshape
                    dc_linear.shape    = self.shape_zyxg
                    dc_nonlinear.shape = self.shape_zyxg

                    # Extract
                    ind_diag     = sum([range(i*nz/nza      , (i+1)*nz/nza-1) for i in range(nza)], [])
                    ind_off_diag = sum([range((i+1)*nz/nza-1, (i+1)*nz/nza  ) for i in range(nza)], [])
                    dc_linear_diag        = dc_linear   [ind_diag    , :, :, :]
                    dc_linear_off_diag    = dc_linear   [ind_off_diag, :, :, :]
                    dc_nonlinear_diag     = dc_nonlinear[ind_diag    , :, :, :]
                    dc_nonlinear_off_diag = dc_nonlinear[ind_off_diag, :, :, :]

                    # Condense and add values to diag
                    if nz != nza:
                        dc_linear_diag        = openmc.kinetics.map_array(dc_linear_diag   , coarse_shape, normalize=False).flatten()
                        dc_nonlinear_diag     = openmc.kinetics.map_array(dc_nonlinear_diag, coarse_shape, normalize=False).flatten()
                        dc_linear_data[0]    += dc_linear_diag.flatten()
                        dc_nonlinear_data[0] += dc_nonlinear_diag.flatten()

                    # Condense and add values to off diag
                    if nza > 1:
                        dc_linear_off_diag    = openmc.kinetics.map_array(dc_linear_off_diag   , coarse_shape, normalize=False).flatten()[:-nya*nxa*ng]
                        dc_nonlinear_off_diag = openmc.kinetics.map_array(dc_nonlinear_off_diag, coarse_shape, normalize=False).flatten()[:-nya*nxa*ng]
                        dc_linear_data   .append(dc_linear_off_diag)
                        dc_nonlinear_data.append(dc_nonlinear_off_diag)
                        diags            .append(nya*nxa*ng)

        # Form a matrix of the surface diffusion coefficients corrections
        dc_linear_matrix    = sps.diags(dc_linear_data   , diags)
        dc_nonlinear_matrix = sps.diags(dc_nonlinear_data, diags)

        return dc_linear_matrix, dc_nonlinear_matrix


class OuterState(ShapeState):

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

        self._inscatter.shape = (self.shape_nxyz, self.ng, self.ng)
        return self._inscatter

    @property
    def outscatter(self):
        return self.inscatter.sum(axis=1)

    @property
    def absorption(self):
        if not self.mgxs_loaded:
            self._absorption = self.mgxs_lib['absorption'].get_xs()

        self._absorption.shape = (self.shape_nxyz, self.ng)
        return self._absorption

    @property
    def kappa_fission(self):
        if not self.mgxs_loaded:
            self._kappa_fission = self.mgxs_lib['kappa-fission'].get_xs()

        self._kappa_fission.shape = (self.shape_nxyz, self.ng)
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

        self._chi_prompt.shape = (self.shape_nxyz, self.ng)
        return self._chi_prompt

    @property
    def prompt_nu_fission(self):
        if not self.mgxs_loaded:
            self._prompt_nu_fission = self.mgxs_lib['prompt-nu-fission'].get_xs()

        self._prompt_nu_fission.shape = (self.shape_nxyz, self.ng)
        return self._prompt_nu_fission

    @property
    def chi_delayed(self):

        if not self.mgxs_loaded:
            self._chi_delayed = self.mgxs_lib['chi-delayed'].get_xs()

            if self.chi_delayed_by_mesh:
                if not self.chi_delayed_by_delayed_group:
                    self._chi_delayed.shape = (self.shape_nxyz, self.ng)
                    self._chi_delayed = np.tile(self._chi_delayed, self.nd)
            else:
                if self.chi_delayed_by_delayed_group:
                    self._chi_delayed = np.tile(self._chi_delayed.flatten(), self.shape_nxyz)
                else:
                    self._chi_delayed = np.tile(self._chi_delayed.flatten(), self.shape_nxyz)
                    self._chi_delayed.shape = (self.shape_nxyz, self.ng)
                    self._chi_delayed = np.tile(self._chi_delayed, self.nd)

        self._chi_delayed.shape = (self.shape_nxyz, self.nd, self.ng)
        return self._chi_delayed

    @property
    def delayed_nu_fission(self):
        if not self.mgxs_loaded:
            self._delayed_nu_fission = self.mgxs_lib['delayed-nu-fission'].get_xs()

        self._delayed_nu_fission.shape = (self.shape_nxyz, self.nd, self.ng)
        return self._delayed_nu_fission

    @property
    def inverse_velocity(self):
        if not self.mgxs_loaded:
            self._inverse_velocity = self.mgxs_lib['inverse-velocity'].get_xs()

        self._inverse_velocity.shape = (self.shape_nxyz, self.ng)
        return self._inverse_velocity

    @property
    def decay_rate(self):
        if not self.mgxs_loaded:
            self._decay_rate = self.mgxs_lib['decay-rate'].get_xs()
            self._decay_rate[self._decay_rate < 1.e-5] = 0.

        self._decay_rate.shape = (self.shape_nxyz, self.nd)
        return self._decay_rate

    @property
    def flux_tallied(self):
        if not self.mgxs_loaded:
            self._flux_tallied = self.mgxs_lib['kappa-fission'].tallies['flux'].get_values()
            self._flux_tallied.shape = (self.shape_nxyz, self.ng)
            self._flux_tallied = self._flux_tallied[:, ::-1]

        self._flux_tallied.shape = (self.shape_nxyz, self.ng)
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

        self._diffusion_coefficient.shape = (self.shape_nxyz, self.ng)
        return self._diffusion_coefficient

    @property
    def flux_frequency(self):

        state_pre = self.states['PREVIOUS_OUTER']

        freq = (1. / self.dt_outer - state_pre.flux / self.flux / self.dt_outer)
        freq = openmc.kinetics.nan_inf_to_zero(freq)

        freq.shape = self.shape_dimension + (self.ng,)
        coarse_shape = (1,1,1,self.ng)
        freq = openmc.kinetics.map_array(freq, coarse_shape, normalize=True)
        freq.shape = (1, self.ng)
        return freq

    @property
    def precursor_frequency(self):

        flux = np.tile(self.flux, self.nd)
        flux.shape = (self.shape_nxyz, self.nd, self.ng)
        del_fis_rate = self.delayed_nu_fission * flux
        freq = del_fis_rate.sum(axis=2) / self.precursors / self.k_crit * self.shape_dxyz - self.decay_rate

        freq = self.decay_rate / (freq + self.decay_rate)
        freq = openmc.kinetics.nan_inf_to_zero(freq)

        return freq

    def extract_shape(self, flux, power):
        amplitude       = self.amplitude
        amplitude.shape = self.amplitude_zyxg
        fine_amp        = openmc.kinetics.map_array(amplitude, self.shape_zyxg, normalize=True)

        fine_amp.shape  = (self.shape_nxyz, self.ng)
        flux.shape      = (self.shape_nxyz, self.ng)

        # Compute and normalize shape
        self.shape     = flux / fine_amp
        self.shape     = self.shape * power / self.core_power_density

    @property
    def destruction_matrix(self):

        linear, non_linear = self.coupling_matrix(False)
        inscatter       = self.inscatter * self.shape_dxyz
        absorb_outscat  = self.outscatter + self.absorption
        absorb_outscat  = absorb_outscat * self.shape_dxyz
        inscatter.shape = (self.shape_nxyz, self.ng, self.ng)
        total = sps.diags([absorb_outscat.flatten()], [0]) - openmc.kinetics.block_diag(inscatter)

        if self.method == 'OMEGA':
            flux_frequency = self.flux_frequency
            flux_frequency.shape = (1,1,1,self.ng)
            coarse_shape = self.shape_dimension + (self.ng,)
            flux_frequency = openmc.kinetics.map_array(flux_frequency, coarse_shape, True)
            flux_frequency.shape = (self.shape_nxyz, self.ng)
            flux_frequency = flux_frequency * self.inverse_velocity * self.shape_dxyz
            total = total + sps.diags([flux_frequency.flatten()], [0])

        return total + linear + non_linear

    @property
    def delayed_production_matrix(self):

        if self.method == 'OMEGA':
            freq = np.repeat(self.precursor_frequency, self.ng)
            freq.shape = (self.shape_nxyz, self.nd, self.ng)
            chi_delayed = self.chi_delayed * freq
            chi_delayed = np.repeat(chi_delayed, self.ng)
            chi_delayed.shape = (self.shape_nxyz, self.nd, self.ng, self.ng)
            delayed_nu_fission = np.tile(self.delayed_nu_fission, self.ng)
            delayed_nu_fission.shape = (self.shape_nxyz, self.nd, self.ng, self.ng)
            delayed_production = (chi_delayed * delayed_nu_fission).sum(axis=1) * self.shape_dxyz
        else:
            delayed_production = self.delayed_production.sum(axis=1) * self.shape_dxyz

        delayed_production = openmc.kinetics.block_diag(delayed_production)
        return delayed_production / self.k_crit

    @property
    def production_matrix(self):
        return self.prompt_production_matrix + self.delayed_production_matrix

    @property
    def prompt_production_matrix(self):
        return openmc.kinetics.block_diag(self.prompt_production * self.shape_dxyz / self.k_crit)

    @property
    def pin_shape(self):

        # Normalize the power mesh flux to the shape mesh
        pm_flux = self.pin_flux_tallied
        pm_flux.shape = self.pin_zyxg
        sm_amp = openmc.kinetics.map_array(pm_flux, self.shape_zyxg, normalize=True)
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
        del_fis_rate.shape = (self.shape_nxyz, self.nd, self.ng)
        precursors = del_fis_rate.sum(axis=2) / self.decay_rate / self.k_crit * self.shape_dxyz
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

    def extract_shape(self, flux, power):
        amplitude       = self.amplitude
        amplitude.shape = self.amplitude_zyxg
        fine_amp        = openmc.kinetics.map_array(amplitude, self.shape_zyxg, normalize=True)
        fine_amp.shape  = (self.shape_nxyz, self.ng)
        flux.shape      = (self.shape_nxyz, self.ng)

        # Compute and normalize shape
        self.shape     = flux / fine_amp
        self.shape     = self.shape * power / self.core_power_density

    @property
    def coupling_terms(self):

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
        net_current[..., 0:2]  = net_current[..., 0:2] / (dy * dz)
        net_current[..., 2:4]  = net_current[..., 2:4] / (dx * dz)
        net_current[..., 4:6]  = net_current[..., 4:6] / (dx * dy)

        # Get the flux
        flux = copy.deepcopy(self.flux_tallied)
        flux.shape = self.shape_zyxg

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
        dc.shape = self.shape_zyxg
        dc_nbr   = np.zeros(self.shape_zyxg + (6,))

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
        flux_array.shape = self.shape_zyxg + (6,)

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
        dc_nonlinear_copy.shape = self.shape_zyxg + (6,)
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


class InterState(ShapeState):

    def __init__(self, states):
        super(InterState, self).__init__(states)

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

    def destruction_matrix(self, collapse=True):

        linear, non_linear = self.coupling_matrix(collapse)
        inscatter          = self.inscatter.flatten() * self.shape_dxyz
        absorb_outscat     = self.outscatter.flatten() + self.absorption.flatten()
        absorb_outscat     = absorb_outscat * self.shape_dxyz

        if collapse:
            inscatter            = inscatter * np.tile(self.shape, self.ng).flatten()
            absorb_outscat       = absorb_outscat * self.shape.flatten()
            inscatter.shape      = self.shape_dimension + (self.ng, self.ng)
            absorb_outscat.shape = self.shape_zyxg
            coarse_shape         = self.amplitude_dimension + (self.ng, self.ng)
            inscatter            = openmc.kinetics.map_array(inscatter, coarse_shape, normalize=False)
            absorb_outscat       = openmc.kinetics.map_array(absorb_outscat, self.amplitude_zyxg, normalize=False)
            inscatter.shape      = (self.amplitude_nxyz, self.ng, self.ng)
        else:
            inscatter.shape = (self.shape_nxyz, self.ng, self.ng)

        total = sps.diags([absorb_outscat.flatten()], [0]) - openmc.kinetics.block_diag(inscatter)
        return total + linear + non_linear

    def prompt_production_matrix(self, collapse=True):
        prompt_production = self.prompt_production * self.shape_dxyz

        if collapse:
            shape = np.tile(self.shape, self.ng).reshape((self.shape_nxyz, self.ng, self.ng))
            prompt_production = prompt_production * shape
            prompt_production.shape = self.shape_dimension + (self.ng, self.ng)
            coarse_shape = self.amplitude_dimension + (self.ng, self.ng)
            prompt_production = openmc.kinetics.map_array(prompt_production, coarse_shape, normalize=False)
            prompt_production.shape = (self.amplitude_nxyz, self.ng, self.ng)

        prompt_production = openmc.kinetics.block_diag(prompt_production)
        return prompt_production / self.k_crit

    def delayed_production_matrix(self, collapse=True):

        delayed_production = self.delayed_production.sum(axis=1) * self.shape_dxyz

        if collapse:
            shape = np.tile(self.shape, self.ng).flatten()
            delayed_production = delayed_production.flatten() * shape
            delayed_production.shape = self.shape_dimension + (self.ng, self.ng)
            coarse_shape = self.amplitude_dimension + (self.ng, self.ng)
            delayed_production = openmc.kinetics.map_array(delayed_production, coarse_shape, normalize=False)
            delayed_production.shape = (self.amplitude_nxyz, self.ng, self.ng)

        delayed_production = openmc.kinetics.block_diag(delayed_production)
        return delayed_production / self.k_crit

    @property
    def time_removal_source(self):
        source = self.time_removal_matrix(False) * self.states['PREVIOUS_INTER'].flux.flatten()

        amp_ratio = self.amplitude / self.states['PREVIOUS_INTER'].amplitude
        amp_ratio.shape = self.amplitude_zyxg
        fine_amp_ratio = openmc.kinetics.map_array(amp_ratio, self.shape_zyxg, True)
        fine_amp_ratio = openmc.kinetics.nan_inf_to_zero(amp_ratio)
        source *= fine_amp_ratio.flatten()
        return source

    @property
    def decay_source(self):
        decay_source = self.decay_rate * self.precursors
        decay_source = np.repeat(decay_source, self.ng)
        decay_source.shape = (self.shape_nxyz, self.nd, self.ng)
        decay_source *= self.chi_delayed
        return decay_source.sum(axis=1).flatten()

    def time_removal_matrix(self, collapse=True):

        if collapse:
            time_removal = self.inverse_velocity / self.dt_inner * self.shape_dxyz
            time_removal *= self.shape
            time_removal.shape = self.shape_zyxg
            time_removal = openmc.kinetics.map_array(time_removal, self.amplitude_zyxg, False)
        else:
            time_removal = self.inverse_velocity / self.dt_inter * self.shape_dxyz

        return sps.diags([time_removal.flatten()], [0])

    def production_matrix(self, collapse=True):
        return self.prompt_production_matrix(collapse) + self.delayed_production_matrix(collapse)

    def transient_matrix(self, collapse=True):
        return self.time_removal_matrix(collapse) + self.flux_deriv_matrix(collapse) \
            - self.prompt_production_matrix(collapse) + self.destruction_matrix(collapse)

    def flux_deriv_matrix(self, collapse=True):

        if collapse:
            state_pre = self.states['PREVIOUS_INTER']
            state_fwd = self.states['FORWARD_INTER']
            flux_deriv = (state_fwd.shape - state_pre.shape) / self.dt_inter
            flux_deriv.shape = (self.shape_nxyz, self.ng)
            flux_deriv *= self.inverse_velocity * self.shape_dxyz
            flux_deriv.shape = self.shape_zyxg
            flux_deriv = openmc.kinetics.map_array(flux_deriv, self.amplitude_zyxg, False)
        else:
            state_pre = self.states['PREVIOUS_INNER']
            state_fwd = self.states['FORWARD_INNER']
            flux_deriv = (state_fwd.amplitude - state_pre.amplitude) / (self.dt_inner * state_fwd.amplitude)
            flux_deriv.shape = self.amplitude_zyxg
            flux_deriv = openmc.kinetics.map_array(flux_deriv, self.shape_zyxg, True)
            flux_deriv.shape = (self.shape_nxyz, self.ng)
            flux_deriv *= self.shape_dxyz * self.inverse_velocity
            flux_deriv = openmc.kinetics.nan_inf_to_zero(flux_deriv)

        return sps.diags([flux_deriv.flatten()], [0])

    @property
    def dump_to_log_file(self):

        time_point = str(self.clock.times[self.time_point])
        f = h5py.File(self._log_file, 'a')
        if time_point not in f['INTER_STEPS'].keys():
            f['INTER_STEPS'].require_group(time_point)

        if 'shape' not in f['INTER_STEPS'][time_point].keys():
            f['INTER_STEPS'][time_point].create_dataset('shape', data=self.shape)
        else:
            shape = f['INTER_STEPS'][time_point]['shape']
            shape[...] = self.shape

        f.close()


class InnerState(State):

    def __init__(self, states):
        super(InnerState, self).__init__(states)

        # Initialize Solver class attributes
        self.fwd_state = states['FORWARD_INTER']
        self.pre_state = states['PREVIOUS_INTER']

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
        fwd_time = self.clock.times['FORWARD_INTER']
        weight = 1 - (fwd_time - time_point) / self.clock.dt_inter
        return weight

    @property
    def time_removal_matrix(self):
        time_removal       = self.inverse_velocity / self.dt_inner * self.shape_dxyz
        time_removal       = time_removal * self.shape
        time_removal.shape = self.shape_zyxg
        time_removal       = openmc.kinetics.map_array(time_removal, self.amplitude_zyxg, normalize=False)

        return sps.diags([time_removal.flatten()], [0])

    @property
    def shape(self):
        wgt = self.weight
        shape_fwd  = self.fwd_state.shape
        shape_prev = self.pre_state.shape
        shape = shape_fwd * wgt + shape_prev * (1 - wgt)
        shape[shape < 0.] = 0.
        return shape

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
    def decay_rate(self):
        wgt = self.weight
        decay_rate_fwd  = self.fwd_state.decay_rate
        decay_rate_prev = self.pre_state.decay_rate
        decay_rate = decay_rate_fwd * wgt + decay_rate_prev * (1 - wgt)
        decay_rate[decay_rate < 0.] = 0.
        decay_rate[decay_rate < 1.e-5] = 0.
        return decay_rate

    @property
    def inverse_velocity(self):
        wgt = self.weight
        inverse_velocity_fwd  = self.fwd_state.inverse_velocity
        inverse_velocity_prev = self.pre_state.inverse_velocity
        inverse_velocity = inverse_velocity_fwd * wgt + inverse_velocity_prev * (1 - wgt)
        inverse_velocity[inverse_velocity < 0.] = 0.
        return inverse_velocity

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
        source.shape = (self.shape_nxyz, self.nd, self.ng)
        source  = source * self.chi_delayed
        source = source.sum(axis=1)

        source.shape = self.shape_zyxg
        source = openmc.kinetics.map_array(source, self.amplitude_zyxg, normalize=False)
        source.shape = (self.amplitude_nxyz, self.ng)

        return source

    @property
    def k2_source_matrix(self):

        k2 = np.repeat(self.decay_rate * self.k2, self.ng * self.ng)
        k2.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        chi = np.repeat(self.chi_delayed, self.ng)
        chi.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        shape = np.tile(self.shape, self.nd).flatten()
        del_fis_rate = (self.delayed_nu_fission.flatten() * shape).reshape((self.shape_nxyz, self.nd, self.ng))
        del_fis_rate = np.tile(del_fis_rate, self.ng)
        del_fis_rate.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        term_k2 = chi * k2 * del_fis_rate * self.shape_dxyz
        coarse_shape = self.amplitude_dimension + (self.nd, self.ng, self.ng)
        term_k2 = openmc.kinetics.map_array(term_k2, coarse_shape, normalize=False)
        term_k2.shape = (self.amplitude_nxyz, self.nd, self.ng, self.ng)

        return openmc.kinetics.block_diag(term_k2.sum(axis=1))

    @property
    def k3_source_matrix(self):

        state = self.states['PREVIOUS_INNER']

        k3 = np.repeat(self.decay_rate * state.k3, self.ng * self.ng)
        k3.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        chi = np.repeat(self.chi_delayed, self.ng)
        chi.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        shape = np.tile(self.shape, self.nd).flatten()
        del_fis_rate = (self.delayed_nu_fission.flatten() * shape).reshape((self.shape_nxyz, self.nd, self.ng))
        del_fis_rate = np.tile(del_fis_rate, self.ng)
        del_fis_rate.shape = self.shape_dimension + (self.nd, self.ng, self.ng)

        term_k3 = chi * k3 * del_fis_rate * self.shape_dxyz
        coarse_shape = self.amplitude_dimension + (self.nd, self.ng, self.ng)
        term_k3 = openmc.kinetics.map_array(term_k3, coarse_shape, normalize=False)
        term_k3.shape = (self.amplitude_nxyz, self.nd, self.ng, self.ng)

        return openmc.kinetics.block_diag(term_k3.sum(axis=1))

    @property
    def propagate_precursors(self):

        state = self.states['PREVIOUS_INNER']

        # Contribution from current precursors
        term_k1 = state.k1 * state.precursors

        # Contribution from generation at current time point
        flux = np.tile(self.flux, self.nd)
        flux.shape = (self.shape_nxyz, self.nd, self.ng)
        term_k2 = self.k2 * (self.delayed_nu_fission * flux).sum(axis=2) * self.shape_dxyz

        # Contribution from generation at previous time step
        flux = np.tile(state.flux, state.nd)
        flux.shape = (state.shape_nxyz, state.nd, state.ng)
        term_k3 = state.k3 * (state.delayed_nu_fission * flux).sum(axis=2) * self.shape_dxyz

        self._precursors = term_k1 + term_k2 - term_k3

    @property
    def dump_to_log_file(self):

        time_point = str(self.clock.times[self.time_point])
        f = h5py.File(self._log_file, 'a')
        if time_point not in f['INNER_STEPS'].keys():
            f['INNER_STEPS'].require_group(time_point)

        if 'amplitude' not in f['INNER_STEPS'][time_point].keys():
            f['INNER_STEPS'][time_point].create_dataset('amplitude', data=self.amplitude)
            f['INNER_STEPS'][time_point].create_dataset('core_power', data=self.core_power_density)
        else:
            amplitude = f['INNER_STEPS'][time_point]['amplitude']
            amplitude[...] = self.amplitude
            core_power = f['INNER_STEPS'][time_point]['core_power']
            core_power[...] = self.core_power_density

        f.close()
