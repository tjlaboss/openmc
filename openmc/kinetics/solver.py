from __future__ import division

from collections import OrderedDict
from numbers import Integral
import warnings
import os
import sys
import copy
import itertools
import subprocess
import time
from shutil import copyfile

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

import openmc
import openmc.checkvalue as cv
import openmc.mgxs
import openmc.kinetics
from openmc.kinetics.clock import TIME_POINTS
import h5py

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

    mgxs_lib_file : openmc.materials.MGXSLibrary
        MGXS Library file containing the multi-group xs for mg Monte Carlo.

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

    threads : int
        The number of OpenMP threads to use.

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

    seed : int
        The constant seed.

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
        self._mgxs_lib_file = None
        self._clock = None
        self._one_group = None
        self._energy_groups = None
        self._fine_groups = None
        self._k_crit = 1.0
        self._num_delayed_groups = 6
        self._initial_power = 1.
        self._states = OrderedDict()
        self._constant_seed = True
        self._seed = 1
        self._core_volume = 1.
        self._chi_delayed_by_delayed_group = False
        self._chi_delayed_by_mesh = False
        self._mpi_procs = 1
        self._threads = 1
        self._use_pregenerated_sps = False
        self._pregenerate_sps = False
        self._log_file = None
        self._log_file_name = 'log_file.h5'
        self._run_on_cluster = False
        self._job_file = 'job.pbs'

    @property
    def log_file(self):
        log_file = os.path.join(self.directory,
                                self.name + '/' + self.log_file_name)
        log_file = log_file.replace(' ', '-')
        return log_file

    @property
    def name(self):
        return self._name

    @property
    def log_file_name(self):
        return self._log_file_name

    @property
    def directory(self):
        return self._directory

    @property
    def use_pregenerated_sps(self):
        return self._use_pregenerated_sps

    @property
    def pregenerate_sps(self):
        return self._pregenerate_sps

    @property
    def core_volume(self):
        return self._core_volume

    @property
    def constant_seed(self):
        return self._constant_seed

    @property
    def seed(self):
        return self._seed

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
    def mgxs_lib_file(self):
        return self._mgxs_lib_file

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
    def threads(self):
        return self._threads

    @property
    def run_directory(self):
        return self.directory + '/' + self.name

    @property
    def run_on_cluster(self):
        return self._run_on_cluster

    @property
    def job_file(self):
        return self._job_file

    @name.setter
    def name(self, name):
        self._name = name

    @log_file_name.setter
    def log_file_name(self, name):
        self._log_file_name = name

    @directory.setter
    def directory(self, directory):
        self._directory = directory

    @use_pregenerated_sps.setter
    def use_pregenerated_sps(self, use_pregenerated_sps):
        self._use_pregenerated_sps = use_pregenerated_sps

    @pregenerate_sps.setter
    def pregenerate_sps(self, pregenerate_sps):
        self._pregenerate_sps = pregenerate_sps

    @core_volume.setter
    def core_volume(self, core_volume):
        self._core_volume = core_volume

    @constant_seed.setter
    def constant_seed(self, constant_seed):
        self._constant_seed = constant_seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

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

    @mgxs_lib_file.setter
    def mgxs_lib_file(self, mgxs_lib_file):
        self._mgxs_lib_file = mgxs_lib_file

    @clock.setter
    def clock(self, clock):
        self._clock = copy.deepcopy(clock)

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

    @threads.setter
    def threads(self, threads):
        self._threads = threads

    @run_on_cluster.setter
    def run_on_cluster(self, run_on_cluster):
        self._run_on_cluster = run_on_cluster

    @job_file.setter
    def job_file(self, job_file):
        self._job_file = job_file

    def job_directory(self, time_point):
        dir = self.run_directory + '/job_{:09.6f}'.format(self.clock.times[time_point])
        dir = dir.replace('.', '_')
        return dir

    def create_log_file(self):

        f = h5py.File(self.log_file, 'w')
        for mesh,name in \
            zip([self.mesh, self.unity_mesh, self.pin_cell_mesh, self.assembly_mesh],
                ['mesh', 'unity_mesh', 'pin_cell_mesh', 'assembly_mesh']):
            f.require_group(name)
            f[name].attrs['id'] = mesh.id
            f[name].attrs['name'] = mesh.name
            f[name].attrs['type'] = mesh.type
            f[name].attrs['dimension'] = mesh.dimension
            f[name].attrs['lower_left'] = mesh.lower_left
            f[name].attrs['width'] = mesh.width

        for groups,name in \
            zip([self.one_group, self.energy_groups, self.fine_groups],
                ['one_group', 'energy_groups', 'fine_groups']):
            f.require_group(name)
            f[name].attrs['group_edges'] = groups.group_edges
            f[name].attrs['num_groups'] = groups.num_groups

        f.attrs['name'] = self.name
        f.attrs['num_delayed_groups'] = self.num_delayed_groups
        f.attrs['chi_delayed_by_delayed_group'] \
            = self.chi_delayed_by_delayed_group
        f.attrs['chi_delayed_by_mesh'] = self.chi_delayed_by_mesh
        f.attrs['num_delayed_groups'] = self.num_delayed_groups
        f.attrs['num_outer_time_steps'] = self.num_outer_time_steps
        f.attrs['use_pregenerated_sps'] = self.use_pregenerated_sps
        f.attrs['constant_seed'] = self.constant_seed
        f.attrs['seed'] = self.seed
        f.attrs['core_volume'] = self.core_volume
        f.attrs['k_crit'] = self.k_crit
        f.require_group('clock')
        f['clock'].attrs['dt_outer'] = self.clock.dt_outer
        f['clock'].attrs['dt_inner'] = self.clock.dt_inner
        f.require_group('time_steps')
        f.close()

    def transfer_states(self, time_from, time_to='all'):

        if time_to == 'all':
            for time_point in TIME_POINTS:
                if time_point != time_from:
                    self.states[time_point] = copy.deepcopy(self.states[time_from])
                    self.states[time_point].time_point = time_point
        else:
            self.states[time_to] = copy.deepcopy(self.states[time_from])
            self.states[time_to].time_point = time_point

    def run_openmc(self, time_point):

        self.setup_openmc(time_point)

        # Names of the statepoint and summary files
        sp_old_name = '{}/statepoint.{}.h5'.format(self.job_directory(time_point),
                                                   self.settings_file.batches)
        sp_new_name = '{}/statepoint_{:.6f}_sec.{}.h5'\
                      .format(self.job_directory(time_point), self.clock.times[time_point],
                              self.settings_file.batches)
        sum_old_name = '{}/summary.h5'.format(self.job_directory(time_point))
        sum_new_name = '{}/summary_{:.6f}_sec.h5'.format(self.job_directory(time_point),
                                                         self.clock.times[time_point])

        # Run OpenMC
        if not self.use_pregenerated_sps:
            if self.run_on_cluster:

                # Copy job file to run directory
                cmd_str = 'cp ' + self.job_file  + ' ' + self.job_directory(time_point)
                subprocess.Popen(cmd_str, shell=True)
                time.sleep(5)

                 # Launch job
                cmd_str = 'qsub -P moose ' + self.job_file
                proc = subprocess.Popen(cmd_str, cwd=self.job_directory(time_point), stdout=subprocess.PIPE, shell=True)

                 # Get the job number
                job_name = proc.stdout.readlines()[0]
                job_number = job_name.split('.')[0]

                 # Pause and wait for run to finish
                cmd_str = 'qstat | grep ' + str(job_number)
                is_file_running = 'initially'
                elapsed_time = 0
                while (is_file_running is not ''):
                    time.sleep(10)
                    proc = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, shell=True)
                    try:
                        is_file_running = proc.stdout.readlines()[0].split()[4]
                        if (is_file_running == 'Q'):
                            print('job {} queued...'.format(job_number))
                        elif (is_file_running == 'R'):
                            elapsed_time = elapsed_time + 10
                            print('job {} running for {} s...'.format(job_number, elapsed_time))
                        else:
                            print('job {} in state {}...'.format(job_number, is_file_running))
                    except:
                        print('job {} done'.format(job_number))
                        break
            else:
                openmc.run(threads=self.threads, mpi_procs=self.mpi_procs,
                           mpi_exec='mpirun', cwd=self.job_directory(time_point))

        # Rename the statepoint and summary files
        copyfile(sp_old_name, sp_new_name)
        copyfile(sum_old_name, sum_new_name)

        # Load the summary and statepoint files
        summary_file = openmc.Summary(sum_new_name)
        statepoint_file = openmc.StatePoint(sp_new_name, False)
        statepoint_file.link_with_summary(summary_file)

        # Load mgxs library
        for mgxs in self.states[time_point].mgxs_lib.values():
            mgxs.load_from_statepoint(statepoint_file)

    def create_state(self, time_point, derived=False):

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
        state.time_point = time_point
        state.clock = self.clock
        state.k_crit = self.k_crit
        state.core_volume = self.core_volume
        state.chi_delayed_by_delayed_group = self.chi_delayed_by_delayed_group
        state.chi_delayed_by_mesh = self.chi_delayed_by_mesh
        state.log_file = self.log_file
        self.states[time_point] = state

    def compute_initial_flux(self):

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

        flux[flux < 0.] = 0.
        adjoint_flux[adjoint_flux < 0.] = 0.

        # Normalize the initial flux
        state.flux = flux
        state.adjoint_flux = adjoint_flux
        norm_factor = self.initial_power / state.core_power_density
        state.flux *= norm_factor
        state.adjoint_flux *= norm_factor
        state.k_crit = self.k_crit

        # Compute the initial precursor concentration
        state.compute_initial_precursor_concentration()

        # Copy data to all other states
        for time_point in TIME_POINTS:
            if time_point != 'START':
                if time_point in ['PREVIOUS_IN', 'FORWARD_IN']:
                    self.create_state(time_point, True)
                    self.copy_states('START', time_point)
                else:
                    self.create_state(time_point, False)
                    self.copy_states('START', time_point, True)

        # Create hdf5 log file
        self.create_log_file()
        self.states['START'].dump_outer_to_log_file()

    def copy_states(self, time_from, time_to='ALL', copy_mgxs=False):

        state_from = self.states[time_from]

        if time_to == 'ALL':
            for time_point in TIME_POINTS:
                if time_point != time_from:
                    state_to = self.states[time_point]
                    state_to.flux = copy.deepcopy(state_from.flux)
                    state_to.adjoint_flux = copy.deepcopy(state_from.adjoint_flux)
                    state_to.precursors = copy.deepcopy(state_from.precursors)
                    if time_point != 'END':
                        self.clock.times[time_point] = self.clock.times[time_from]

                    if copy_mgxs:
                        state_to.mgxs_lib = state_from.mgxs_lib
        else:
            state_from = self.states[time_from]
            state_to = self.states[time_to]
            state_to.flux = copy.deepcopy(state_from.flux)
            state_to.adjoint_flux = copy.deepcopy(state_from.adjoint_flux)
            state_to.precursors = copy.deepcopy(state_from.precursors)
            if time_to != 'END':
                self.clock.times[time_to] = self.clock.times[time_from]

            if copy_mgxs:
                state_to.mgxs_lib = state_from.mgxs_lib

    def take_outer_step(self):

        # Increment clock
        times = self.clock.times
        times['FORWARD_OUT'] += self.clock.dt_outer
        state_pre = self.states['PREVIOUS_IN']
        state_fwd = self.states['FORWARD_IN']

        # Run OpenMC on forward out state
        self.run_openmc('FORWARD_OUT')

        while (times['FORWARD_IN'] < times['FORWARD_OUT'] - 1.e-8):

            # Increment forward in time
            times['FORWARD_IN'] += self.clock.dt_inner

            # Get the transient matrix and time source
            time_source = state_pre.time_source_matrix * \
                          state_pre.flux.flatten()
            source = time_source + state_pre.decay_source(state_pre).flatten()

            # Compute the flux at the FORWARD_IN time step
            state_fwd.flux = spsolve(state_fwd.transient_matrix, source)

            # Propagate the precursors
            state_fwd.propagate_precursors(state_pre)

            # Update the values for the time step
            self.copy_states('FORWARD_IN', 'PREVIOUS_IN')

            # Dump data at FORWARD_OUT state to log file
            if (times['FORWARD_IN'] < times['FORWARD_OUT'] - 1.e-8):
                state_fwd.dump_inner_to_log_file()

            # Save the core power at FORWARD_IN
            print('t: {0:1.3f} s, P: {1:1.3e} W/cm^3, rho: {2:+1.3f} pcm'
                  ', beta_eff: {3:1.5f}, pnl: {4:1.3e} s'.\
                  format(times['FORWARD_IN'], state_fwd.core_power_density,
                         state_fwd.reactivity * 1.e5,
                         state_fwd.beta_eff, state_fwd.pnl))

        # Copy the flux, precursors, and time from FORWARD_IN to
        # FORWARD_OUT
        self.copy_states('FORWARD_IN', 'FORWARD_OUT')

        # Dump data at FORWARD_OUT state to log file
        self.states['FORWARD_OUT'].dump_outer_to_log_file()

        # Copy the flux, precursors, time, and MGXS from FORWARD_OUT to
        # PREVIOUS_OUT
        self.copy_states('FORWARD_OUT', 'PREVIOUS_OUT', True)

    def compute_eigenvalue(self, A, M, flux):

        # Ensure flux is a 1D array
        flux = flux.flatten()

        # Compute the initial source
        old_source = M * flux
        norm = old_source.mean()
        old_source /= norm
        flux /= norm
        k_eff = 1.0

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

            if residual < 1.e-6 and i > 10:
                break

        return flux, k_eff

    def generate_tallies_file(self, time_point):

        # Generate a new tallies file
        tallies_file = openmc.Tallies()

        # Get the MGXS library
        mgxs_lib = self.states[time_point].mgxs_lib

        # Add the tallies to the file
        for mgxs in mgxs_lib.values():
            tallies = mgxs.tallies.values()
            for tally in tallies:
                tallies_file.append(tally, True)

        # Export the tallies file to xml
        tallies_file.export_to_xml(self.job_directory(time_point) + '/tallies.xml')

    def solve(self):

        # Create run directory
        if not os.path.exists(self.run_directory):
            os.makedirs(self.run_directory)

        # Create states and run initial OpenMC on initial state
        self.create_state('START')

        if self.pregenerate_sps:
            self.run_openmc_all()
            self.use_pregenerated_sps = True

        # Compute the initial steady state flux
        self.compute_initial_flux()

        # Solve the transient
        for i in range(self.num_outer_time_steps):
            self.take_outer_step()

    def setup_openmc(self, time_point):

        # Create job directory
        if not os.path.exists(self.job_directory(time_point)):
            os.makedirs(self.job_directory(time_point))

        # Create a new random seed for the xml file
        if self.constant_seed:
            self.settings_file.seed = self.seed
        else:
            self.settings_file.seed = np.random.randint(1, 1e6, 1)[0]

        if self.mgxs_lib_file:
            self.materials_file.cross_sections = './mgxs.h5'
            self.mgxs_lib_file.export_to_hdf5(self.job_directory(time_point) + '/mgxs.h5')
            self.settings_file.energy_mode = 'multi-group'

        # Create MGXS
        self.states[time_point].initialize_mgxs()

        # Create the xml files
        self.geometry.time = self.clock.times[time_point]
        self.geometry.export_to_xml(self.job_directory(time_point) + '/geometry.xml')
        self.materials_file.export_to_xml(self.job_directory(time_point) + '/materials.xml')
        self.settings_file.export_to_xml(self.job_directory(time_point) + '/settings.xml')
        self.generate_tallies_file(time_point)

    @property
    def num_outer_time_steps(self):
        return int(round((self.clock.times['END'] - self.clock.times['START']) \
                             / self.clock.dt_outer))

    def run_openmc_all(self):

        start_time = self.clock.times['START']

        # Launch jobs
        jobs = []
        for i in range(self.num_outer_time_steps + 1):

            self.setup_openmc('START')
            job = openmc.kinetics.Job()
            job.job_directory = self.job_directory('START')
            job.mpi_procs = self.mpi_procs
            job.job_file = self.job_file
            job.launch()
            jobs.append(job)

            self.clock.times['START'] += self.clock.dt_outer

        # Check for jobs completion
        jobs_running = True
        jobs_status = {}
        elapsed_time = 0
        while jobs_running:
            jobs_status['DONE'] = 0
            jobs_status['Q'] = 0
            jobs_status['R'] = 0
            jobs_status['OTHER'] = 0

            # Get status of all jobs
            for job in jobs:
                jobs_status[job.status()] += 1

            print('Jobs ["Queued": {}, "Running": {}, "Other": {}, "Done": {}] after {} seconds'.\
                      format(jobs_status['Q'], jobs_status['R'],
                             jobs_status['OTHER'], jobs_status['DONE'], elapsed_time))

            if jobs_status['DONE'] == len(jobs):
                jobs_running = False
            else:
                time.sleep(10)
                elapsed_time += 10

        self.clock.times['START'] = start_time
