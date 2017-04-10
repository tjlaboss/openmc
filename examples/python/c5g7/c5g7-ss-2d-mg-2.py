#coding=utf-8

import math
import pickle
import matplotlib
from copy import deepcopy
from shutil import copyfile
import copy

import numpy as np
import os

import openmc
import openmc.mgxs
import openmc.plotter
import openmc.kinetics as kinetics

from geometry_2d_mg import materials, surfaces, universes, cells, lattices, geometry, mgxs_lib_file
from plots import plots
from mgxs_lib import mgxs_data

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set the base control rod bank positions
cells['Control Rod Base Bank 1'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 2'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 3'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 4'].translation = [0., 0., 64.26]

# Create the materials file
materials_file = openmc.Materials(geometry.get_all_materials().values())

case = '3.0'
omega = 1.0

# Adjust the cells to have the desired moderator densities
if case == '3.1':
    omega = 0.95
elif case == '3.2':
    omega = 0.90
elif case == '3.3':
    omega = 0.85
elif case == '3.4':
    omega = 0.8

for bank in range(1,5):
    name = 'Moderator Bank {}'.format(bank)
    d = materials[name].density
    density = np.array([[0., 1.     , 2.],
                        [d , omega*d, d ]])
    materials[name].set_density('macro', density)

# OpenMC simulation parameters
batches = 130
inactive = 30
particles = 10000000
sp_interval = 5

# Instantiate a Settings object
settings_file = openmc.Settings()
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.output = {'tallies': False}

statepoint = dict()
sp_batches = range(inactive + sp_interval, batches + sp_interval, sp_interval)
statepoint['batches'] = sp_batches
settings_file.statepoint = statepoint

# Create an initial uniform spatial source distribution over fissionable zones
source_bounds  = [-32.13, -10.71, -64.26, 10.71,  32.13,  64.26]
uniform_dist = openmc.stats.Box(source_bounds[:3], source_bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

entropy_mesh = openmc.Mesh()
entropy_mesh.type = 'regular'
entropy_mesh.dimension = [34,34,1]
entropy_mesh.lower_left  = source_bounds[:3]
entropy_mesh.upper_right = source_bounds[3:]
settings_file.entropy_mesh = entropy_mesh

# Instantiate an EnergyGroups object for the diffusion coefficients
fine_groups = openmc.mgxs.EnergyGroups()
fine_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 1.0e7]
#fine_groups.group_edges = [0., 55.6, 1.0e7]

# Instantiate an EnergyGroups object for the transient solve
energy_groups = openmc.mgxs.EnergyGroups()
energy_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 1.0e7]
#energy_groups.group_edges = [0., 0.63, 1.0e7]

# Instantiate an EnergyGroups object for one group data
one_group = openmc.mgxs.EnergyGroups()
one_group.group_edges = [fine_groups.group_edges[0], fine_groups.group_edges[-1]]

# Create pin cell mesh
point_mesh = openmc.Mesh()
point_mesh.type = 'regular'
point_mesh.dimension = [1,1,1]
point_mesh.lower_left  = [-32.13, -10.71, -64.26]
point_mesh.width = [42.84/point_mesh.dimension[0],
                    42.84/point_mesh.dimension[1],
                    128.52]

full_point_mesh = openmc.Mesh()
full_point_mesh.type = 'regular'
full_point_mesh.dimension = [1,1,1]
full_point_mesh.lower_left  = [-32.13, -32.13, -64.26]
full_point_mesh.width = [64.26/full_point_mesh.dimension[0],
                         64.26/full_point_mesh.dimension[1],
                         128.52]

pin_cell_mesh = openmc.Mesh()
pin_cell_mesh.type = 'regular'
pin_cell_mesh.dimension = [34,34,1]
pin_cell_mesh.lower_left  = [-32.13, -10.71, -64.26]
pin_cell_mesh.width = [42.84/pin_cell_mesh.dimension[0],
                       42.84/pin_cell_mesh.dimension[1],
                       128.52]

full_pin_cell_mesh = openmc.Mesh()
full_pin_cell_mesh.type = 'regular'
full_pin_cell_mesh.dimension = [51,51,1]
full_pin_cell_mesh.lower_left  = [-32.13, -32.13, -64.26]
full_pin_cell_mesh.width = [64.26/full_pin_cell_mesh.dimension[0],
                            64.26/full_pin_cell_mesh.dimension[1],
                            128.52]

assembly_mesh = openmc.Mesh()
assembly_mesh.type = 'regular'
assembly_mesh.dimension = [2,2,1]
assembly_mesh.lower_left  = [-32.13, -10.71, -64.26]
assembly_mesh.width = [42.84/assembly_mesh.dimension[0],
                       42.84/assembly_mesh.dimension[1],
                       128.52]

full_assembly_mesh = openmc.Mesh()
full_assembly_mesh.type = 'regular'
full_assembly_mesh.dimension = [3,3,1]
full_assembly_mesh.lower_left  = [-32.13, -32.13, -64.26]
full_assembly_mesh.width = [64.26/full_assembly_mesh.dimension[0],
                            64.26/full_assembly_mesh.dimension[1],
                            128.52]

# Instantiate a clock object
clock = openmc.kinetics.Clock(start=0., end=2., dt_outer=5.e-1, dt_inner=1.e-2)

# Instantiate a kinetics solver object
solver = openmc.kinetics.Solver(name='MG_SS', directory='C5G7_2D')
solver.num_delayed_groups           = 8
solver.amplitude_mesh               = full_assembly_mesh
solver.shape_mesh                   = full_pin_cell_mesh
solver.one_group                    = one_group
solver.energy_groups                = energy_groups
solver.fine_groups                  = fine_groups
solver.geometry                     = geometry
solver.settings_file                = settings_file
solver.materials_file               = materials_file
solver.inner_tolerance              = np.inf
solver.outer_tolerance              = 1.e-3
solver.mgxs_lib_file                = mgxs_lib_file
solver.method                       = 'STATIC-FLUX'
solver.multi_group                  = True
solver.clock                        = clock
solver.mpi_procs                    = 36*1
solver.threads                      = 1
solver.ppn                          = 36
solver.core_volume                  = 42.84 * 42.84 * 128.52
solver.constant_seed                = False
solver.seed                         = 1
solver.chi_delayed_by_delayed_group = True
solver.chi_delayed_by_mesh          = False
solver.use_pregenerated_sps         = False
solver.pregenerate_sps              = False
solver.run_on_cluster               = False
solver.job_file                     = 'job.pbs'
solver.log_file_name                = 'log_file_sf_1.h5'

# Solve transient problem
#solver.solve()

# Compute the flux with the statepoints at different batches
# Create run directory
if not os.path.exists(solver.run_directory):
    os.makedirs(solver.run_directory)

# Create states and run initial OpenMC on initial state
solver.create_state('START')

# Get the START state
state = solver.states['START']

time_point = 'START'

# Get a fresh copy of the settings file
settings_file = copy.deepcopy(solver.settings_file)

# Create job directory
if not os.path.exists(solver.job_directory(time_point)):
    os.makedirs(solver.job_directory(time_point))

# Create a new random seed for the xml file
if solver.constant_seed:
    settings_file.seed = solver.seed
else:
    settings_file.seed = np.random.randint(1, 1e6, 1)[0]

if solver.mgxs_lib_file:
    solver.materials_file.cross_sections = './mgxs.h5'
    solver.mgxs_lib_file.export_to_hdf5(solver.job_directory(time_point) + '/mgxs.h5')
    settings_file.energy_mode = 'multi-group'

# Create MGXS
solver.states[time_point].initialize_mgxs()

# Create the xml files
solver.geometry.time = solver.clock.times[time_point]
solver.geometry.export_to_xml(solver.job_directory(time_point) + '/geometry.xml')
solver.materials_file.export_to_xml(solver.job_directory(time_point) + '/materials.xml')
settings_file.export_to_xml(solver.job_directory(time_point) + '/settings.xml')
solver.generate_tallies_file(time_point)

# Run OpenMC
#openmc.run(threads=solver.threads, mpi_procs=solver.mpi_procs,
#           mpi_exec='mpirun', cwd=solver.job_directory(time_point))

# Loop over batches
pin_powers = []
k_effs = []
tallied_powers = []
tallied_power_errors = []
for i,b in enumerate(sp_batches):

    print('batch {}'.format(b))

    # Names of the statepoint and summary files
    if batches < 100:
        sp_old_name = '{}/statepoint.{:02d}.h5'.format(solver.job_directory(time_point), b)
        sp_new_name = '{}/statepoint_{:.6f}_sec.{:02d}.h5'\
            .format(solver.job_directory(time_point), solver.clock.times[time_point], b)
    elif batches >= 100 and batches < 1000:
        sp_old_name = '{}/statepoint.{:03d}.h5'.format(solver.job_directory(time_point), b)
        sp_new_name = '{}/statepoint_{:.6f}_sec.{:03d}.h5'\
            .format(solver.job_directory(time_point), solver.clock.times[time_point], b)
    else:
        sp_old_name = '{}/statepoint.{:04d}.h5'.format(solver.job_directory(time_point), b)
        sp_new_name = '{}/statepoint_{:.6f}_sec.{:04d}.h5'\
            .format(solver.job_directory(time_point), solver.clock.times[time_point], b)

    sum_old_name = '{}/summary.h5'.format(solver.job_directory(time_point))
    sum_new_name = '{}/summary_{:.6f}_sec.h5'.format(solver.job_directory(time_point),
                                                     solver.clock.times[time_point])

    # Rename the statepoint and summary files
    copyfile(sp_old_name, sp_new_name)
    copyfile(sum_old_name, sum_new_name)

    # Load the summary and statepoint files
    summary_file = openmc.Summary(sum_new_name)
    statepoint_file = openmc.StatePoint(sp_new_name, False)
    statepoint_file.link_with_summary(summary_file)

    # Load mgxs library
    for mgxs in solver.states[time_point].mgxs_lib.values():
        mgxs.load_from_statepoint(statepoint_file)

    # Compute the initial eigenvalue
    flux, k_crit = solver.compute_eigenvalue(state.destruction_matrix(False),
                                             state.production_matrix(False),
                                             state.flux_tallied)

    # Normalize the initial flux
    #state.k_crit = solver.k_crit

    # Get the amplitude on the fine and coarse meshes
    coarse_shape = state.amplitude_dimension + (solver.ng,)
    fine_shape   = state.shape_dimension     + (solver.ng,)
    flux.shape   = fine_shape
    coarse_amp   = openmc.kinetics.map_array(flux, coarse_shape, normalize=True)
    fine_amp     = openmc.kinetics.map_array(coarse_amp, fine_shape, normalize=True)

    # Set the unnormalized amplitude and shape
    state.amplitude = coarse_amp
    state.shape     = flux.flatten() / fine_amp.flatten()

    # Compute the power and normalize the amplitude
    norm_factor         = solver.initial_power / state.core_power_density
    state.amplitude    *= norm_factor

    # Store the pin powers
    power = state.power
    k_effs.append(k_crit)
    power[power < 1.e-6] = 0.0
    pin_powers.append(power)

    nz , ny , nx  = state.shape_dimension
    fig = plt.figure()
    ax = fig.add_subplot(111)
    power[power == 0.0] = np.nan
    power.shape = (ny, nx)
    cax = ax.imshow(power, interpolation='none', cmap='jet')
    fig.colorbar(cax)
    ax.set_title('pin powers')
    plt.savefig('pin_powers_{:03d}_batches.png'.format(b))
    plt.close()

    mesh_volume = state.dxyz * state.nxyz
    tallied_power = state.mgxs_lib['kappa-fission'].rxn_rate_tally.summation(filter_type=openmc.EnergyFilter)
    power = tallied_power.get_values(scores=['kappa-fission']).flatten()
    tallied_power_density = power.sum() * mesh_volume / state.core_volume
    power *= solver.initial_power / tallied_power_density
    power.shape = (51,51)
    tallied_powers.append(power)

    power_error = tallied_power.get_values(scores=['kappa-fission'], value='rel_err').flatten()
    power_error.shape = (51,51)
    power_error *= 100.
    tallied_power_errors.append(power_error)


ref_pin_powers = tallied_powers[-1]
ref_pin_powers[ref_pin_powers < 1.e-6] = 0.0
for i,b in enumerate(sp_batches):
    power_error = (pin_powers[i] - ref_pin_powers) / ref_pin_powers * 100.
    power_error[power_error ==  np.inf] = 0.0
    power_error[power_error == -np.inf] = 0.0
    power_error = np.nan_to_num(power_error)
    max_error = np.max(np.abs(power_error))
    power_error[power_error == 0.0] = np.nan

    nz , ny , nx  = state.shape_dimension

    fig = plt.figure()
    ax = fig.add_subplot(111)
    power_error.shape = (ny, nx)
    cax = ax.imshow(power_error, interpolation='none', cmap='jet')#, vmin=-5, vmax=5)
    fig.colorbar(cax)
    ax.set_title('power errors - max error {:1.4f} %'.format(max_error))
    plt.savefig('pin_power_errors_{:03d}_batches.png'.format(b))
    plt.close()

    pe = tallied_power_errors[i]
    pe[pe ==  np.inf] = 0.0
    pe[pe == -np.inf] = 0.0
    pe = np.nan_to_num(pe)
    pe[pe > 50.] = 0.
    max_error = np.max(np.abs(pe))
    pe[pe == 0.0] = np.nan

    fig = plt.figure()
    ax = fig.add_subplot(111)
    pe.shape = (ny, nx)
    cax = ax.imshow(pe, interpolation='none', cmap='jet')#, vmin=-5, vmax=5)
    fig.colorbar(cax)
    ax.set_title('power unc - max error {:1.4f} %'.format(max_error))
    plt.savefig('pin_power_unc_{:03d}_batches.png'.format(b))
    plt.close()

#kinetics.plotter.scalar_plot('core_power_density', 'C5G7_2D/MG/log_file.h5',
#                             directory='C5G7_2D/MG')
