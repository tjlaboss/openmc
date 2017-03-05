import math
import pickle
import matplotlib
from copy import deepcopy

import numpy as np
import os

import openmc
import openmc.mgxs
import openmc.plotter
import openmc.kinetics as kinetics

from geometry_ce import materials, surfaces, universes, cells, lattices, geometry
from plots import plots
from mgxs_lib import mgxs_data

# Create the materials file
materials_file = openmc.Materials(geometry.get_all_materials())

# Set the base control rod bank positions
cells['Control Rod Base Bank 1'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 2'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 3'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 4'].translation = [0., 0., 64.26]

case = '5.4'

# Adjust the cells to have the desired translations or moderator densities
if case == '4.1':
    cells['Control Rod Base Bank 1'].translation = [[0., 0., 64.26],
                                                    [0., 0., 21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 1'].translation_times = [0., 2., 4.]
elif case == '4.2':
    cells['Control Rod Base Bank 3'].translation = [[0., 0., 64.26],
                                                    [0., 0.,-21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 3'].translation_times = [0., 4., 8.]
elif case == '4.3':
    cells['Control Rod Base Bank 1'].translation = [[0., 0., 64.26],
                                                    [0., 0., 21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 1'].translation_times = [2., 4., 6.]
    cells['Control Rod Base Bank 3'].translation = [[0., 0., 64.26],
                                                    [0., 0.,-21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 3'].translation_times = [0., 4., 8.]
elif case == '4.4':
    cells['Control Rod Base Bank 3'].translation = [[0., 0., 64.26],
                                                    [0., 0., 21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 3'].translation_times = [4., 6., 8.]
    cells['Control Rod Base Bank 4'].translation = [[0., 0., 64.26],
                                                    [0., 0., 21.42],
                                                    [0., 0., 21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 4'].translation_times = [0., 2., 4., 6.]
elif case == '4.5':
    cells['Control Rod Base Bank 1'].translation = [[0., 0., 64.26],
                                                    [0., 0., 21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 1'].translation_times = [0., 2., 4.]
    cells['Control Rod Base Bank 3'].translation = [[0., 0., 64.26],
                                                    [0., 0., 21.42],
                                                    [0., 0., 21.42],
                                                    [0., 0., 64.26]]
    cells['Control Rod Base Bank 3'].translation_times = [2., 4., 6., 8.]
elif case == '5.1':
    d = materials['Moderator Bank 1'].mass_density
    density = np.array([[0., 2.    , 4.],
                        [d , 0.90*d, d ]])
    materials['Moderator Bank 1'].set_density('g/cm3', density)
    d = materials['Moderator Bank 3'].mass_density
    density = np.array([[1., 2.    , 3.],
                        [d , 0.95*d, d ]])
    materials['Moderator Bank 3'].set_density('g/cm3', density)
elif case == '5.2':
    d = materials['Moderator Bank 1'].mass_density
    density = np.array([[0., 2.    , 4.],
                        [d , 0.80*d, d ]])
    materials['Moderator Bank 1'].set_density('g/cm3', density)
    d = materials['Moderator Bank 3'].mass_density
    density = np.array([[1., 2.    , 3.],
                        [d , 0.95*d, d ]])
    materials['Moderator Bank 3'].set_density('g/cm3', density)
elif case == '5.3':
    d = materials['Moderator Bank 1'].mass_density
    density = np.array([[0., 2.    , 4.],
                        [d , 0.80*d, d ]])
    materials['Moderator Bank 1'].set_density('g/cm3', density)
    d = materials['Moderator Bank 3'].mass_density
    density = np.array([[0., 2.    , 4.],
                        [d , 0.90*d, d ]])
    materials['Moderator Bank 3'].set_density('g/cm3', density)
    d = materials['Moderator Bank 4'].mass_density
    density = np.array([[1., 2.    , 3.],
                        [d , 0.95*d, d ]])
    materials['Moderator Bank 4'].set_density('g/cm3', density)
elif case == '5.4':
    d = materials['Moderator Bank 2'].mass_density
    density = np.array([[0., 2.    , 4.],
                        [d , 0.80*d, d ]])
    materials['Moderator Bank 2'].set_density('g/cm3', density)
    d = materials['Moderator Bank 3'].mass_density
    density = np.array([[0., 2.    , 4.],
                        [d , 0.90*d, d ]])
    materials['Moderator Bank 3'].set_density('g/cm3', density)
    d = materials['Moderator Bank 4'].mass_density
    density = np.array([[1., 2.    , 3.],
                        [d , 0.95*d, d ]])
    materials['Moderator Bank 4'].set_density('g/cm3', density)


# OpenMC simulation parameters
batches = 200
inactive = 100
particles = 250000

# Instantiate a Settings object
settings_file = openmc.Settings()
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.output = {'tallies': False}

# Create an initial uniform spatial source distribution over fissionable zones
bounds = [-32.13, -10.71, -64.26, 10.71,  32.13,  64.26]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

settings_file.entropy_lower_left  = bounds[:3]
settings_file.entropy_upper_right = bounds[3:]
settings_file.entropy_dimension   = [34,34,1]

# Instantiate a 50-group EnergyGroups object
fine_groups = openmc.mgxs.EnergyGroups()
fine_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 2.0e7]
#fine_groups.group_edges = [0., 55.6, 1.0e7]

# Instantiate a 2-group EnergyGroups object
energy_groups = openmc.mgxs.EnergyGroups()
energy_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 2.0e7]
#energy_groups.group_edges = [0., 55.6, 1.0e7]

# Instantiate a 1-group EnergyGroups object
one_group = openmc.mgxs.EnergyGroups()
one_group.group_edges = [fine_groups.group_edges[0], fine_groups.group_edges[-1]]

# Create pin cell mesh
mesh = openmc.Mesh()
mesh.type = 'regular'
mesh.dimension = [34,34,1]
mesh.lower_left  = [-32.13, -10.71, -85.68]
mesh.width = [42.84/mesh.dimension[0],
              42.84/mesh.dimension[1],
              171.36/mesh.dimension[2]]

pin_cell_mesh = openmc.Mesh()
pin_cell_mesh.type = 'regular'
pin_cell_mesh.dimension = [34,34,1]
pin_cell_mesh.lower_left  = [-32.13, -10.71, -85.68]
pin_cell_mesh.width = [42.84/pin_cell_mesh.dimension[0],
                       42.84/pin_cell_mesh.dimension[1],
                       171.36/pin_cell_mesh.dimension[2]]

assembly_mesh = openmc.Mesh()
assembly_mesh.type = 'regular'
assembly_mesh.dimension = [2,2,1]
assembly_mesh.lower_left  = [-32.13, -10.71, -85.68]
assembly_mesh.width = [42.84/assembly_mesh.dimension[0],
                       42/.84/assembly_mesh.dimension[1],
                       171.36/assembly_mesh.dimension[2]]

# Instantiate a clock object
clock = openmc.kinetics.Clock(start=0., end=10., dt_outer=5.e-1, dt_inner=1.e-2)

# Instantiate a kinetics solver object
solver = openmc.kinetics.Solver(name='CE_4_5', directory='C5G7_3D')
solver.num_delayed_groups           = 6
solver.mesh                         = mesh
solver.pin_cell_mesh                = pin_cell_mesh
solver.assembly_mesh                = assembly_mesh
solver.one_group                    = one_group
solver.energy_groups                = energy_groups
solver.fine_groups                  = fine_groups
solver.geometry                     = geometry
solver.settings_file                = settings_file
solver.materials_file               = materials_file
solver.clock                        = clock
solver.mpi_procs                    = 24*5
solver.threads                      = 1
solver.ppn                          = 24
solver.core_volume                  = 42.84 * 42.84 * 128.52
solver.constant_seed                = True
solver.chi_delayed_by_delayed_group = False
solver.chi_delayed_by_mesh          = False
solver.use_pregenerated_sps         = False
solver.pregenerate_sps              = False
solver.run_on_cluster               = True
solver.job_file                     = 'job.pbs'
solver.log_file_name                = 'log_file.h5'

# Run OpenMC
solver.solve()
