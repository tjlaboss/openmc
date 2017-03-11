#coding=utf-8

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

from geometry_2d_mg import materials, surfaces, universes, cells, lattices, geometry, mgxs_lib_file
from plots import plots
from mgxs_lib import mgxs_data

# Set the base control rod bank positions
cells['Control Rod Base Bank 1'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 2'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 3'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 4'].translation = [0., 0., 64.26]

# Create the materials file
materials_file = openmc.Materials(geometry.get_all_materials().values())

case = '3.1'
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
batches = 80
inactive = 30
particles = 1000000

# Instantiate a Settings object
settings_file = openmc.Settings()
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.output = {'tallies': True}

# Create an initial uniform spatial source distribution over fissionable zones
source_bounds  = [-32.13, -10.71, -64.26, 10.71,  32.13,  64.26]
entropy_bounds = [-32.13, -10.71, -64.26, 10.71,  32.13,  64.26]
uniform_dist = openmc.stats.Box(source_bounds[:3], source_bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

entropy_mesh = openmc.Mesh()
entropy_mesh.type = 'regular'
entropy_mesh.dimension = [34,34,1]
entropy_mesh.lower_left  = entropy_bounds[:3]
entropy_mesh.upper_right = entropy_bounds[3:]
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
clock = openmc.kinetics.Clock(start=0., end=2., dt_outer=1.e-1, dt_inner=1.e-2)

# Instantiate a kinetics solver object
solver = openmc.kinetics.Solver(name='MG_PC_TEST', directory='C5G7_2D')
solver.num_delayed_groups           = 8
solver.mesh                         = pin_cell_mesh
solver.pin_cell_mesh                = pin_cell_mesh
solver.assembly_mesh                = assembly_mesh
solver.one_group                    = one_group
solver.energy_groups                = energy_groups
solver.fine_groups                  = fine_groups
solver.geometry                     = geometry
solver.settings_file                = settings_file
solver.materials_file               = materials_file
solver.mgxs_lib_file                = mgxs_lib_file
solver.clock                        = clock
solver.mpi_procs                    = 32*1
solver.threads                      = 1
solver.ppn                          = 32
solver.core_volume                  = 42.84 * 42.84 * 128.52
solver.constant_seed                = False
solver.seed                         = 1
solver.chi_delayed_by_delayed_group = True
solver.chi_delayed_by_mesh          = True
solver.chi_analog                   = False
solver.use_pregenerated_sps         = False
solver.pregenerate_sps              = True
solver.run_on_cluster               = False
solver.job_file                     = 'job_fission.pbs'
solver.log_file_name                = 'log_file.h5'

# Solve transient problem
solver.solve()

#kinetics.plotter.scalar_plot('core_power_density', 'C5G7_2D/MG/log_file.h5',
#                             directory='C5G7_2D/MG')
