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

from geometry_2d_ce import materials, surfaces, universes, cells, lattices, geometry
from plots import plots

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
    d = materials[name].mass_density
    density = np.array([[0., 1.     , 2.],
                        [d , omega*d, d ]])
    materials[name].set_density('g/cm3', density)

# OpenMC simulation parameters
batches = 70
inactive = 40
particles = 10000000

# Instantiate a Settings object
settings_file = openmc.Settings()
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.output = {'tallies': False}

# Create an initial uniform spatial source distribution over fissionable zones
source_bounds  = [-32.13, -10.71, -64.26, 10.71,  32.13,  64.26]
uniform_dist = openmc.stats.Box(source_bounds[:3], source_bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

sourcepoint = dict()
sourcepoint['batches'] = []
sourcepoint['write'] = False
settings_file.sourcepoint = sourcepoint

entropy_mesh = openmc.Mesh()
entropy_mesh.type = 'regular'
entropy_mesh.dimension = [34,34,1]
entropy_mesh.lower_left  = source_bounds[:3]
entropy_mesh.upper_right = source_bounds[3:]
settings_file.entropy_mesh = entropy_mesh

casmo_70_groups = openmc.mgxs.EnergyGroups()
casmo_70_groups.group_edges = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.042, 0.05, 0.058, 0.067, 0.08, 0.1,
                               0.14, 0.18, 0.22, 0.25, 0.28, 0.3, 0.32, 0.35, 0.4, 0.5, 0.625, 0.78, 0.85, 0.91,
                               0.95, 0.972, 0.996, 1.02, 1.045, 1.071, 1.097, 1.123, 1.15, 1.3, 1.5, 1.855, 2.1,
                               2.6, 3.3, 4.0, 9.877, 15.968, 27.7, 48.052, 75.5014, 148.728, 367.262, 906.898,
                               1425.1, 2239.45, 3519.1, 5530.0, 9118.0, 0.01503e6, 0.02478e6, 0.04085e6, 0.06743e6,
                               0.111e6, 0.183e6, 0.3025e6, 0.5e6, 0.821e6, 1.353e6, 2.231e6, 3.679e6, 6.0655e6, 1.e7]
casmo_40_groups = openmc.mgxs.EnergyGroups()
casmo_40_groups.group_edges = [0.0, 0.015, 0.03, 0.042, 0.058, 0.08, 0.1, 0.14, 0.18, 0.22, 0.28, 0.35, 0.625, 0.85,
                               0.95, 0.972, 1.02, 1.097, 1.15, 1.3, 1.5, 1.855, 2.1, 2.6, 4.0, 9.877, 15.968, 27.7,
                               48.052, 148.728, 5530.0, 9118.0, 0.111e6, 0.5e6, 0.821e6, 1.353e6, 2.231e6, 3.679e6, 6.0655e6, 1.e7]
casmo_25_groups = openmc.mgxs.EnergyGroups()
casmo_25_groups.group_edges = [0.0, 0.03, 0.058, 0.14, 0.28, 0.35, 0.625, 0.972, 1.02, 1.097, 1.15, 1.855, 4.0, 9.877, 15.968,
                               148.728, 5530.0, 9118.0, 0.111e6, 0.5e6, 0.821e6, 1.353e6, 2.231e6, 3.679e6, 6.0655e6, 1.e7]
casmo_23_groups = openmc.mgxs.EnergyGroups()
casmo_23_groups.group_edges = [0.0, 0.03, 0.058, 0.14, 0.28, 0.35, 0.625, 1.02, 1.097, 1.855, 4.0, 9.877, 15.968,
                               148.728, 5530.0, 9118.0, 0.111e6, 0.5e6, 0.821e6, 1.353e6, 2.231e6, 3.679e6, 6.0655e6, 1.e7]
casmo_18_groups = openmc.mgxs.EnergyGroups()
casmo_18_groups.group_edges = [0.0, 0.058, 0.14, 0.28, 0.625, 0.972, 1.15, 1.855, 4.0, 9.877, 15.968,
                               148.728, 5530.0, 9118.0, 0.111e6, 0.5e6, 0.821e6, 2.231e6, 1.e7]
casmo_16_groups = openmc.mgxs.EnergyGroups()
casmo_16_groups.group_edges = [0.0, 0.03, 0.058, 0.14, 0.28, 0.35, 0.625, 0.85, 0.972, 1.02, 1.097, 1.15, 1.3, 4.0,
                               5530.0, 0.821e6, 1.e7]
casmo_14_groups = openmc.mgxs.EnergyGroups()
casmo_14_groups.group_edges = [0.0, 0.03, 0.058, 0.14, 0.28, 0.35, 0.625, 0.972, 1.15, 1.855, 4.0, 48.052,
                               5530.0, 0.821e6, 2.231e6, 1.e7]
casmo_12_groups = openmc.mgxs.EnergyGroups()
casmo_12_groups.group_edges = [0.0, 0.03, 0.058, 0.14, 0.28, 0.35, 0.625, 4.0, 48.052, 5530.0, 0.821e6, 2.231e6, 1.e7]
casmo_9_groups = openmc.mgxs.EnergyGroups()
casmo_9_groups.group_edges = [0.0, 0.058, 0.14, 0.625, 0.972, 1.15, 4.0, 5530.0, 0.821e6, 1.e7]
casmo_8_groups = openmc.mgxs.EnergyGroups()
casmo_8_groups.group_edges = [0.0, 0.058, 0.14, 0.28, 0.625, 4.0, 5530.0, 0.821e6, 1.e7]
casmo_7_groups = openmc.mgxs.EnergyGroups()
casmo_7_groups.group_edges = [0.0, 0.058, 0.14, 0.625, 4.0, 5530.0, 0.821e6, 1.e7]
casmo_4_groups = openmc.mgxs.EnergyGroups()
casmo_4_groups.group_edges = [0.0, 0.625, 5530.0, 0.821e6, 1.e7]
casmo_3_groups = openmc.mgxs.EnergyGroups()
casmo_3_groups.group_edges = [0.0, 0.625, 5530.0, 1.e7]
casmo_2_groups = openmc.mgxs.EnergyGroups()
casmo_2_groups.group_edges = [0.0, 0.625, 1.e7]
casmo_1_group = openmc.mgxs.EnergyGroups()
casmo_1_group.group_edges = [0.0, 1.e7]

# Instantiate an EnergyGroups object for the transient solve
energy_groups = openmc.mgxs.EnergyGroups()
energy_groups.group_edges = [0., 0.14, 0.625, 4.0e0, 4.80521e1, 9.118e3, 1.353e6, 1.0e7]

# Instantiate an EnergyGroups object for the diffusion coefficients
fine_groups = openmc.mgxs.EnergyGroups()
fine_groups.group_edges = [0., 5.e-3, 1.e-2, 1.5e-2, 2.e-2, 2.5e-2, 3.e-2, 3.5e-2, 4.2e-2, 5.e-2, 5.8e-2, 6.7e-2, 8.e-2, 1.e-1, 1.4e-1,
                           1.8e-1, 2.2e-1, 2.5e-1, 2.8e-1, 3.e-1, 3.2e-1, 3.5e-1, 4.e-1, 5.e-1, 6.25e-1,
                           7.8e-1, 8.5e-1, 9.1e-1, 9.5e-1, 9.72e-1, 9.96e-1, 1.02e0, 1.045e0, 1.071e0, 1.097e0, 1.123e0, 1.15e0, 1.3e0, 1.5e0, 2.1e0, 2.6e0, 3.3e0, 4.0e0,
                           9.877e0, 1.5968e1, 2.77e1, 4.80521e1,
                           7.55014e1, 1.48729e2, 3.67263e2, 9.06899e2, 1.4251e3, 2.23945e3, 3.5191e3, 5.53e3, 9.118e3,
                           1.503e4, 2.478e4, 4.085e4, 6.734e4, 1.11e5, 1.83e5, 3.025e5, 5.e5, 8.21e5, 1.353e6,
                           2.231e6, 3.679e6, 6.0655e6, 1.0e7]

# Instantiate an EnergyGroups object for one group data
one_group = openmc.mgxs.EnergyGroups()
one_group.group_edges = [fine_groups.group_edges[0], fine_groups.group_edges[-1]]

# Create pin cell mesh
point_mesh = openmc.Mesh()
point_mesh.type = 'regular'
point_mesh.dimension = [1,1,1]
point_mesh.lower_left  = [-32.13, -10.71, -64.26]
point_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_point_mesh = openmc.Mesh()
full_point_mesh.type = 'regular'
full_point_mesh.dimension = [1,1,1]
full_point_mesh.lower_left  = [-32.13, -32.13, -64.26]
full_point_mesh.upper_right = [ 32.13,  32.13,  64.26]

pin_cell_mesh = openmc.Mesh()
pin_cell_mesh.type = 'regular'
pin_cell_mesh.dimension = [34,34,1]
pin_cell_mesh.lower_left  = [-32.13, -10.71, -64.26]
pin_cell_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_pin_cell_mesh = openmc.Mesh()
full_pin_cell_mesh.type = 'regular'
full_pin_cell_mesh.dimension = [51,51,1]
full_pin_cell_mesh.lower_left  = [-32.13, -32.13, -64.26]
full_pin_cell_mesh.upper_right = [ 32.13,  32.13,  64.26]

assembly_mesh = openmc.Mesh()
assembly_mesh.type = 'regular'
assembly_mesh.dimension = [2,2,1]
assembly_mesh.lower_left  = [-32.13, -10.71, -64.26]
assembly_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_assembly_mesh = openmc.Mesh()
full_assembly_mesh.type = 'regular'
full_assembly_mesh.dimension = [3,3,1]
full_assembly_mesh.lower_left  = [-32.13, -32.13, -64.26]
full_assembly_mesh.upper_right = [ 32.13,  32.13,  64.26]

quarter_assembly_mesh = openmc.Mesh()
quarter_assembly_mesh.type = 'regular'
quarter_assembly_mesh.dimension = [4,4,1]
quarter_assembly_mesh.lower_left  = [-32.13, -10.71, -64.26]
quarter_assembly_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_quarter_assembly_mesh = openmc.Mesh()
full_quarter_assembly_mesh.type = 'regular'
full_quarter_assembly_mesh.dimension = [6,6,1]
full_quarter_assembly_mesh.lower_left  = [-32.13, -32.13, -64.26]
full_quarter_assembly_mesh.upper_right = [ 32.13,  32.13,  64.26]

# Instantiate a clock object
clock = openmc.kinetics.Clock(start=0., end=2., dt_outer=2.5e-1, dt_inner=1.e-2)

# Instantiate a kinetics solver object
solver = openmc.kinetics.Solver(name='CE_OMEGA', directory='C5G7_2D')
solver.num_delayed_groups           = 6
solver.flux_mesh                    = full_quarter_assembly_mesh
solver.pin_mesh                     = full_pin_cell_mesh
solver.one_group                    = casmo_1_group
solver.energy_groups                = casmo_8_groups
solver.fine_groups                  = casmo_70_groups
solver.geometry                     = geometry
solver.settings_file                = settings_file
solver.materials_file               = materials_file
solver.inner_tolerance              = 1.e-3
solver.outer_tolerance              = np.inf
solver.method                       = 'OMEGA'
solver.multi_group                  = False
solver.clock                        = clock
solver.mpi_procs                    = 36*1
solver.threads                      = 1
solver.core_volume                  = 42.84 * 42.84 * 128.52
solver.constant_seed                = True
solver.seed                         = 1
solver.chi_delayed_by_delayed_group = True
solver.chi_delayed_by_mesh          = False
solver.use_pregenerated_sps         = False
solver.pregenerate_sps              = False
solver.run_on_cluster               = False
solver.job_file                     = 'job_broadwell.pbs'
solver.log_file_name                = 'log_file.h5'

# Solve transient problem
solver.solve()