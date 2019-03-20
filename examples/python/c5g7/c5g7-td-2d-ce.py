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
from casmo_groups import casmo_group_structures

case = '3.0'
omega = 1.0

# Adjust the cells to have the desired moderator densities
if case == '3.0':
    omega = 1.0
elif case == '3.1':
    omega = 0.95
elif case == '3.2':
    omega = 0.90
elif case == '3.3':
    omega = 0.85
elif case == '3.4':
    omega = 0.8

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


for i in range(10):

    print('case {}'.format(i))

    # Set the base control rod bank positions
    cells['Control Rod Base Bank 1'].translation = [0., 0., 64.26]
    cells['Control Rod Base Bank 2'].translation = [0., 0., 64.26]
    cells['Control Rod Base Bank 3'].translation = [0., 0., 64.26]
    cells['Control Rod Base Bank 4'].translation = [0., 0., 64.26]

    # Create the materials file
    materials_file = openmc.Materials(geometry.get_all_materials().values())

    for bank in range(1,5):
        name = 'Moderator Bank {}'.format(bank)
        d = materials[name].mass_density
        density = np.array([[0., 1.     , 2.],
                            [d , omega*d, d ]])
        materials[name].set_density('g/cm3', density)

    # OpenMC simulation parameters
    batches = 80
    inactive = 40
    particles = 100000

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
    sourcepoint['batches'] = [batches]
    sourcepoint['separate'] = True
    sourcepoint['write'] = True
    settings_file.sourcepoint = sourcepoint

    entropy_mesh = openmc.Mesh()
    entropy_mesh.type = 'regular'
    entropy_mesh.dimension = [4,4,1]
    entropy_mesh.lower_left  = source_bounds[:3]
    entropy_mesh.upper_right = source_bounds[3:]
    settings_file.entropy_mesh = entropy_mesh

    t_outer = np.arange(0., 2.5, 5.e-1)
    dt_inner = [1.e-2 for j in range(len(t_outer))]

    # Instantiate a clock object
    clock = openmc.kinetics.Clock(start=0., end=2., dt_inner=dt_inner, t_outer=t_outer)

    # Instantiate a kinetics solver object
    solver = openmc.kinetics.Solver(directory='C5G7_2D_CE_{}'.format(i))
    solver.num_delayed_groups           = 6
    solver.amplitude_mesh               = full_pin_cell_mesh
    solver.shape_mesh                   = full_pin_cell_mesh
    solver.tally_mesh                   = full_pin_cell_mesh
    solver.one_group                    = casmo_group_structures[1]
    solver.energy_groups                = casmo_group_structures[8]
    solver.fine_groups                  = casmo_group_structures[8]
    solver.tally_groups                 = casmo_group_structures[8]
    solver.geometry                     = geometry
    solver.settings_file                = settings_file
    solver.materials_file               = materials_file
    solver.inner_tolerance              = np.inf
    solver.outer_tolerance              = np.inf
    solver.method                       = 'OMEGA'
    solver.multi_group                  = False
    solver.clock                        = clock
    solver.mpi_procs                    = 36*1
    solver.threads                      = 1
    solver.core_volume                  = 42.84 * 42.84 * 128.52
    solver.constant_seed                = False
    solver.seed                         = 1
    solver.use_pcmfd                    = False
    solver.use_agd                      = False
    solver.condense_dif_coef            = True
    solver.chi_delayed_by_delayed_group = True
    solver.chi_delayed_by_mesh          = False
    solver.use_pregenerated_sps         = False
    solver.run_on_cluster               = False
    solver.job_file                     = 'job_broadwell.pbs'
    solver.log_file_name                = 'log_file.h5'

    # Solve transient problem
    solver.solve()

