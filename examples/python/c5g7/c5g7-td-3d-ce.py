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
from casmo_groups import casmo_group_structures

# Create the materials file
materials_file = openmc.Materials(geometry.get_all_materials().values())

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
batches = 110
inactive = 60
particles = 1000000

# Instantiate a Settings object
settings_file = openmc.Settings()
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.output = {'tallies': False}

# Create an initial uniform spatial source distribution over fissionable zones
bounds = [-32.13, -10.71, -64.26, 10.71,  32.13,  64.26]
entropy_bounds = [-32.13, -10.71, -85.68, 10.71,  32.13,  85.68]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

sourcepoint = dict()
sourcepoint['batches'] = [batches]
sourcepoint['separate'] = True
sourcepoint['write'] = True
settings_file.sourcepoint = sourcepoint

entropy_mesh = openmc.Mesh()
entropy_mesh.type = 'regular'
entropy_mesh.dimension = [4,4,32]
entropy_mesh.lower_left  = entropy_bounds[:3]
entropy_mesh.upper_right = entropy_bounds[3:]
settings_file.entropy_mesh = entropy_mesh

# Create pin cell mesh
point_mesh = openmc.Mesh()
point_mesh.type = 'regular'
point_mesh.dimension = [1,1,1]
point_mesh.lower_left  = [-32.13, -10.71, -64.26]
point_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_point_mesh = openmc.Mesh()
full_point_mesh.type = 'regular'
full_point_mesh.dimension = [1,1,1]
full_point_mesh.lower_left  = [-32.13, -32.13, -85.68]
full_point_mesh.upper_right = [ 32.13,  32.13,  85.68]

pin_cell_mesh = openmc.Mesh()
pin_cell_mesh.type = 'regular'
pin_cell_mesh.dimension = [34,34,24]
pin_cell_mesh.lower_left  = [-32.13, -10.71, -64.26]
pin_cell_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_pin_cell_mesh = openmc.Mesh()
full_pin_cell_mesh.type = 'regular'
full_pin_cell_mesh.dimension = [51,51,32]
full_pin_cell_mesh.lower_left  = [-32.13, -32.13, -85.68]
full_pin_cell_mesh.upper_right = [ 32.13,  32.13,  85.68]

assembly_mesh = openmc.Mesh()
assembly_mesh.type = 'regular'
assembly_mesh.dimension = [2,2,6]
assembly_mesh.lower_left  = [-32.13, -10.71, -64.26]
assembly_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_assembly_mesh = openmc.Mesh()
full_assembly_mesh.type = 'regular'
full_assembly_mesh.dimension = [3,3,8]
full_assembly_mesh.lower_left  = [-32.13, -32.13, -85.68]
full_assembly_mesh.upper_right = [ 32.13,  32.13,  85.68]

quarter_assembly_mesh = openmc.Mesh()
quarter_assembly_mesh.type = 'regular'
quarter_assembly_mesh.dimension = [4,4,6]
quarter_assembly_mesh.lower_left  = [-32.13, -10.71, -64.26]
quarter_assembly_mesh.upper_right = [ 10.71,  32.13,  64.26]

full_quarter_assembly_mesh = openmc.Mesh()
full_quarter_assembly_mesh.type = 'regular'
full_quarter_assembly_mesh.dimension = [6,6,8]
full_quarter_assembly_mesh.lower_left  = [-32.13, -32.13, -85.68]
full_quarter_assembly_mesh.upper_right = [ 32.13,  32.13,  85.68]

t_outer = np.arange(0., 2.5, 5.e-1)

# Instantiate a clock object
clock = openmc.kinetics.Clock(start=0., end=2., dt_inner=1.e-2, t_outer=t_outer)

# Instantiate a kinetics solver object
solver = openmc.kinetics.Solver(directory='C5G7_3D_CE')
solver.num_delayed_groups           = 6
solver.amplitude_mesh               = full_point_mesh
solver.shape_mesh                   = full_quarter_assembly_mesh
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
solver.method                       = 'ADIABATIC'
solver.multi_group                  = False
solver.clock                        = clock
solver.mpi_procs                    = 36*10
solver.threads                      = 1
solver.core_volume                  = 42.84 * 42.84 * 128.52
solver.constant_seed                = True
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

# Run OpenMC
solver.solve()
