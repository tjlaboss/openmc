#coding=utf-8

import math
import pickle
import matplotlib
from copy import deepcopy
from collections import OrderedDict

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
import seaborn as sns

run_directory = 'xs_conv'

# Create run directory
if not os.path.exists(run_directory):
    os.makedirs(run_directory)


###############################################################
#                      geometry.xml
##############################################################

# Set the base control rod bank positions
cells['Control Rod Base Bank 1'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 2'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 3'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 4'].translation = [0., 0., 64.26]

geometry.time = 0.0
geometry.export_to_xml(run_directory + '/geometry.xml')


###############################################################
#                        mgxs.h5
##############################################################

# Create mgxs file
mgxs_lib_file.export_to_hdf5(run_directory + '/mgxs.h5')


###############################################################
#                      materials.xml
##############################################################

# Create materials file
materials_file = openmc.Materials(geometry.get_all_materials().values())
materials_file.cross_sections = './mgxs.h5'
materials_file.export_to_xml(run_directory + '/materials.xml')


###############################################################
#                      settings.xml
##############################################################

inactive    = 30
active      = 100
particles   = 1000000
sp_interval = 10

# Create settings file
settings_file = openmc.Settings()
settings_file.energy_mode = 'multi-group'
settings_file.batches = active + inactive
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.output = {'tallies': False}

statepoint = dict()
sp_batches = range(inactive + sp_interval, inactive + sp_interval + active, sp_interval)
statepoint['batches'] = sp_batches
settings_file.statepoint = statepoint

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

settings_file.export_to_xml(run_directory + '/settings.xml')


###############################################################
#                      tallies.xml
##############################################################

chi_delayed_by_delayed_group = True
chi_delayed_by_mesh          = True
chi_analog                   = False
scat_analog                  = False

# Instantiate an EnergyGroups object
fine_groups = openmc.mgxs.EnergyGroups()
fine_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 1.0e7]

energy_groups = openmc.mgxs.EnergyGroups()
energy_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 1.0e7]

one_group = openmc.mgxs.EnergyGroups()
one_group.group_edges = [fine_groups.group_edges[0], fine_groups.group_edges[-1]]

# Create a tally mesh
mesh = openmc.Mesh()
mesh.type = 'regular'
mesh.dimension = [34,34,1]
mesh.lower_left  = [-32.13, -10.71, -64.26]
mesh.width = [42.84/mesh.dimension[0], 42.84/mesh.dimension[1], 128.52/mesh.dimension[2]]

unity_mesh = openmc.Mesh()
unity_mesh.type = mesh.type
unity_mesh.dimension = [1,1,1]
unity_mesh.lower_left  = mesh.lower_left
unity_mesh.width = [i*j for i,j in zip(mesh.dimension, mesh.width)]

# Instantiate a list of the delayed groups
delayed_groups = list(range(1,9))

# Create elements and ordered dicts and initialize to None
mgxs_dict = OrderedDict()

mgxs_types = ['absorption', 'diffusion-coefficient', 'decay-rate',
              'kappa-fission', 'nu-scatter matrix', 'chi-prompt',
              'chi-delayed', 'inverse-velocity', 'prompt-nu-fission',
              'current', 'delayed-nu-fission']

mgxs_uncertainty = OrderedDict()

# Populate the MGXS in the MGXS lib
for mgxs_type in mgxs_types:

    mgxs_uncertainty[mgxs_type] = np.zeros(len(sp_batches))

    if mgxs_type == 'diffusion-coefficient':
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=fine_groups, by_nuclide=False,
            name=mgxs_type)
        mgxs_dict[mgxs_type].energy_mode = 'multi-group'
    elif 'nu-scatter matrix' in mgxs_type:
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups, by_nuclide=False,
            name=mgxs_type)
        mgxs_dict[mgxs_type].correction = None
        mgxs_dict[mgxs_type].energy_mode = 'multi-group'
    elif mgxs_type == 'decay-rate':
        mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=one_group,
            delayed_groups=delayed_groups, by_nuclide=False,
            name=mgxs_type)
        mgxs_dict[mgxs_type].energy_mode = 'multi-group'
    elif mgxs_type == 'chi-prompt':
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups, by_nuclide=False,
            name=mgxs_type)
        mgxs_dict[mgxs_type].energy_mode = 'multi-group'
        if chi_analog:
            mgxs_dict[mgxs_type].estimator = 'analog'
    elif mgxs_type in openmc.mgxs.MGXS_TYPES:
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups, by_nuclide=False,
            name=mgxs_type)
        mgxs_dict[mgxs_type].energy_mode = 'multi-group'
    elif mgxs_type == 'chi-delayed':
        if chi_delayed_by_delayed_group:
            if chi_delayed_by_mesh:
                mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=energy_groups,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name=mgxs_type)
            else:
                mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=unity_mesh, domain_type='mesh',
                    energy_groups=energy_groups,
                    delayed_groups=delayed_groups, by_nuclide=False,
                    name=mgxs_type)
        else:
            if chi_delayed_by_mesh:
                mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=mesh, domain_type='mesh',
                    energy_groups=energy_groups, by_nuclide=False,
                    name=mgxs_type)
            else:
                mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
                    mgxs_type, domain=unity_mesh, domain_type='mesh',
                    energy_groups=energy_groups, by_nuclide=False,
                    name=mgxs_type)
        mgxs_dict[mgxs_type].energy_mode = 'multi-group'
        if chi_analog:
            mgxs_dict[mgxs_type].estimator = 'analog'
    elif mgxs_type in openmc.mgxs.MDGXS_TYPES:
        mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups,
            delayed_groups=delayed_groups, by_nuclide=False,
            name=mgxs_type)
        mgxs_dict[mgxs_type].energy_mode = 'multi-group'

# Generate a new tallies file
tallies_file = openmc.Tallies()

# Add the tallies to the file
for mgxs in mgxs_dict.values():
    tallies = mgxs.tallies.values()
    for tally in tallies:
        tallies_file.append(tally, True)

# Export the tallies file to xml
tallies_file.export_to_xml(run_directory + '/tallies.xml')

openmc.run(threads=1, mpi_procs=24, mpi_exec='mpirun', cwd=run_directory)

###############################################################
#                   data processing
##############################################################

for i,batch in enumerate(sp_batches):
    su = openmc.Summary(run_directory + '/summary.h5')
    if inactive + active >= 100:
        sp = openmc.StatePoint(run_directory + '/statepoint.{:03d}.h5'.format(batch), False)
    else:
        sp = openmc.StatePoint(run_directory + '/statepoint.{:d}.h5'.format(batch), False)
    sp.link_with_summary(su)

    print('Loading MGXS {} of {}'.format(i+1, len(sp_batches)))

    for mgxs_type,mgxs in mgxs_dict.items():
        mgxs.load_from_statepoint(sp)
        mgxs_uncertainty[mgxs_type][i] = np.linalg.norm(mgxs.get_xs(value='rel_err'))

sns.set_palette(sns.husl_palette(len(mgxs_dict.values()), l=0.5))
plt.figure(figsize=(9,6))
sp_particles = [i*particles for i in sp_batches]
i = 0
for mgxs_type,mgxs in mgxs_dict.items():
    if i % 3 == 0:
        plt.loglog(sp_particles, mgxs_uncertainty[mgxs_type], label=mgxs_type, marker='o')
    elif i % 3 == 1:
        plt.loglog(sp_particles, mgxs_uncertainty[mgxs_type], label=mgxs_type, marker='D')
    else:
        plt.loglog(sp_particles, mgxs_uncertainty[mgxs_type], label=mgxs_type, marker='<')
    i = i + 1

legend = plt.legend(ncol=3, loc=9, frameon=True, fancybox=True, facecolor='white', edgecolor='black')
plt.ylim([1e-0, 1e4])
plt.xlabel('# of neutrons')
plt.ylabel('relative error')
plt.title('MGXS relative error for 2D MG C5G7')
plt.savefig('mg_mgxs_uncertainty.png')
plt.close()

for mgxs_type in mgxs_dict.keys():
    print('{}: {}'.format(mgxs_type, mgxs_uncertainty[mgxs_type]))

