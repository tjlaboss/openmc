#coding=utf-8

import matplotlib
matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import math
import pickle
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import os

import openmc
import openmc.mgxs
import openmc.plotter
import openmc.kinetics as kinetics

from geometry_mg import materials, surfaces, universes, cells, lattices, geometry, mgxs_lib_file
from plots import plots
from mgxs_lib import mgxs_data

run_directory = 'C5G7_SS_3D_MG'

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

inactive    = 60
active      = 200
particles   = 10000000
sp_interval = 5

# Create settings file
settings_file = openmc.Settings()
settings_file.energy_mode = 'multi-group'
settings_file.batches = active + inactive
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.seed = 1
settings_file.output = {'tallies': False}

statepoint = dict()
sp_batches = range(inactive + sp_interval, inactive + sp_interval + active, sp_interval)
sp_particles = [(i-inactive)*particles for i in sp_batches]
#sp_batches = [40, 130]
statepoint['batches'] = sp_batches
settings_file.statepoint = statepoint

# Create an initial uniform spatial source distribution over fissionable zones
source_bounds  = [-32.13, -10.71, -64.26, 10.71,  32.13,  64.26]
entropy_bounds = [-32.13, -10.71, -85.68, 10.71,  32.13,  85.68]
uniform_dist = openmc.stats.Box(source_bounds[:3], source_bounds[3:], only_fissionable=True)
settings_file.source = openmc.source.Source(space=uniform_dist)

entropy_mesh = openmc.Mesh()
entropy_mesh.type = 'regular'
entropy_mesh.dimension = [4,4,32]
entropy_mesh.lower_left  = entropy_bounds[:3]
entropy_mesh.upper_right = entropy_bounds[3:]
settings_file.entropy_mesh = entropy_mesh

settings_file.export_to_xml(run_directory + '/settings.xml')

###############################################################
#                      tallies.xml
##############################################################

pin_mesh = openmc.Mesh()
pin_mesh.type = 'regular'
pin_mesh.dimension = [34,34,1]
pin_mesh.lower_left  = [-32.13, -10.71, -85.68]
pin_mesh.upper_right = [ 10.71,  32.13,  85.68]

mesh_filter = openmc.MeshFilter(pin_mesh)
tally = openmc.Tally(tally_id=1)
tally.filters = [mesh_filter]
tally.scores = ['kappa-fission']

# Generate a new tallies file
tallies_file = openmc.Tallies([tally])
tallies_file.export_to_xml(run_directory + '/tallies.xml')

openmc.run(threads=1, mpi_procs=36*2, mpi_exec='mpirun', cwd=run_directory)

###############################################################
#                   data processing
##############################################################

pin_powers = []
k_eff_unc = []
for i,batch in enumerate(sp_batches):
    su = openmc.Summary(run_directory + '/summary.h5')
    if inactive + active >= 10000:
        sp = openmc.StatePoint(run_directory + '/statepoint.{:05d}.h5'.format(batch), False)
    elif inactive + active >= 1000:
        sp = openmc.StatePoint(run_directory + '/statepoint.{:04d}.h5'.format(batch), False)
    elif inactive + active >= 100:
        sp = openmc.StatePoint(run_directory + '/statepoint.{:03d}.h5'.format(batch), False)
    else:
        sp = openmc.StatePoint(run_directory + '/statepoint.{:d}.h5'.format(batch), False)
    sp.link_with_summary(su)

    print('Loading tallies {} of {}'.format(i+1, len(sp_batches)))

    tally = sp.get_tally(scores=['kappa-fission'])
    powers = tally.get_values(scores=['kappa-fission'])
    powers.shape = (34,34)
    powers[8,8] = 0.
    powers[8,25] = 0.
    powers[25,8] = 0.
    powers[25,25] = 0.

    pin_powers.append(powers)
    k_eff_unc.append(sp.k_combined[1])

rms_errors = []
max_errors = []
for i,batch in enumerate(sp_batches):
    power_errors = np.abs((pin_powers[i] - pin_powers[-1]) / pin_powers[-1] * 100.)
    power_errors = np.nan_to_num(power_errors)
    rms_errors.append(np.sqrt(np.mean((power_errors**2).flatten())))
    max_errors.append(np.max(power_errors.flatten()))


import seaborn as sns
colors = sns.color_palette()

sp_particles = np.array(sp_particles)
rms_errors = np.array(rms_errors)
max_errors = np.array(max_errors)
k_eff_unc = np.array(k_eff_unc)

arrays = [sp_particles.tolist(), rms_errors.tolist(), max_errors.tolist(), k_eff_unc.tolist()]
pickle.dump(arrays, open( "c5g7_ss_3d_mg_data.pkl", "wb" ) )

fig = plt.figure(figsize=(9,7))
plt.scatter(sp_particles[:-20], rms_errors[:-20]    , marker='o', s=200, c='b', label='RMS error', zorder=5)
plt.scatter(sp_particles[:-20], max_errors[:-20]    , marker='v', s=200, c='b', label='Max error', zorder=4)
plt.scatter(sp_particles[:-20], k_eff_unc[:-20]*1.e5, marker='o', s=200, c='r', label='k-eff', zorder=3)
plt.loglog([1.e5, 1.e10], [1.0, 1.0], 'k--', linewidth=3, zorder=2)
plt.loglog([1.e5, 1.e10], [0.5, 0.5], 'k--', linewidth=3, zorder=1)
plt.legend(loc=1, fontsize=16)
plt.xlabel('# of particles', fontsize=16)
#plt.ylabel(r'\textcolor{blue}{Relative error (\%)} \textcolor{black}{or} \textcolor{red}{k-eff unc. (pcm)}', fontsize=16)
plt.ylabel('Relative error (%) or k-eff uncertainty (pcm)', fontsize=16)
plt.gca().tick_params(labelsize=14)
plt.gca().grid(True, which='both')

plt.xlim([3.e7, 1.3e9])
plt.ylim([5.e-2, 3.e1])
plt.tight_layout()
plt.savefig('c5g7_ss_3d_mg_pp_conv.png')




