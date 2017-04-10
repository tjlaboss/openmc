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

from geometry_2d_ce import materials, surfaces, universes, cells, lattices, geometry
from plots import plots

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use("Agg")
import matplotlib.pyplot as plt


run_directory = 'ce_xs_conv_local_tl'

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
#                      materials.xml
##############################################################

# Create materials file
materials_file = openmc.Materials(geometry.get_all_materials().values())
materials_file.export_to_xml(run_directory + '/materials.xml')


###############################################################
#                      settings.xml
##############################################################

inactive    = 30
active      = 100
particles   = 1000000
sp_interval = 100

# Create settings file
settings_file = openmc.Settings()
settings_file.batches = active + inactive
settings_file.inactive = inactive
settings_file.particles = particles
settings_file.seed = 2
settings_file.output = {'tallies': False}

statepoint = dict()
sp_batches = range(inactive + sp_interval, inactive + sp_interval + active, sp_interval)
sp_particles = [(i-inactive)*particles for i in sp_batches]
sp_batches = [40, 130]
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

# Instantiate an EnergyGroups object
fine_groups = openmc.mgxs.EnergyGroups()
fine_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 2.0e7]

energy_groups = openmc.mgxs.EnergyGroups()
energy_groups.group_edges = [0., 0.13, 0.63, 4.1, 55.6, 9.2e3, 1.36e6, 2.0e7]

one_group = openmc.mgxs.EnergyGroups()
one_group.group_edges = [fine_groups.group_edges[0], fine_groups.group_edges[-1]]

# Create a tally mesh
mesh = openmc.Mesh()
mesh.type = 'regular'
mesh.dimension = [3,3,1]
mesh.lower_left  = [-32.13, -32.13, -64.26]
mesh.width = [64.26/mesh.dimension[0], 42.84/mesh.dimension[1], 128.52/mesh.dimension[2]]

pin_mesh = openmc.Mesh()
pin_mesh.type = 'regular'
pin_mesh.dimension = [51,51,1]
pin_mesh.lower_left  = [-32.13, -32.13, -64.26]
pin_mesh.width = [64.26/mesh.dimension[0], 42.84/mesh.dimension[1], 128.52/mesh.dimension[2]]

unity_mesh = openmc.Mesh()
unity_mesh.type = mesh.type
unity_mesh.dimension = [1,1,1]
unity_mesh.lower_left  = mesh.lower_left
unity_mesh.width = [i*j for i,j in zip(mesh.dimension, mesh.width)]

# Instantiate a list of the delayed groups
delayed_groups = list(range(1,7))

# Create elements and ordered dicts and initialize to None
mgxs_dict = OrderedDict()


mgxs_uncertainty = OrderedDict()

# Populate the MGXS in the MGXS lib
for mgxs_type in mgxs_types:

    mgxs_uncertainty[mgxs_type] = np.zeros(len(sp_batches))

    if mgxs_type == 'diffusion-coefficient':
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=fine_groups, by_nuclide=False,
            name=mgxs_type)
    elif 'nu-scatter matrix' in mgxs_type:
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups, by_nuclide=False,
            name=mgxs_type)
        mgxs_dict[mgxs_type].correction = None
    elif mgxs_type == 'decay-rate':
        mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=one_group,
            delayed_groups=delayed_groups, by_nuclide=False,
            name=mgxs_type)
    elif mgxs_type == 'chi-prompt':
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups, by_nuclide=False,
            name=mgxs_type)
    elif mgxs_type in openmc.mgxs.MGXS_TYPES:
        mgxs_dict[mgxs_type] = openmc.mgxs.MGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups, by_nuclide=False,
            name=mgxs_type)
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
        mgxs_dict[mgxs_type]._estimator = 'tracklength'
    elif mgxs_type in openmc.mgxs.MDGXS_TYPES:
        mgxs_dict[mgxs_type] = openmc.mgxs.MDGXS.get_mgxs(
            mgxs_type, domain=mesh, domain_type='mesh',
            energy_groups=energy_groups,
            delayed_groups=delayed_groups, by_nuclide=False,
            name=mgxs_type)

# Generate a new tallies file
tallies_file = openmc.Tallies()

# Add the tallies to the file
for mgxs in mgxs_dict.values():
    tallies = mgxs.tallies.values()
    for tally in tallies:
        tallies_file.append(tally, True)

# Export the tallies file to xml
tallies_file.export_to_xml(run_directory + '/tallies.xml')

#openmc.run(threads=1, mpi_procs=36, mpi_exec='mpirun', cwd=run_directory)

###############################################################
#                   data processing
##############################################################

chi_delayed_no_mesh = np.array([[9.734825e-03, 9.881542e-01, 2.096576e-03, 1.279322e-05, 1.435046e-06, 1.371583e-07, 3.806795e-09],
                                [1.317222e-02, 9.784912e-01, 8.286263e-03, 4.696721e-05, 2.783692e-06, 4.664741e-07, 9.021934e-08],
                                [1.550382e-02, 9.770476e-01, 7.405128e-03, 4.025243e-05, 2.631446e-06, 5.238836e-07, 8.047936e-08],
                                [5.789806e-02, 9.374928e-01, 4.581016e-03, 2.599106e-05, 1.694224e-06, 3.328842e-07, 7.550656e-08],
                                [5.555028e-02, 9.415188e-01, 2.913182e-03, 1.622109e-05, 1.356798e-06, 1.390715e-07, 2.552194e-08],
                                [7.341913e-02, 9.243975e-01, 2.169580e-03, 1.303999e-05, 5.431774e-07, 2.202977e-07, 3.063495e-08]])

chi_delayed_otf_1122 = np.array([[9.704783e-03, 9.881272e-01, 2.149093e-03, 1.686983e-05, 2.048067e-06, 6.521984e-09, 0.000000e+00],
                                 [1.198646e-02, 9.785364e-01, 9.411030e-03, 6.075959e-05, 4.644908e-06, 6.988225e-07, 0.000000e+00],
                                 [1.618169e-02, 9.768547e-01, 6.930081e-03, 3.224398e-05, 1.303273e-06, 8.904960e-09, 1.205971e-13],
                                 [6.067789e-02, 9.346514e-01, 4.633779e-03, 3.498179e-05, 6.862805e-08, 1.914023e-06, 1.621486e-09],
                                 [5.868388e-02, 9.382919e-01, 3.007904e-03, 1.593041e-05, 4.163026e-07, 0.000000e+00, 0.000000e+00],
                                 [7.448721e-02, 9.233963e-01, 2.101319e-03, 1.429788e-05, 8.458049e-07, 5.517833e-15, 0.000000e+00]])

analog_global_10 = np.array([[6.265751e-03, 9.912280e-01, 2.506199e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                             [1.140865e-02, 9.813865e-01, 7.104748e-03, 1.000631e-04, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                             [1.769958e-02, 9.752641e-01, 7.036285e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                             [5.626745e-02, 9.386351e-01, 5.048867e-03, 4.856048e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                             [5.690954e-02, 9.392633e-01, 3.827184e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                             [6.830347e-02, 9.299255e-01, 1.518103e-03, 2.528980e-04, 0.000000e+00, 0.000000e+00, 0.000000e+00]])

analog_global_100 = np.array([[1.049783e-02, 9.873302e-01, 2.171966e-03, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                              [1.301227e-02, 9.790777e-01, 7.859675e-03, 5.031895e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                              [1.535682e-02, 9.769316e-01, 7.622983e-03, 8.863823e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                              [5.854500e-02, 9.367989e-01, 4.622233e-03, 1.936193e-05, 1.451851e-05, 0.000000e+00, 0.000000e+00],
                              [5.615090e-02, 9.405921e-01, 3.226548e-03, 3.043702e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00],
                              [7.302889e-02, 9.245155e-01, 2.430014e-03, 2.557500e-05, 0.000000e+00, 0.000000e+00, 0.000000e+00]])

analog_local_10 = np.array([[ 0.       , 1.       , 0.       , 0.       , 0.       , 0.       , 0.      ],
                            [ 0.       , 1.       , 0.       , 0.       , 0.       , 0.       , 0.      ],
                            [ 0.       , 1.       , 0.       , 0.       , 0.       , 0.       , 0.      ],
                            [ 0.061216 , 0.938784 , 0.       , 0.       , 0.       , 0.       , 0.      ],
                            [ 0.       , 1.       , 0.       , 0.       , 0.       , 0.       , 0.      ],
                            [ 0.0833   , 0.9167   , 0.       , 0.       , 0.       , 0.       , 0.      ]])

analog_local_100 = np.array([[ 0.021276 , 0.978724 , 0.       , 0.       , 0.       , 0.       , 0.      ],
                             [ 0.020512 , 0.969231 , 0.010257 , 0.       , 0.       , 0.       , 0.      ],
                             [ 0.018018 , 0.977477 , 0.004506 , 0.       , 0.       , 0.       , 0.      ],
                             [ 0.068321 , 0.929609 , 0.00207  , 0.       , 0.       , 0.       , 0.      ],
                             [ 0.045661 , 0.954339 , 0.       , 0.       , 0.       , 0.       , 0.      ],
                             [ 0.05494  , 0.94506  , 0.       , 0.       , 0.       , 0.       , 0.      ]])

tl_global_10 = np.array([[9.781782e-03, 9.881178e-01, 2.089036e-03, 1.142538e-05, 8.864954e-07, 5.971975e-08, 1.665943e-08],
                         [1.317101e-02, 9.785158e-01, 8.261397e-03, 4.891888e-05, 3.595489e-06, 5.821815e-07, 4.448714e-08],
                         [1.552558e-02, 9.770720e-01, 7.361479e-03, 3.942576e-05, 2.582287e-06, 5.651190e-07, 3.857399e-08],
                         [5.799225e-02, 9.374027e-01, 4.579703e-03, 2.569990e-05, 1.455925e-06, 3.217934e-07, 3.171641e-08],
                         [5.557518e-02, 9.415148e-01, 2.895997e-03, 1.571359e-05, 1.125806e-06, 1.554757e-07, 3.089813e-08],
                         [7.339171e-02, 9.244266e-01, 2.170706e-03, 1.301687e-05, 9.745855e-07, 2.534085e-08, 1.839035e-09]])

tl_global_100 = np.array([[9.754861e-03, 9.881526e-01, 2.080830e-03, 1.175409e-05, 7.733026e-07, 1.341036e-07, 5.230978e-08],
                          [1.319860e-02, 9.784559e-01, 8.296015e-03, 4.692805e-05, 3.355125e-06, 4.628784e-07, 1.095110e-07],
                          [1.553861e-02, 9.770302e-01, 7.389604e-03, 4.013478e-05, 2.572747e-06, 4.153812e-07, 1.411911e-07],
                          [5.798683e-02, 9.374146e-01, 4.573171e-03, 2.548919e-05, 1.748370e-06, 2.503972e-07, 7.890650e-08],
                          [5.550380e-02, 9.415832e-01, 2.898589e-03, 1.610868e-05, 1.092049e-06, 1.337361e-07, 4.733903e-08],
                          [7.339890e-02, 9.244216e-01, 2.169395e-03, 1.226129e-05, 8.334886e-07, 7.989607e-08, 3.527510e-08]])

tl_local_10 = np.array([[8.959130e-03, 9.885896e-01, 2.436623e-03, 9.671728e-06, 5.725020e-06, 0.000000e+00, 0.000000e+00],
                        [1.125081e-02, 9.800552e-01, 8.603275e-03, 9.128752e-05, 5.998982e-07, 0.000000e+00, 0.000000e+00],
                        [1.558544e-02, 9.771043e-01, 7.286304e-03, 2.295849e-05, 2.388643e-06, 0.000000e+00, 0.000000e+00],
                        [6.060020e-02, 9.349762e-01, 4.334691e-03, 9.065410e-05, 5.286473e-08, 0.000000e+00, 0.000000e+00],
                        [5.959332e-02, 9.374280e-01, 2.953782e-03, 2.721057e-05, 2.113665e-07, 0.000000e+00, 0.000000e+00],
                        [7.586069e-02, 9.221866e-01, 1.931301e-03, 1.933783e-05, 4.616370e-06, 0.000000e+00, 0.000000e+00]])

tl_local_100 = np.array([[9.704790e-03, 9.881280e-01, 2.149094e-03, 1.686984e-05, 2.048068e-06, 6.521989e-09, 0.000000e+00],
                        [1.198648e-02, 9.785375e-01, 9.411041e-03, 6.075965e-05, 4.644913e-06, 6.988233e-07, 0.000000e+00],
                        [1.618171e-02, 9.768560e-01, 6.930090e-03, 3.224402e-05, 1.303275e-06, 8.904972e-09, 1.205973e-13],
                        [6.067799e-02, 9.346529e-01, 4.633787e-03, 3.498185e-05, 6.862816e-08, 1.914026e-06, 1.621488e-09],
                        [5.868402e-02, 9.382942e-01, 3.007911e-03, 1.593045e-05, 4.163037e-07, 0.000000e+00, 0.000000e+00],
                        [7.448739e-02, 9.233985e-01, 2.101324e-03, 1.429791e-05, 8.458070e-07, 5.517846e-15, 0.000000e+00]])

analog_local_10_avg = np.average(np.abs((analog_local_10 - tl_global_100)))
analog_local_10_max = np.max(np.abs((analog_local_10 - tl_global_100)))
analog_global_10_avg = np.average(np.abs((analog_global_10 - tl_global_100)))
analog_global_10_max = np.max(np.abs((analog_global_10 - tl_global_100)))
tl_local_10_avg = np.average(np.abs((tl_local_10 - tl_global_100)))
tl_local_10_max = np.max(np.abs((tl_local_10 - tl_global_100)))

analog_local_100_avg = np.average(np.abs((analog_local_100 - tl_global_100)))
analog_local_100_max = np.max(np.abs((analog_local_100 - tl_global_100)))
analog_global_100_avg = np.average(np.abs((analog_global_100 - tl_global_100)))
analog_global_100_max = np.max(np.abs((analog_global_100 - tl_global_100)))
tl_local_100_avg = np.average(np.abs((tl_local_100 - tl_global_100)))
tl_local_100_max = np.max(np.abs((tl_local_100 - tl_global_100)))

print(analog_local_10_avg)
print(analog_local_10_max)
print(analog_global_10_avg)
print(analog_global_10_max)
print(tl_local_10_avg)
print(tl_local_10_max)

print(analog_local_100_avg)
print(analog_local_100_max)
print(analog_global_100_avg)
print(analog_global_100_max)
print(tl_local_100_avg)
print(tl_local_100_max)

chi_delayed = []
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

    print('Loading MGXS {} of {}'.format(i+1, len(sp_batches)))

    for mgxs_type,mgxs in mgxs_dict.items():
        mgxs.load_from_statepoint(sp)

        if mgxs_type == 'chi-delayed':
            print(mgxs.get_xs()[1122])
            chi_delayed.append(mgxs.get_xs())

#fuel_pins = np.array([i > 0 for i in np.sum(np.sum(chi_delayed[-1], axis=1), axis=1)])
#fuel_pins = np.repeat(fuel_pins, 7*6)
#fuel_pins.shape = (1156,6,7)

#for i,batch in enumerate(sp_batches):
#    chi_error = chi_delayed[i] - chi_delayed_no_mesh
#    chi_error = np.abs(fuel_pins * chi_error)
#    print(chi_error.max())
#    print(np.sum(chi_error) / (1056 * 6 * 7.))
#    print(chi_delayed[i][1122])

#import seaborn as sns
#sns.palplot(sns.color_palette("muted"))
#for p in range(34*33, 34*33+1):
#    plt.figure(figsize=(22,13))
#    for dg in range(6):
#        ax = plt.subplot(2,3,dg+1)
#        for i,batch in enumerate(sp_batches):
#            ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], chi_delayed[i][p][dg][::-1]), [0.]), where='pre', linewidth=4)

#        ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], chi_delayed_no_mesh[dg][::-1]), [0.]), where='pre', color='k', linestyle='--', linewidth=4)
#        plt.title('Delayed group {} spectrum'.format(dg+1))
#        plt.xlabel('outgoing energy group', fontsize=14)
#        plt.ylabel('probability', fontsize=14)

#        plt.legend(['{:d}M neutrons (local)'.format(int((i-inactive)*particles/1.e6)) for i in sp_batches] + ['10M neutrons (global)'], fontsize=12, frameon=True, fancybox=True, facecolor='white', edgecolor='black')
#        plt.xticks(range(1,8), range(7, 0, -1), fontsize=14)
#        plt.yticks(fontsize=14)
#        plt.xlim([0,8])
#        plt.ylim([1.e-9,1.e1])
#        plt.yscale('log')
#    plt.savefig('chi_delayed_pin_{:04d}.png'.format(p))
#    plt.close()


#import seaborn as sns
#sns.palplot(sns.color_palette("muted"))
#for p in range(34*33, 34*33+1):
#    plt.figure(figsize=(22,13))
#    for dg in range(6):
#        ax = plt.subplot(2,3,dg+1)
#        for i,batch in enumerate(sp_batches):
#            ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], chi_delayed[i][p][dg][::-1]), [0.]), where='pre', linewidth=4)

#        ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], chi_delayed_no_mesh[dg][::-1]), [0.]), where='pre', color='k', linestyle='--', linewidth=4)
#        ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], chi_delayed_otf_1122[dg][::-1]), [0.]), where='pre', color='r', linestyle='-', linewidth=4)
#        plt.title('Delayed group {} spectrum'.format(dg+1))
#        plt.xlabel('outgoing energy group', fontsize=14)
#        plt.ylabel('probability', fontsize=14)

#        plt.legend(['{:d}M neutrons (local-A)'.format(int((i-inactive)*particles/1.e6)) for i in sp_batches] + ['10M neutrons (global)'] + ['100M neutrons (local-TL)'], fontsize=12, frameon=True, fancybox=True, facecolor='white', edgecolor='black')
#        plt.xticks(range(1,8), range(7, 0, -1), fontsize=14)
#        plt.yticks(fontsize=14)
#        plt.xlim([0,8])
#        plt.ylim([1.e-9,1.e1])
#        plt.yscale('log')
#    plt.savefig('chi_delayed_otf_pin_{:04d}.png'.format(p))
#    plt.close()

import seaborn as sns
sns.palplot(sns.color_palette("muted"))
plt.figure(figsize=(22,13))
for dg in range(6):
    ax = plt.subplot(2,3,dg+1)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], analog_global_10[dg][::-1]), [0.]), where='pre', color='k', linestyle='-', linewidth=4)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], analog_local_10[dg][::-1]), [0.]), where='pre', color='r', linestyle='-', linewidth=4)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], tl_global_10[dg][::-1]), [0.]), where='pre', color='b', linestyle='-', linewidth=4)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], tl_local_10[dg][::-1]), [0.]), where='pre', color='g', linestyle='-', linewidth=4)
    plt.title('Delayed group {} spectrum'.format(dg+1))
    plt.xlabel('outgoing energy group', fontsize=14)
    plt.ylabel('probability', fontsize=14)
    plt.legend(['10M neutrons (global-A)', '10M neutrons (local-A)', '10M neutrons (global-TL)', '10M neutrons (local-TL)'], fontsize=12, frameon=True, fancybox=True, facecolor='white', edgecolor='black')
    plt.xticks(range(1,8), range(7, 0, -1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0,8])
    plt.ylim([1.e-9,1.e1])
    plt.yscale('log')
plt.savefig('chi_delayed_global_v_local_10M.png')
plt.close()

plt.figure(figsize=(22,13))
for dg in range(6):
    ax = plt.subplot(2,3,dg+1)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], analog_global_100[dg][::-1]), [0.]), where='pre', color='k', linestyle='-', linewidth=4)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], analog_local_100[dg][::-1]), [0.]), where='pre', color='r', linestyle='-', linewidth=4)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], tl_global_100[dg][::-1]), [0.]), where='pre', color='b', linestyle='-', linewidth=4)
    ax.step(np.arange(0.5,energy_groups.num_groups+2.5), np.append(np.append([0.], tl_local_100[dg][::-1]), [0.]), where='pre', color='g', linestyle='-', linewidth=4)
    plt.title('Delayed group {} spectrum'.format(dg+1))
    plt.xlabel('outgoing energy group', fontsize=14)
    plt.ylabel('probability', fontsize=14)
    plt.legend(['100M neutrons (global-A)', '100M neutrons (local-A)', '100M neutrons (global-TL)', '100M neutrons (local-TL)'], fontsize=12, frameon=True, fancybox=True, facecolor='white', edgecolor='black')
    plt.xticks(range(1,8), range(7, 0, -1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([0,8])
    plt.ylim([1.e-9,1.e1])
    plt.yscale('log')
plt.savefig('chi_delayed_global_v_local_100M.png')
plt.close()
