import numpy as np

import openmc
import openmc.mgxs

###############################################################################
#                      Simulation Input File Parameters
###############################################################################

# OpenMC simulation parameters
batches = 20
inactive = 10
particles = 1000

###############################################################################
#                 Exporting to OpenMC mgxs.xml file
###############################################################################

# Instantiate the energy group data
groups = openmc.mgxs.EnergyGroups(group_edges=[
    1e-5, 0.0635, 10.0, 1.0e2, 1.0e3, 0.5e6, 1.0e6, 20.0e6])

transport      = np.array([1.77949E-01, 3.29805E-01, 4.80388E-01, 5.54367E-01, 3.11801E-01, 3.95168E-01, 5.64406E-01])
absorption     = np.array([8.02480E-03, 3.71740E-03, 2.67690E-02, 9.62360E-02, 3.00200E-02, 1.11260E-01, 2.82780E-01])
capture        = np.array([8.12740E-04, 2.89810E-03, 2.03158E-02, 7.76712E-02, 1.22116E-02, 2.82252E-02, 6.67760E-02])
velocity       = np.array([2.23466E+09, 5.07347E+08, 3.86595E+07, 5.13931E+06, 1.67734E+06, 7.28603E+05, 2.92902E+05])
fission        = np.array([7.21206E-03, 8.19301E-04, 6.45320E-03, 1.85648E-02, 1.78084E-02, 8.30348E-02, 2.16004E-01])
nu             = np.array([2.78145E+00, 2.47443E+00, 2.43383E+00, 2.43380E+00, 2.43380E+00, 2.43380E+00, 2.43380E+00])
nu_fission     = nu * fission
chi            = np.array([5.87910E-01, 4.11760E-01, 3.39060E-04, 1.17610E-07, 0.00000E+00, 0.00000E+00, 0.00000E+00])
beta           = np.array([2.13333E-04, 1.04514E-03, 6.03969E-04, 1.33963E-03, 2.29386E-03, 7.05174E-04, 6.00381E-04, 2.07736E-04])
decay_constant = np.array([1.247E-02, 2.829E-02, 4.252E-02, 1.330E-01, 2.925E-01, 6.665E-01, 1.635E+00, 3.555E+00])
chi_delayed    = np.array([[0.00075, 0.98512, 0.01413, 0.00000, 0.00000, 0.00000, 0.00000],
                           [0.03049, 0.96907, 0.00044, 0.00000, 0.00000, 0.00000, 0.00000],
                           [0.00457, 0.97401, 0.02142, 0.00000, 0.00000, 0.00000, 0.00000],
                           [0.02002, 0.97271, 0.00727, 0.00000, 0.00000, 0.00000, 0.00000],
                           [0.05601, 0.93818, 0.00581, 0.00000, 0.00000, 0.00000, 0.00000],
                           [0.06098, 0.93444, 0.93444, 0.00000, 0.00000, 0.00000, 0.00000],
                           [0.10635, 0.88298, 0.01067, 0.00000, 0.00000, 0.00000, 0.00000],
                           [0.09346, 0.90260, 0.00394, 0.00000, 0.00000, 0.00000, 0.00000]])
scatter        = np.array([[[1.27537E-01, 4.23780E-02, 9.43740E-06, 5.51630E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
                            [0.00000E+00, 3.24456E-01, 1.63140E-03, 3.14270E-09, 0.00000E+00, 0.00000E+00, 0.00000E+00],
                            [0.00000E+00, 0.00000E+00, 4.50940E-01, 2.67920E-03, 0.00000E+00, 0.00000E+00, 0.00000E+00],
                            [0.00000E+00, 0.00000E+00, 0.00000E+00, 4.52565E-01, 5.56640E-03, 0.00000E+00, 0.00000E+00],
                            [0.00000E+00, 0.00000E+00, 0.00000E+00, 1.25250E-04, 2.71401E-01, 1.02550E-02, 1.00210E-08],
                            [0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 1.29680E-03, 2.65802E-01, 1.68090E-02],
                            [0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 0.00000E+00, 8.54580E-03, 2.73080E-01]]])


# Instantiate the 7-group (C5G7) cross section data
uo2_xsdata = openmc.XSdata('UO2', groups)
uo2_xsdata.order = 0
uo2_xsdata.num_delayed_groups = 8
uo2_xsdata.set_total(transport)
uo2_xsdata.set_absorption(absorption)
uo2_xsdata.set_scatter_matrix(np.rollaxis(scatter, 0, 3))
uo2_xsdata.set_fission(fission)
uo2_xsdata.set_nu_fission(nu_fission)
uo2_xsdata.set_chi(chi)
uo2_xsdata.set_beta(beta)
#uo2_xsdata.set_inverse_velocity(1.0 / velocity)
#uo2_xsdata.set_decay_rate(decay_constant)
#uo2_xsdata.set_chi_delayed(chi_delayed)

h2o_xsdata = openmc.XSdata('LWTR', groups)
h2o_xsdata.order = 0
h2o_xsdata.set_total([0.15920605, 0.412969593, 0.59030986, 0.58435,
                      0.718, 1.2544497, 2.650379])
h2o_xsdata.set_absorption([6.0105E-04, 1.5793E-05, 3.3716E-04,
                           1.9406E-03, 5.7416E-03, 1.5001E-02,
                           3.7239E-02])
scatter_matrix = np.array(
    [[[0.0444777, 0.1134000, 0.0007235, 0.0000037, 0.0000001, 0.0000000, 0.0000000],
      [0.0000000, 0.2823340, 0.1299400, 0.0006234, 0.0000480, 0.0000074, 0.0000010],
      [0.0000000, 0.0000000, 0.3452560, 0.2245700, 0.0169990, 0.0026443, 0.0005034],
      [0.0000000, 0.0000000, 0.0000000, 0.0910284, 0.4155100, 0.0637320, 0.0121390],
      [0.0000000, 0.0000000, 0.0000000, 0.0000714, 0.1391380, 0.5118200, 0.0612290],
      [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0022157, 0.6999130, 0.5373200],
      [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.1324400, 2.4807000]]])
scatter_matrix = np.rollaxis(scatter_matrix, 0, 3)
h2o_xsdata.set_scatter_matrix(scatter_matrix)

mg_cross_sections_file = openmc.MGXSLibrary(groups, 8)
mg_cross_sections_file.add_xsdatas([uo2_xsdata, h2o_xsdata])
mg_cross_sections_file.export_to_hdf5()


###############################################################################
#                 Exporting to OpenMC materials.xml file
###############################################################################

# Instantiate some Macroscopic Data
uo2_data = openmc.Macroscopic('UO2')
h2o_data = openmc.Macroscopic('LWTR')

# Instantiate some Materials and register the appropriate Macroscopic objects
uo2 = openmc.Material(material_id=1, name='UO2 fuel')
uo2.set_density('macro', 1.0)
uo2.add_macroscopic(uo2_data)

water = openmc.Material(material_id=2, name='Water')
water.set_density('macro', 1.0)
water.add_macroscopic(h2o_data)

# Instantiate a Materials collection and export to XML
materials_file = openmc.Materials([uo2, water])
materials_file.cross_sections = "./mgxs.h5"
materials_file.export_to_xml()


###############################################################################
#                 Exporting to OpenMC geometry.xml file
###############################################################################

# Instantiate ZCylinder surfaces
fuel_or = openmc.ZCylinder(surface_id=1, x0=0, y0=0, R=0.54, name='Fuel OR')
left = openmc.XPlane(surface_id=4, x0=-0.63, name='left')
right = openmc.XPlane(surface_id=5, x0=0.63, name='right')
bottom = openmc.YPlane(surface_id=6, y0=-0.63, name='bottom')
top = openmc.YPlane(surface_id=7, y0=0.63, name='top')

left.boundary_type = 'reflective'
right.boundary_type = 'reflective'
top.boundary_type = 'reflective'
bottom.boundary_type = 'reflective'

# Instantiate Cells
fuel = openmc.Cell(cell_id=1, name='cell 1')
moderator = openmc.Cell(cell_id=2, name='cell 2')

# Use surface half-spaces to define regions
fuel.region = -fuel_or
moderator.region = +fuel_or & +left & -right & +bottom & -top

# Register Materials with Cells
fuel.fill = uo2
moderator.fill = water

# Instantiate Universe
root = openmc.Universe(universe_id=0, name='root universe')

# Register Cells with Universe
root.add_cells([fuel, moderator])

# Instantiate a Geometry, register the root Universe, and export to XML
geometry = openmc.Geometry(root)
geometry.export_to_xml()


###############################################################################
#                   Exporting to OpenMC settings.xml file
###############################################################################

# Instantiate a Settings object, set all runtime parameters, and export to XML
settings_file = openmc.Settings()
settings_file.energy_mode = "multi-group"
settings_file.batches = batches
settings_file.inactive = inactive
settings_file.particles = particles

# Create an initial uniform spatial source distribution over fissionable zones
bounds = [-0.63, -0.63, -1, 0.63, 0.63, 1]
uniform_dist = openmc.stats.Box(bounds[:3], bounds[3:])
settings_file.source = openmc.source.Source(space=uniform_dist)

settings_file.export_to_xml()

###############################################################################
#                   Exporting to OpenMC tallies.xml file
###############################################################################

# Instantiate a tally mesh
mesh = openmc.Mesh(mesh_id=1)
mesh.type = 'regular'
mesh.dimension = [100, 100, 1]
mesh.lower_left = [-0.63, -0.63, -1.e50]
mesh.upper_right = [0.63, 0.63, 1.e50]

# Instantiate some tally Filters
energy_filter = openmc.EnergyFilter([1e-5, 0.0635, 10.0, 1.0e2, 1.0e3, 0.5e6,
                                     1.0e6, 20.0e6])
mesh_filter = openmc.MeshFilter(mesh)

# Instantiate the Tally
tally = openmc.Tally(tally_id=1, name='tally 1')
tally.filters = [energy_filter, mesh_filter]
tally.scores = ['flux', 'fission', 'nu-fission']

# Instantiate a Tallies collection, register all Tallies, and export to XML
tallies_file = openmc.Tallies([tally])
tallies_file.export_to_xml()
