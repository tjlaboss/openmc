import openmc
from settings import settings_file, mpi_procs
from plots import plots
from tallies import tallies, geometry, cells, universes


###############################################################################
#                      Simulation Input File Parameters
###############################################################################

cells['Control Rod Base Bank 1'].translation = [0., 0., 64.26]
cells['Control Rod Base Bank 2'].translation = [0., 0., 0.]
cells['Control Rod Base Bank 3'].translation = [0., 0., 0.]
cells['Control Rod Base Bank 4'].translation = [0., 0., 0.]

# Create the geometry file
geometry.export_to_xml()

# Create the materials file
materials_file = openmc.Materials(geometry.get_all_materials())
materials_file.export_to_xml()

# Create the plots file
plot_file = openmc.Plots(plots.values())
plot_file.export_to_xml()

# Create the settings file
settings_file.export_to_xml()

# Create the tallies file
tallies_file = openmc.Tallies(tallies.values())
tallies_file.export_to_xml()
