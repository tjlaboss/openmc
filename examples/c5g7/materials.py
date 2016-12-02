import openmc

###############################################################################
#                 Exporting to OpenMC materials.xml File
###############################################################################

materials = {}

# Instantiate some Materials and register the appropriate Nuclides
materials['UO2'] = openmc.Material(name='UO2')
materials['UO2'].set_density('sum')
materials['UO2'].add_nuclide('U235', 8.6500E-4 , 'ao')
materials['UO2'].add_nuclide('U238', 2.2250E-2 , 'ao')
materials['UO2'].add_element('O'   , 4.62200E-2, 'ao')

materials['MOX 4.3%'] = openmc.Material(name='MOX 4.3%')
materials['MOX 4.3%'].set_density('sum')
materials['MOX 4.3%'].add_nuclide('U235' , 5.0000E-5, 'ao')
materials['MOX 4.3%'].add_nuclide('U238' , 2.2100E-2, 'ao')
materials['MOX 4.3%'].add_nuclide('Pu238', 1.5000E-5, 'ao')
materials['MOX 4.3%'].add_nuclide('Pu239', 5.8000E-4, 'ao')
materials['MOX 4.3%'].add_nuclide('Pu240', 2.4000E-4, 'ao')
materials['MOX 4.3%'].add_nuclide('Pu241', 9.8000E-5, 'ao')
materials['MOX 4.3%'].add_nuclide('Pu242', 5.4000E-5, 'ao')
materials['MOX 4.3%'].add_nuclide('Am241', 1.3000E-5, 'ao')
materials['MOX 4.3%'].add_element('O'    , 4.6300E-2, 'ao')

materials['MOX 7.0%'] = openmc.Material(name='MOX 7.0%')
materials['MOX 7.0%'].set_density('sum')
materials['MOX 7.0%'].add_nuclide('U235' , 5.0000E-5, 'ao')
materials['MOX 7.0%'].add_nuclide('U238' , 2.2100E-2, 'ao')
materials['MOX 7.0%'].add_nuclide('Pu238', 2.4000E-5, 'ao')
materials['MOX 7.0%'].add_nuclide('Pu239', 9.3000E-4, 'ao')
materials['MOX 7.0%'].add_nuclide('Pu240', 3.9000E-4, 'ao')
materials['MOX 7.0%'].add_nuclide('Pu241', 1.5200E-4, 'ao')
materials['MOX 7.0%'].add_nuclide('Pu242', 8.4000E-5, 'ao')
materials['MOX 7.0%'].add_nuclide('Am241', 2.0000E-5, 'ao')
materials['MOX 7.0%'].add_element('O'    , 4.6300E-2, 'ao')

materials['MOX 8.7%'] = openmc.Material(name='MOX 8.7%')
materials['MOX 8.7%'].set_density('sum')
materials['MOX 8.7%'].add_nuclide('U235' , 5.0000E-5, 'ao')
materials['MOX 8.7%'].add_nuclide('U238' , 2.2100E-2, 'ao')
materials['MOX 8.7%'].add_nuclide('Pu238', 3.0000E-5, 'ao')
materials['MOX 8.7%'].add_nuclide('Pu239', 1.1600E-3, 'ao')
materials['MOX 8.7%'].add_nuclide('Pu240', 4.9000E-4, 'ao')
materials['MOX 8.7%'].add_nuclide('Pu241', 1.9000E-4, 'ao')
materials['MOX 8.7%'].add_nuclide('Pu242', 1.0500E-4, 'ao')
materials['MOX 8.7%'].add_nuclide('Am241', 2.5000E-5, 'ao')
materials['MOX 8.7%'].add_element('O'    , 4.6300E-2, 'ao')

materials['Moderator'] = openmc.Material(name='Moderator')
materials['Moderator'].set_density('sum')
materials['Moderator'].add_element('H', 2*3.3500E-2, 'ao')
materials['Moderator'].add_element('O',   3.3500E-2, 'ao')
materials['Moderator'].add_element('B',   2.7800E-5, 'ao')

materials['Al Clad'] = openmc.Material(name='Al Clad')
materials['Al Clad'].set_density('sum')
materials['Al Clad'].add_element('Al', 6.0000E-2, 'ao')

materials['Zr Clad'] = openmc.Material(name='Zr Clad')
materials['Zr Clad'].set_density('sum')
materials['Zr Clad'].add_element('Zr', 4.3000E-2, 'ao')

materials['Void'] = openmc.Material(name='Void')
materials['Void'].set_density('sum')
materials['Void'].add_element('He', 1.e-10, 'ao')

materials['Control Rod'] = openmc.Material(name='Control Rod')
materials['Control Rod'].set_density('sum')
materials['Control Rod'].add_nuclide('Ag107', 2.27105E-2, 'ao')
materials['Control Rod'].add_nuclide('Ag109', 2.27105E-2, 'ao')
materials['Control Rod'].add_nuclide('In115', 8.00080E-3, 'ao')
materials['Control Rod'].add_nuclide('Cd113', 2.72410E-3, 'ao')

materials['Fission Chamber'] = openmc.Material(name='Fission Chamber')
materials['Fission Chamber'].set_density('sum')
materials['Fission Chamber'].add_element('H'   , 2*3.3500E-2, 'ao')
materials['Fission Chamber'].add_element('O'   ,   3.3500E-2, 'ao')
materials['Fission Chamber'].add_element('B'   ,   2.7800E-5, 'ao')
#materials['Fission Chamber'].add_nuclide('U235',   1.0000E-8, 'ao')
