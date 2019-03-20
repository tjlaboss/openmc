import openmc
from materials import materials
from surfaces import surfaces

###############################################################################
#                 Exporting to OpenMC geometry.xml File
###############################################################################

cells = {}
universes = {}

rings = ['Base', 'Inner Void', 'Inner Clad', 'Outer Void', 'Outer Clad', 'Moderator']

# Instantiate Cells
for mat in ['UO2']:
    for bank in [1,4]:
        univ_name = '{} Bank {}'.format(mat, bank)
        universes[univ_name] = openmc.Universe(name=univ_name)
        for ring in rings:
            name = '{} {} Bank {}'.format(mat, ring, bank)
            cells[name] = openmc.Cell(name=name)
            universes[univ_name].add_cell(cells[name])

        cells['{} {} Bank {}'.format(mat, 'Base', bank)].region       = -surfaces['Fuel OR']
        cells['{} {} Bank {}'.format(mat, 'Inner Void', bank)].region = +surfaces['Fuel OR'] & -surfaces['Fuel Inner Clad IR']
        cells['{} {} Bank {}'.format(mat, 'Inner Clad', bank)].region = +surfaces['Fuel Inner Clad IR'] & -surfaces['Fuel Inner Clad OR']
        cells['{} {} Bank {}'.format(mat, 'Outer Void', bank)].region = +surfaces['Fuel Inner Clad OR'] & -surfaces['Fuel Outer Clad IR']
        cells['{} {} Bank {}'.format(mat, 'Outer Clad', bank)].region = +surfaces['Fuel Outer Clad IR'] & -surfaces['Fuel Outer Clad OR']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].region  = +surfaces['Fuel Outer Clad OR']
        cells['{} {} Bank {}'.format(mat, 'Base', bank)].fill       = materials[mat]
        cells['{} {} Bank {}'.format(mat, 'Inner Void', bank)].fill = materials['Void']
        cells['{} {} Bank {}'.format(mat, 'Inner Clad', bank)].fill = materials['Zr Clad']
        cells['{} {} Bank {}'.format(mat, 'Outer Void', bank)].fill = materials['Void']
        cells['{} {} Bank {}'.format(mat, 'Outer Clad', bank)].fill = materials['Al Clad']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].fill  = materials['Moderator Bank {}'.format(bank)]


for mat in ['MOX 4.3%', 'MOX 7.0%', 'MOX 8.7%']:
    for bank in [2,3]:
        univ_name = '{} Bank {}'.format(mat, bank)
        universes[univ_name] = openmc.Universe(name=univ_name)
        for ring in rings:
            name = '{} {} Bank {}'.format(mat, ring, bank)
            cells[name] = openmc.Cell(name=name)
            universes[univ_name].add_cell(cells[name])

        cells['{} {} Bank {}'.format(mat, 'Base', bank)].region       = -surfaces['Fuel OR']
        cells['{} {} Bank {}'.format(mat, 'Inner Void', bank)].region = +surfaces['Fuel OR'] & -surfaces['Fuel Inner Clad IR']
        cells['{} {} Bank {}'.format(mat, 'Inner Clad', bank)].region = +surfaces['Fuel Inner Clad IR'] & -surfaces['Fuel Inner Clad OR']
        cells['{} {} Bank {}'.format(mat, 'Outer Void', bank)].region = +surfaces['Fuel Inner Clad OR'] & -surfaces['Fuel Outer Clad IR']
        cells['{} {} Bank {}'.format(mat, 'Outer Clad', bank)].region = +surfaces['Fuel Outer Clad IR'] & -surfaces['Fuel Outer Clad OR']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].region  = +surfaces['Fuel Outer Clad OR']
        cells['{} {} Bank {}'.format(mat, 'Base', bank)].fill       = materials[mat]
        cells['{} {} Bank {}'.format(mat, 'Inner Void', bank)].fill = materials['Void']
        cells['{} {} Bank {}'.format(mat, 'Inner Clad', bank)].fill = materials['Zr Clad']
        cells['{} {} Bank {}'.format(mat, 'Outer Void', bank)].fill = materials['Void']
        cells['{} {} Bank {}'.format(mat, 'Outer Clad', bank)].fill = materials['Al Clad']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].fill  = materials['Moderator Bank {}'.format(bank)]

rings = ['Base', 'Clad', 'Moderator']
mats = ['Guide Tube']
for mat in mats:
    for bank in range(1,5):
        for ring in rings:
            name = '{} {} Bank {}'.format(mat, ring, bank)
            cells[name] = openmc.Cell(name=name)

        cells['{} {} Bank {}'.format(mat, 'Base', bank)].region      = -surfaces['Guide Tube IR'] & -surfaces['Axial Midplane']
        cells['{} {} Bank {}'.format(mat, 'Clad', bank)].region      = +surfaces['Guide Tube IR'] & -surfaces['Guide Tube OR'] & -surfaces['Axial Midplane']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].region = +surfaces['Guide Tube OR'] & -surfaces['Axial Midplane']

        cells['{} {} Bank {}'.format(mat, 'Base', bank)].fill      = materials['Moderator']
        cells['{} {} Bank {}'.format(mat, 'Clad', bank)].fill      = materials['Al Clad']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].fill = materials['Moderator Bank {}'.format(bank)]

rings = ['Base', 'Clad', 'Moderator']
mats = ['Fission Chamber']
for mat in mats:
    for bank in range(0,5):
        univ_name = '{} Bank {}'.format(mat, bank)
        universes[univ_name] = openmc.Universe(name=univ_name)
        for ring in rings:
            name = '{} {} Bank {}'.format(mat, ring, bank)
            cells[name] = openmc.Cell(name=name)
            universes[univ_name].add_cell(cells[name])

        cells['{} {} Bank {}'.format(mat, 'Base', bank)].region      = -surfaces['Guide Tube IR']
        cells['{} {} Bank {}'.format(mat, 'Clad', bank)].region      = +surfaces['Guide Tube IR'] & -surfaces['Guide Tube OR']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].region = +surfaces['Guide Tube OR']

        cells['{} {} Bank {}'.format(mat, 'Base', bank)].fill      = materials[mat]
        cells['{} {} Bank {}'.format(mat, 'Clad', bank)].fill      = materials['Al Clad']
        cells['{} {} Bank {}'.format(mat, 'Moderator', bank)].fill = materials['Moderator Bank {}'.format(bank)]

rings = ['Base', 'Core', 'Core Clad', 'Core Moderator']
mats = ['Control Rod']
for mat in mats:
    for bank in range(1,5):
        univ_name = '{} {} Bank {}'.format(mat, 'Core', bank)
        universes[univ_name] = openmc.Universe(name=univ_name)
        for ring in rings:
            name = '{} {} Bank {}'.format(mat, ring, bank)
            cells[name] = openmc.Cell(name=name)
            if ring != 'Base':
                universes[univ_name].add_cell(cells[name])

        universes[univ_name].add_cell(cells['Guide Tube Base Bank {}'.format(bank)])
        universes[univ_name].add_cell(cells['Guide Tube Clad Bank {}'.format(bank)])
        universes[univ_name].add_cell(cells['Guide Tube Moderator Bank {}'.format(bank)])

        cells['{} {} Bank {}'.format(mat, 'Core', bank)].region           = -surfaces['Guide Tube IR'] & +surfaces['Axial Midplane']
        cells['{} {} Bank {}'.format(mat, 'Core Clad', bank)].region      = +surfaces['Guide Tube IR'] & -surfaces['Guide Tube OR'] & +surfaces['Axial Midplane']
        cells['{} {} Bank {}'.format(mat, 'Core Moderator', bank)].region = +surfaces['Guide Tube OR'] & +surfaces['Axial Midplane']

        cells['{} {} Bank {}'.format(mat, 'Core', bank)].fill           = materials[mat]
        cells['{} {} Bank {}'.format(mat, 'Core Clad', bank)].fill      = materials['Al Clad']
        cells['{} {} Bank {}'.format(mat, 'Core Moderator', bank)].fill = materials['Moderator Bank {}'.format(bank)]

        cells['{} {} Bank {}'.format(mat, 'Base', bank)].fill = universes['{} {} Bank {}'.format(mat, 'Core', bank)]
        univ_name = '{} {} Bank {}'.format(mat, 'Base', bank)
        universes[univ_name] = openmc.Universe(name=univ_name)
        universes[univ_name].add_cell(cells['{} {} Bank {}'.format(mat, 'Base', bank)])

rings = ['Reflector', 'Reflector Clad', 'Reflector Moderator']
mats = ['Control Rod']
for mat in mats:
    univ_name = '{} Reflector'.format(mat)
    universes[univ_name] = openmc.Universe(name=univ_name)
    for ring in rings:
        name = '{} {}'.format(mat, ring)
        cells[name] = openmc.Cell(name=name)
        universes[univ_name].add_cell(cells[name])

    cells['{} {}'.format(mat, 'Reflector')].region           = -surfaces['Guide Tube IR']
    cells['{} {}'.format(mat, 'Reflector Clad')].region      = +surfaces['Guide Tube IR'] & -surfaces['Guide Tube OR']
    cells['{} {}'.format(mat, 'Reflector Moderator')].region = +surfaces['Guide Tube OR']

    cells['{} {}'.format(mat, 'Reflector')].fill           = materials[mat]
    cells['{} {}'.format(mat, 'Reflector Clad')].fill      = materials['Al Clad']
    cells['{} {}'.format(mat, 'Reflector Moderator')].fill = materials['Moderator']

cells['Moderator'] = openmc.Cell(name='Moderator')
cells['Moderator'].fill = materials['Moderator']
universes['Moderator'] = openmc.Universe(name='Moderator')
universes['Moderator'].add_cell(cells['Moderator'])
