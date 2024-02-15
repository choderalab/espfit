from pint import UnitRegistry

# Define pint unit registry
ureg = UnitRegistry()
hartree = 1 * ureg.hartree
bohr = 1 * ureg.bohr
angstrom = 1 * ureg.angstrom

# Conversion factors
#HARTEE_TO_KCALPERMOL = 627.509
#BOHR_TO_ANGSTROMS = 0.529
HARTREE_TO_KCALPERMOL = hartree.to(ureg.kilocalorie/(ureg.avogadro_constant*ureg.mole)).magnitude
BOHR_TO_ANGSTROMS = bohr.to(ureg.angstrom).magnitude
