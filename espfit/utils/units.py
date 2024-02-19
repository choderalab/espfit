import openmm.unit as unit
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


def convert_string_to_unit(unit_string):
    """Convert a unit string to a openmm unit object.
    
    Parameters
    ----------
    unit_string : str
        The string representation of the unit.

    Returns
    -------
    openmm.unit
        The openmm unit object.
    """
    unit_mapping = {
        "nanometer": unit.nanometer,
        "angstrom": unit.angstrom,
        "nanometers": unit.nanometers,
        "angstroms": unit.angstroms,
        "kelvin": unit.kelvin,
        "molar": unit.molar,
        "millimolar": unit.millimolar,
        "micromolar": unit.micromolar,
        "atomsphere": unit.atmosphere,
        "bar": unit.bar,
        "nanoseconds": unit.nanoseconds,
        "picoseconds": unit.picoseconds,
        "femtoseconds": unit.femtoseconds,
        "nanosecond": unit.nanosecond,
        "picosecond": unit.picosecond,
        "femtosecond": unit.femtosecond,
        # Add more units as needed
    }
    if unit_string in unit_mapping:
        return unit_mapping[unit_string]
    else:
        raise ValueError(f"Unit '{unit_string}' is not recognized.")