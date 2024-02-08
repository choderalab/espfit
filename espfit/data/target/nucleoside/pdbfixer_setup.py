"""
Create initial structures of nucleosides for MD simulation using pdbfixer.

Note
----
Structures are minimized with restrained heavy atoms to fix poorly assinged hydrogens 
for pyrimidine bases (i.e. cytidine and uridine).
"""
import click
from openmm import CustomExternalForce, LangevinMiddleIntegrator
from openmm.unit import *
from openmm.app import *
from pdbfixer import PDBFixer


def prep(inputfile):
    fixer = PDBFixer(filename=inputfile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(True)
    fixer.findMissingAtoms()
    fixer.addMissingHydrogens(7.0)  # default: 7
    PDBFile.writeFile(fixer.topology, fixer.positions, open('pdbfixer.pdb', 'w'))

    # http://docs.openmm.org/latest/userguide/application/02_running_sims.html#using-amber-files
    model = Modeller(fixer.topology, fixer.positions)
    ff = ForceField('amber/RNA.OL3.xml', 'implicit/gbn2.xml')
    system = ff.createSystem(model.topology, nonbondedMethod=NoCutoff, nonbondedCutoff=2*nanometers, constraints=HBonds, rigidWater=True)

    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 100.0*kilocalories_per_mole/angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i, atom in enumerate(model.topology.atoms()):
        if atom.element.symbol != "H":
            atom_crd = model.positions[i]
            force.addParticle(i, atom_crd.value_in_unit(nanometers))
    system.addForce(force)

    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(model.topology, system, integrator)
    simulation.context.setPositions(model.positions)
    
    # minimize: fix hydrogen positions
    simulation.minimizeEnergy(maxIterations=50)
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(model.topology, positions, open("pdbfixer_min.pdb", 'w'))   


@click.command()
@click.option('--inputfile', '-i', help='Input PDB file')
def cli(**kwargs):
    inputfile = kwargs['inputfile']
    prep(inputfile)


if __name__ == "__main__":
    cli()