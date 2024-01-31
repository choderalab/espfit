"""
Create MD system and perform MD simulation.

Notes
-----
Use Interchange to create biopolymer-ligand system when it supports RNA systems. Currently (2024-01-09), OpenFF Toolkit identifies residues 
by matching chemical substructures rather than by residue name, it currently only supports the 20 'canonical' amino acids.

TODO
----
"""
import openmm.app as app
import openmm.unit as unit
import logging

_logger = logging.getLogger(__name__)


class BaseSimulation(object):
    """Base class for MD sampler.

    Methods
    -------
    minimize(maxIterations=100):
        Minimize solvated system.
    
    run(checkpoint_frequency=25000, logging_frequency=250000, netcdf_frequency=250000, nsteps=250000, atom_indices=None):
        Run standard MD simulation.

    export_xml(exportSystem=True, exportState=True, exportIntegrator=True):
        Export serialized system XML file and solvated pdb file.
    """
    def __init__(self, output_prefix='examples/sampler'):
        self.output_prefix = output_prefix        


    def minimize(self, maxIterations=100):
        """Minimize solvated system.

        Parameters
        ----------
        maxIterations : int, default=100
            Maximum number of iterations to perform.

        Returns
        -------
        None
        """
        _logger.info(f"Minimizing system...")
        self.simulation.minimizeEnergy(maxIterations)


    def run(self, checkpoint_frequency=25000, logging_frequency=250000, netcdf_frequency=250000, nsteps=250000, atom_indices=None):
        """Run standard MD simulation.

        Parameters
        ----------
        checkpoint_frequency : int, default=25000 (1 ns)
            Frequency (in steps) at which to write checkpoint files.

        logging_frequency : int, default=250000 (10 ns)
            Frequency (in steps) at which to write logging files.

        netcdf_frequency : int, default=250000 (10 ns)
            Frequency (in steps) at which to write netcdf files.

        nsteps : int, default=250000 (10 ns)
            Number of steps to run the simulation.

        atom_indices : list, default=None
            List of atom indices to save. If None, save all atoms except water and ions.

        Returns
        -------
        None
        """
        self.checkpoint_frequency = checkpoint_frequency
        self.logging_frequency = logging_frequency
        self.netcdf_frequency = netcdf_frequency
        self.nsteps = nsteps
        self.atom_indices = atom_indices

        # Selet atoms to save
        import mdtraj as md
        if self.atom_indices is None:
            self.atom_indices = []
            mdtop = md.Topology.from_openmm(self.simulation.topology)
            res = [ r for r in mdtop.residues if r.name not in ("HOH", "NA", "CL", "K") ]
            for r in res:
                for a in r.atoms:
                    self.atom_indices.append(a.index)

        # Define reporter
        import os
        from mdtraj.reporters import NetCDFReporter
        from openmm.app import CheckpointReporter, StateDataReporter
        self.simulation.reporters.append(NetCDFReporter(os.path.join(self.output_prefix, 'traj.nc'), self.netcdf_frequency, atomSubset=self.atom_indices))
        self.simulation.reporters.append(CheckpointReporter(os.path.join(self.output_prefix, 'checkpoint.chk'), self.checkpoint_frequency))
        self.simulation.reporters.append(StateDataReporter(os.path.join(self.output_prefix, 'reporter.log'), self.logging_frequency, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))
        # Run
        _logger.info(f"Run MD simulation for {self.nsteps} steps")
        self.simulation.step(self.nsteps)


    def export_xml(self, exportSystem=True, exportState=True, exportIntegrator=True):
        """Export serialized system XML file and solvated pdb file.

        TODO
        ----
        * Currently, the output filenames are hard-coded. Should output filenames be specified by users?

        Parameters
        ----------
        exportSystem : bool, default=True
            Whether to export system XML file.

        exportState : bool, default=True
            Whether to export state XML file.

        exportIntegrator : bool, default=True
            Whether to export integrator XML file.

        Returns
        -------
        None
        """
        _logger.info(f"Serialize and export system")

        import os
        from openmm import XmlSerializer
        state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)

        # Save system
        if exportSystem:
            outfile = os.path.join(self.output_prefix, "system.xml")
            with open(f"{outfile}", "w") as wf:
                xml = XmlSerializer.serialize(self.simulation.system)
                wf.write(xml)

        # Save and serialize the final state
        if exportState:
            outfile = os.path.join(self.output_prefix, "state.xml")
            with open(f"{outfile}", "w") as wf:
                xml = XmlSerializer.serialize(state)
                wf.write(xml)
            # Save the final state as a PDB
            outfile = os.path.join(self.output_prefix, "state.pdb")
            with open(f"{outfile}", "w") as wf:
                app.PDBFile.writeFile(
                    self.simulation.topology,
                    self.simulation.context.getState(
                        getPositions=True,
                        enforcePeriodicBox=False).getPositions(),
                        #enforcePeriodicBox=True).getPositions(),
                        file=wf,
                        keepIds=True
                )

        # Save and serialize integrator
        if exportIntegrator:
            outfile = os.path.join(self.output_prefix, "integrator.xml")
            with open(f"{outfile}", "w") as wf:
                xml = XmlSerializer.serialize(self.simulation.integrator)
                wf.write(xml)


class SetupSampler(BaseSimulation):
    """Create biopolymer-ligand system.

    Use espaloma force field as default to self-consistently parameterize the biopolymer-ligand system.
    Use Perses 0.10.1 default parameter settings to setup the system.

    Parameters
    ----------
    small_molecule_forcefield : str, optional
        The force field to be used for small molecules. Default is 'openff-2.1.0'.
    forcefield_files : list, optional
        List of force field files. Default is ['amber14-all.xml'].
    water_model : str, optional
        The water model to be used. Default is 'tip3p'.
    solvent_padding : Quantity, optional
        The padding distance around the solute in the solvent box. Default is 9.0 * unit.angstroms.
    ionic_strength : Quantity, optional
        The ionic strength of the solvent. Default is 0.15 * unit.molar.
    constraints : object, optional
        The type of constraints to be applied to the system. Default is app.HBonds.
    hmass : Quantity, optional
        The mass of the hydrogen atoms. Default is 3.0 * unit.amu.
    temperature : Quantity, optional
        The temperature of the system. Default is 300.0 * unit.kelvin.
    pressure : Quantity, optional
        The pressure of the system. Default is 1.0 * unit.atmosphere.
    pme_tol : float, optional
        The Ewald error tolerance for PME electrostatics. Default is 2.5e-04.
    nonbonded_method : object, optional
        The nonbonded method to be used for the system. Default is app.PME.
    barostat_period : int, optional
        The frequency at which the barostat is applied. Default is 50.
    timestep : Quantity, optional
        The integration timestep. Default is 4 * unit.femtoseconds.
    override_with_espaloma : bool, optional
        Whether to override the original parameters with espaloma. Default is True.

    Methods
    -------
    create_system(biopolymer_file=None, ligand_file=None):
        Create biopolymer-ligand system and export serialized system XML file and solvated pdb file.

    Examples
    --------
    >>> from espfit.app.sampler import SetupSampler
    >>> c = SetupSampler()
    >>> c.create_system(biopolymer_file='protein.pdb', ligand_file='ligand.sdf')
    >>> c.minimize()
    >>> c.run()

    Notes
    -----
    For some reason, the following forcefield files fail to construct systems for test systems stored in `espfit/data/target-debug`:

    ['amber14-all.xml', 'amber/phosaa14SB.xml']             : pl-multi: NG, pl-single: NG, RNA: NG
    ['amber/protein.ff14SB.xml', 'amber/phosaa14SB.xml']    : pl-multi: NG, pl-single: NG, RNA: NG
    ['amber14-all.xml']                                     : pl-multi: NG, pl-single: OK, RNA: OK
    ['amber/protein.ff14SB.xml', 'amber/RNA.OL3.xml']       : pl-multi: NG, pl-single: OK, RNA: OK
    """
    def __init__(self, 
                 small_molecule_forcefield='openff-2.1.0',                  
                 forcefield_files = ['amber/ff14SB.xml', 'amber/phosaa14SB.xml'], # pl-multi: OK, pl-single: OK, RNA: OK
                 water_model='tip3p', 
                 solvent_padding=9.0 * unit.angstroms, 
                 ionic_strength=0.15 * unit.molar, 
                 constraints=app.HBonds, 
                 hmass=3.0 * unit.amu, 
                 temperature=300.0 * unit.kelvin, 
                 pressure=1.0 * unit.atmosphere, 
                 pme_tol=2.5e-04, 
                 nonbonded_method=app.PME, 
                 barostat_period=50, 
                 timestep=4 * unit.femtoseconds, 
                 override_with_espaloma=True,
                 ):
        super(SetupSampler, self).__init__()
        self.small_molecule_forcefield = small_molecule_forcefield
        self.water_model = water_model
        self.forcefield_files = self._update_forcefield_files(forcefield_files)
        self.solvent_padding = solvent_padding
        self.ionic_strength = ionic_strength
        self.constraints = constraints
        self.hmass = hmass
        self.temperature = temperature
        self.pressure = pressure
        self.pme_tol = pme_tol
        self.nonbonded_method = nonbonded_method
        self.barostat_period = barostat_period
        self.timestep = timestep
        self.override_with_espaloma = override_with_espaloma


    def _update_forcefield_files(self, forcefield_files):
        """Get forcefield files.

        Update `forcefield_files` depending on the type of water model.

        Returns
        -------
        updated_forcefield_files : list
            List of forcefield files
        """
        # 3-site water models
        if self.water_model == 'tip3p':
            forcefield_files.append(['amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml'])
        elif self.water_model == 'tip3pfb':
            self.water_model = 'tip3p'
            forcefield_files.append(['amber/tip3pfb_standard.xml', 'amber/tip3pfb_HFE_multivalent.xml'])
        elif self.water_model == 'spce':
            self.water_model = 'tip3p'
            forcefield_files.append(['amber/spce_standard.xml', 'amber/spce_HFE_multivalent.xml'])
        elif self.water_model == 'opc3':
            raise NotImplementedError('see https://github.com/choderalab/rna-espaloma/blob/main/experiment/nucleoside/script/create_system_espaloma.py#L366')
        # 4-site water models
        elif self.water_model == 'tip4pew':
            self.water_model = 'tip4pew'
            forcefield_files.append(['amber/tip4pew_standard.xml', 'amber/tip4pew_HFE_multivalent.xml'])
        elif self.water_model == 'tip4pfb':
            self.water_model = 'tip4pew'
            forcefield_files.append(['amber/tip4pfb_standard.xml', 'amber/tip4pfb_HFE_multivalent.xml'])
        elif self.water_model == 'opc':
            self.water_model = 'tip4pew'
            forcefield_files.append(['amber/opc_standard.xml'])
        else:
            raise NotImplementedError(f'Water model {self.water_model} is not supported.')
        # Flatten list
        new_forcefield_files = []
        for f in forcefield_files:
            if isinstance(f, list):
                new_forcefield_files.extend(f)
            else:
                new_forcefield_files.append(f)

        return new_forcefield_files


    def create_system(self, biopolymer_file=None, ligand_file=None):
        """
        Create biopolymer-ligand system and export serialized system XML file and solvated pdb file.

        Parameters
        ---------
        ligand_file : str
            ligand sdf file. The first ligand entry will be used if multiple ligands are stored.

        Returns
        -------
        None
        """
        import os
        import numpy as np
        import mdtraj as md
        from rdkit import Chem
        from openmmforcefields.generators import SystemGenerator
        from openff.toolkit.topology import Molecule
        from openmm import MonteCarloBarostat
        from openmm import LangevinMiddleIntegrator

        if biopolymer_file is None and ligand_file is None:
            raise ValueError("At least one biopolymer (.pdb) or ligand (.sdf) file must be provided")
        
        # Load biopolymer (biopolymer, rna) pdb
        if biopolymer_file is not None:
            with open(biopolymer_file, 'r') as f:
                biopolymer = app.PDBFile(f)  # biopolymer.positions is openmm.unit.quantity.Quantity
                # TODO: is this necessary?
                if biopolymer.positions.unit != unit.nanometers:
                    raise Warning(f"biopolymer positions unit is expected to be nanometers but got {biopolymer.positions.unit}")
            # Set topology and positions
            if ligand_file is None:
                complex_topology = biopolymer.topology
                complex_positions = biopolymer.positions
        # Load ligand
        if ligand_file is not None:
            ext = os.path.splitext(ligand_file)[-1].lower()
            assert ext == '.sdf', f'Ligand file format must be SDF but got {ext}'
            suppl = Chem.SDMolSupplier(ligand_file)
            mols = [ x for x in suppl ]
            mol = mols[0]
            mol.SetProp("_Name", "MOL")
            offmol = Molecule.from_rdkit(mol)   
            #offmol = Molecule.from_file(ligand_file, allow_undefined_stereo=True)   # Is this better?
            ligand_positions = offmol.conformers[0]   # ligand.position is pint.util.Quantity            
            ligand_positions = ligand_positions.to_openmm()
            ligand_positions = ligand_positions.in_units_of(unit.nanometers)
            ligand_topology = offmol.to_topology().to_openmm()
            # Set topology and positions
            if biopolymer_file is None:
                complex_topology = ligand_topology
                complex_positions = ligand_positions
        # Merge biopolymer and ligand
        if biopolymer_file is not None and ligand_file is not None:
            _logger.debug("Merge biopolymer-ligand topology")
            # Convert openmm topology to mdtraj topology
            biopolymer_md_topology = md.Topology.from_openmm(biopolymer.topology)
            ligand_md_topology = md.Topology.from_openmm(ligand_topology)
            
            # Merge topology
            complex_md_topology = biopolymer_md_topology.join(ligand_md_topology)
            complex_topology = complex_md_topology.to_openmm()
            
            # Get number of atoms
            n_atoms_total = complex_topology.getNumAtoms()
            n_atoms_biopolymer = biopolymer.topology.getNumAtoms()
            n_atoms_ligand = ligand_topology.getNumAtoms()
            assert n_atoms_total == n_atoms_biopolymer + n_atoms_ligand, "Number of atoms after merging the biopolymer and ligand topology does not match"
            _logger.debug(f"Total atoms: {n_atoms_total} (biopolymer: {n_atoms_biopolymer}, ligand: {n_atoms_ligand})")
            
            # Complex positons: do we need to ensure the units to be the same? Or will it automatically convert to nanometers if the units are different?
            # Currently, ligand position units are converted to nanometers before combining the positions
            complex_positions = unit.Quantity(np.zeros([n_atoms_total, 3]), unit=unit.nanometers)
            complex_positions[:n_atoms_biopolymer, :] = biopolymer.positions
            complex_positions[n_atoms_biopolymer:n_atoms_biopolymer+n_atoms_ligand, :] = ligand_positions

        # Initialize system generator
        forcefield_kwargs = {'removeCMMotion': True, 'ewaldErrorTolerance': self.pme_tol, 'constraints' : self.constraints, 'rigidWater': True, 'hydrogenMass' : self.hmass}
        periodic_forcefield_kwargs = {'nonbondedMethod': self.nonbonded_method}
        barostat = MonteCarloBarostat(self.pressure, self.temperature, self.barostat_period)
        if ligand_file is not None:
            _logger.debug("Initialize system generator for biopolymer-ligand or ligand only system")
            self._system_generator = SystemGenerator(
                forcefields=self.forcefield_files, forcefield_kwargs=forcefield_kwargs, periodic_forcefield_kwargs = periodic_forcefield_kwargs, barostat=barostat, 
                small_molecule_forcefield=self.small_molecule_forcefield, molecules=offmol, cache=None)
        else:
            _logger.debug("Initialize system generator for biopolymer system")
            self._system_generator = SystemGenerator(
                forcefields=self.forcefield_files, forcefield_kwargs=forcefield_kwargs, periodic_forcefield_kwargs = periodic_forcefield_kwargs, barostat=barostat, 
                cache=None)

        # Solvate system
        _logger.info("Solvating system...")
        modeller = app.Modeller(complex_topology, complex_positions)
        modeller.addSolvent(self._system_generator.forcefield, model=self.water_model, padding=self.solvent_padding, ionicStrength=self.ionic_strength)
        
        # Create system
        self._solvated_topology = modeller.getTopology()
        self._solvated_positions = modeller.getPositions()
        self._solvated_system = self._system_generator.create_system(self._solvated_topology)

        # Regenerate system if espaloma is used
        if "espaloma" in self.small_molecule_forcefield and self.override_with_espaloma == True:
            _logger.info("Regenerate system with espaloma. This will overwride the original parameters with espaloma.")
            self._new_solvated_system, self._new_solvated_topology = self._regenerate_espaloma_system()
        else:
            self._new_solvated_system = self._solvated_system
            self._new_solvated_topology = self._solvated_topology

        # Create simulation
        self.integrator = LangevinMiddleIntegrator(self.temperature, 1/unit.picosecond, self.timestep)
        self.simulation = app.Simulation(self._new_solvated_topology, self._new_solvated_system, self.integrator)
        self.simulation.context.setPositions(self._solvated_positions)


    def _regenerate_espaloma_system(self):
        """Regenerate system with espaloma. Parameterization of biopolymer and ligand self-consistently.

        Reference
        ---------
        https://github.com/kntkb/perses/blob/support-protein-espaloma/perses/app/relative_setup.py#L883

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        import os
        import glob
        import mdtraj as md
        from openff.toolkit import Molecule

        _logger.info("Regenerate system with espaloma")

        # Check biopolymer chains
        mdtop = md.Topology.from_openmm(self._solvated_topology)
        chain_indices = [ chain.index for chain in self._solvated_topology.chains() ]
        #chain_indices = [ chain.index for chain in mdtop.chains ]
        #biopolymer_chain_indices = [ chain_index for chain_index in chain_indices if mdtop.select(f"biopolymer and chainid == {chain_index}").any() ]
        biopolymer_chain_indices = [ chain_index for chain_index in chain_indices if mdtop.select(f"(water or resname NA or resname K or resname CL) and chainid == {chain_index}").any() ]
        _logger.info(f"biopolymer chain indices: {biopolymer_chain_indices}")

        # Check conflicting residue names. Espaloma will use residue name "XX".
        conflict_resnames = [ residue.name for residue in mdtop.residues if residue.name.startswith("XX") ]
        if conflict_resnames:
            raise Exception('Found conflict residue name in biopolymer.')

        # Initialize
        self._new_solvated_topology = app.Topology()
        self._new_solvated_topology.setPeriodicBoxVectors(self._solvated_topology.getPeriodicBoxVectors())
        new_atoms = {}

        # Regenerate biopolymer topology
        chain_counter = 0
        _logger.info(f"Regenerating biopolymer topology...")
        for chain in self._solvated_topology.chains():
            new_chain = self._new_solvated_topology.addChain(chain.id)
            # Convert biopolymer into a single residue
            if chain.index in biopolymer_chain_indices:
                resname = f'XX{chain_counter:01d}'
                resid = '1'
                chain_counter += 1
                new_residue = self._new_solvated_topology.addResidue(resname, new_chain, resid)
            #for i, residue in enumerate(chain.residues()):
            for residue in chain.residues():
                if residue.chain.index not in biopolymer_chain_indices:
                    new_residue = self._new_solvated_topology.addResidue(residue.name, new_chain, residue.id)
                for atom in residue.atoms():
                    new_atom = self._new_solvated_topology.addAtom(atom.name, atom.element, new_residue, atom.id)
                    new_atoms[atom] = new_atom

        # Regenerate bond information
        for bond in self._solvated_topology.bonds():
            if bond[0] in new_atoms and bond[1] in new_atoms:
                self._new_solvated_topology.addBond(new_atoms[bond[0]], new_atoms[bond[1]])
        
        # Save the updated complex model as pdb
        complex_espaloma_filename = f"complex-solvated-espaloma.pdb"
        if not os.path.exists(complex_espaloma_filename):
            with open(complex_espaloma_filename, 'w') as outfile:
                app.PDBFile.writeFile(self._new_solvated_topology, self._solvated_positions, outfile)
        
        # Seperate biopolymers into indivdual pdb files according to chain ID.
        biopolymer_espaloma_filenames = glob.glob("biopolymer-espaloma-*.pdb")
        if not biopolymer_espaloma_filenames:
            for chain_index in biopolymer_chain_indices:
                t = md.load_pdb(complex_espaloma_filename)
                indices = t.topology.select(f"chainid == {chain_index}")
                t.atom_slice(indices).save_pdb(f"biopolymer-espaloma-{chain_index}.pdb")
            biopolymer_espaloma_filenames = glob.glob("biopolymer-espaloma-*.pdb")
        
        # Load individual biopolymer structure into openff.toolkit.topology.Molecule
        biopolymer_molecules = [ Molecule.from_file(biopolymer_filename) for biopolymer_filename in biopolymer_espaloma_filenames ]
        
        # We already added small molecules to template generator when we first created ``self._system_generator``.
        # So we only need to add biopolymer molecule to template generator (EspalomaTemplateGenerator).
        self._system_generator.template_generator.add_molecules(biopolymer_molecules)
        # Regenerate system with system generator.
        self._new_solvated_system = self._system_generator.create_system(self._new_solvated_topology)

        return self._new_solvated_system, self._new_solvated_topology


    def _update_topology(self):
        """Update topology to reflect the new system.
        """
        pass


    def load_from_xml(cls):
        """Load serialized system XML file and solvated pdb file.
        """
        raise NotImplementedError