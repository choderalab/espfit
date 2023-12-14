import logging
import espaloma as esp
from espaloma.data.dataset import GraphDataset
#logger = logging.getLogger(__name__)
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)



class CustomGraphDataset(GraphDataset):
    """Custom GraphDataset with additional support to manipulate DGL graphs

    Methods
    -------
    filter()
        Filter high energy conformers and ensure minimum number of conformers

    compute_baseline_forcefields()
        Compute energies and forces using baseline force fields 

    compute_

    """

    def __init__(self, graphs=None, reference_forcefield='openff-2.0.0', logging_level='info', RANDOM_SEED=2666):
        super(CustomGraphDataset, self).__init__()
        self.graphs = graphs
        self.reference_forcefield = reference_forcefield
        self.logging_level = logging_level
        self.random_seed = RANDOM_SEED


    def filter_high_energy_conformers(self):
        """
        Filter high energy conformers

        Parameters
        ----------
        max_relative_energy : float, default 0.1
            the maximum relative energy respect to minima
        min_conformers : int, default=3
            the minimium number of conformers per molecule entry

        Returns
        -------
        g : CustomGraphs
            returns filtered graphs
        """


        raise NotImplementedError



    def compute_baseline_forcefields(self, forcefield_list):
        """Compute energies and forces using baseline force fields
        """
        #raise NotImplementedError

        new_graphs = [ _compute_baseline_forcefields(g, forcefield_list) for g in self.graphs ]
        self.graphs = new_graphs



    def compute_am1bcc_elf10():
        """
        Compute partial charges
        """
        raise NotImplementedError


    def compute_relative_energy(self):
        """Compute relative energy
        """
        import copy

        new_graphs = []
        for g in self.graphs:
            _g = copy.deepcopy(g)
            _g.nodes['g'].data['u_ref_relative'] = _g.nodes['g'].data['u_ref'].detach().clone()
            _g.nodes['g'].data['u_ref_relative'] = (_g.nodes['g'].data['u_ref_relative'] - _g.nodes['g'].data['u_ref_relative'].mean(dim=-1, keepdims=True)).float()
            #_g.nodes['g'].data.pop('u_ref')
            new_graphs.append(_g)

        self.graphs = new_graphs



    def subtract_nonbonded_interactions(self):
        """Subtract nonbonded energies and forces

        Paramterers
        -----------
        """

        from espaloma.data.md import subtract_nonbonded_force
        new_graphs = [ subtract_nonbonded_force(g, forcefield=self.reference_forcefield, subtract_charges=True) for g in self.graphs ]
        self.graphs = new_graphs



    def reshape_conformation_size(self, n_confs=50):
        """Reshape conformation size

        This is a work around to handle different graph size (shape). DGL requires at least one dimension with same size. 
        Here, we will modify the graphs so that each graph has the same number of conformations instead fo concatenating 
        graphs into heterogenous graphs with the same number of conformations. This will allow batching and shuffling 
        during the training. 
        """
        #raise NotImplementedError
        import random
        import copy
        import torch

        new_graphs = []
        for i, g in enumerate(self.graphs):
            n = g.nodes['n1'].data['xyz'].shape[1]
            #logging.debug(f">{i}: {n} conformations")

            if n == n_confs:
                new_graphs.append(g.heterograph)

            elif n < n_confs:
                random.seed(self.random_seed)
                index = random.choices(range(0, n), k=n_confs-n)
                print(f"Randomly select {len(index)} conformers")            
                #logging.debug(f"Randomly select {len(index)} conformers")

                _g = copy.deepcopy(g)
                _g.nodes["g"].data["u_ref"] = torch.cat((_g.nodes['g'].data['u_ref'], _g.nodes['g'].data['u_ref'][:, index]), dim=-1)
                _g.nodes["n1"].data["xyz"] = torch.cat((_g.nodes['n1'].data['xyz'], _g.nodes['n1'].data['xyz'][:, index, :]), dim=1)
                _g.nodes['n1'].data['u_ref_prime'] = torch.cat((_g.nodes['n1'].data['u_ref_prime'], _g.nodes['n1'].data['u_ref_prime'][:, index, :]), dim=1)
                new_graphs.append(_g.heterograph)

            else:
                random.seed(self.random_seed)
                idx_range = random.sample(range(n), k=n)
                for j in range(n // n_confs + 1):
                    _g = copy.deepcopy(g)

                    if (j+1)*n_confs > n:
                        _index = range(j*n_confs, n)
                        random.seed(self.random_seed)
                        index = random.choices(range(0, n), k=(j+1)*n_confs-n)
                        print(f"Iteration {j}: Randomly select {len(index)} conformers")
                        #logging.debug(f"Iteration {j}: Randomly select {len(index)} conformers")

                        _g.nodes["g"].data["u_ref"] = torch.cat((_g.nodes['g'].data['u_ref'][:, index], _g.nodes['g'].data['u_ref'][:, _index]), dim=-1)
                        _g.nodes["n1"].data["xyz"] = torch.cat((_g.nodes['n1'].data['xyz'][:, index, :], _g.nodes['n1'].data['xyz'][:, _index, :]), dim=1)
                        _g.nodes["n1"].data["u_ref_prime"] = torch.cat((_g.nodes['n1'].data['u_ref_prime'][:, index, :], _g.nodes['n1'].data['u_ref_prime'][:, _index, :]), dim=1)       
                    else:            
                        idx1 = j*n_confs
                        idx2 = (j+1)*n_confs
                        _index = idx_range[idx1:idx2]
                        print(f"Iteration {j}: Extract indice from {idx1} to {idx2}")
                        #logging.debug(f"Iteration {j}: Extract indice from {idx1} to {idx2}")

                        _g.nodes["g"].data["u_ref"] = _g.nodes['g'].data['u_ref'][:, _index]
                        _g.nodes["n1"].data["xyz"] = _g.nodes['n1'].data['xyz'][:, _index, :]
                        _g.nodes["n1"].data["u_ref_prime"] = _g.nodes['n1'].data['u_ref_prime'][:, _index, :]
                    
                    new_graphs.append(_g.heterograph)        

        self.graphs = new_graphs


    def _compute_baseline_energy_force(g, forcefield_list):
        """Calculate baseline energy using legacy forcefields
        
        reference:
        https://github.com/choderalab/espaloma/espaloma/data/md.py
        """

        from openmm import openmm, unit
        from openmm.app import Simulation
        from openmm.unit import Quantity
        from openmmforcefields.generators import SystemGenerator

        # Simulation Specs (not important, just place holders)
        TEMPERATURE = 350 * unit.kelvin
        STEP_SIZE = 1.0 * unit.femtosecond
        COLLISION_RATE = 1.0 / unit.picosecond
        EPSILON_MIN = 0.05 * unit.kilojoules_per_mole

        for forcefield in forcefield_list:
            if forcefield in ['gaff-1.81', 'gaff-2.11', 'openff-1.2.0', 'openff-2.0.0']:
                generator = SystemGenerator(
                    small_molecule_forcefield=forcefield,
                    molecules=[g.mol],
                    forcefield_kwargs={"constraints": None, "removeCMMotion": False},
                )
                suffix = forcefield
            elif forcefield in ['amber14-all.xml']:
                generator = SystemGenerator(
                    forcefields=[forcefield],
                    molecules=[g.mol],
                    forcefield_kwargs={"constraints": None, "removeCMMotion": False},
                )
                suffix = "amber14"
            else:
                raise Exception('force field not supported')


            # parameterize topology
            topology = g.mol.to_topology().to_openmm()

            # create openmm system
            system = generator.create_system(
                topology,
            )

            # use langevin integrator, although it's not super useful here
            integrator = openmm.LangevinIntegrator(
                TEMPERATURE, COLLISION_RATE, STEP_SIZE
            )

            # create simulation
            simulation = Simulation(
                topology=topology, system=system, integrator=integrator
            )

            # get energy
            us = []
            us_prime = []
            xs = (
                Quantity(
                    g.nodes["n1"].data["xyz"].detach().numpy(),
                    esp.units.DISTANCE_UNIT,
                )
                .value_in_unit(unit.nanometer)
                .transpose((1, 0, 2))
            )
            for x in xs:
                simulation.context.setPositions(x)
                us.append(
                    simulation.context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(esp.units.ENERGY_UNIT)
                )
                us_prime.append(
                    simulation.context.getState(getForces=True)
                    .getForces(asNumpy=True)
                    .value_in_unit(esp.units.FORCE_UNIT) * -1
                )

            #us = torch.tensor(us)[None, :]
            us = torch.tensor(us, dtype=torch.float64)[None, :]
            us_prime = torch.tensor(
                np.stack(us_prime, axis=1),
                dtype=torch.get_default_dtype(),
            )

            g.nodes['g'].data['u_%s' % suffix] = us
            g.nodes['n1'].data['u_%s_prime' % suffix] = us_prime

            return g