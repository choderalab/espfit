"""
Custom GraphDataset with additional support to manipulate DGL graphs.

TODO
----
* Use type hint?
* Add option to keep minimum energy conformer for each chunked dataset in `reshape_conformation_size()`
"""

import logging
from espaloma.data.dataset import GraphDataset

_logger = logging.getLogger(__name__)


class CustomGraphDataset(GraphDataset):
    """Custom GraphDataset with additional support to manipulate DGL graphs.

    Methods
    -------
    drop_and_merge_duplicates(save_merged_dataset=True, dataset_name='misc', output_directory_path=None):
        Drop and merge duplicate nonisomeric smiles across different data sources.

    subtract_nonbonded_interactions(subtract_vdw=False, subtract_ele=True):
        Subtract nonbonded interactions from QC reference.

    filter_high_energy_conformers(relative_energy_threshold=0.1, node_feature='u_ref'):
        Filter high energy conformers and ensure minimum number of conformers.

    filter_minimum_conformers(n_conformer_threshold=3):
        Filter molecules with conformers below given threshold.

    compute_baseline_energy_force(forcefield_list=['openff-2.0.0']):
        Compute energies and forces using other force fields.

    reshape_conformation_size(n_confs=50):
        Reshape conformation size.
    
    compute_relative_energy():
        Compute relative energy for both QM and MM energies with the mean set to zero.

    Notes
    -----
    DGL graph assumes two node types, `g` and `n1`, which represents the molecular and atomic level.
    The quantum chemical energies and forces are stored in `g.nodes['g'].data['u_ref']` and 
    `g.nodes['n1'].data['u_ref_prime']`, respectively. Note that `u_ref` and `u_ref_prime` will be 
    updated in-place as graphs are processed with CustomGraphDataset methods.

    Examples
    --------
    >>> from espaloma.data.dataset import GraphDataset
    >>> path = 'espfit/data/qcdata/before-ff-calc/protein-torsion-sm'
    >>> ds = GraphDataset.load(path)
    >>> # Drop and merge duplicate molecules. Save merged dataset as a new dataset.
    >>> # If `output_directory_path` is None, then the current working directory is used.
    >>> ds.drop_and_merge_duplicates(save_merged_dataset=True, dataset_name='misc', output_directory_path=None)
    >>> # Subtract nonbonded energies and forces from QC reference (e.g. subtract all valence and ele interactions)
    >>> # This will update u_ref and u_ref_relative in-place. copy of raw u_ref (QM reference) will be copied to u_qm.
    >>> ds.subtract_nonbonded_interactions(subtract_vdw=False, subtract_ele=True)
    >>> # Filter high energy conformers (u_qm: QM reference before nonbonded interations are subtracted)
    >>> ds.filter_high_energy_conformers(relative_energy_threshold=0.1, node_feature='u_qm')
    >>> # Filter high energy conformers (u_ref: QM reference after nonbonded interactions are subtracted)
    >>> ds.filter_high_energy_conformers(relative_energy_threshold=0.1, node_feature='u_ref')
    >>> # Filter conformers below certain number
    >>> ds.filter_minimum_conformers(n_conformer_threshold=3)
    >>> # Compute energies and forces using other force fields
    >>> ds.compute_baseline_energy_force(forcefield_list=['openff-2.0.0'])
    >>> # Regenerate improper torsions in-place
    >>> from espaloma.graphs.utils.regenerate_impropers import regenerate_impropers
    >>> ds.apply(regenerate_impropers, in_place=True)
    >>> # Reshape conformation size
    >>> ds.reshape_conformation_size(n_confs=50)
    >>> # Compute relative energy. QM and MM energies mean are set to zero.
    >>> ds.compute_relative_energy()
    """


    def __init__(self, graphs=[], reference_forcefield='openff-2.0.0', random_seed=2666):
        """Construct custom GraphDataset instance to prepare QC dataset for espaloma training.

        Parameters
        ----------
        graphs : list of espaloma.graphs.graph.Graph, default=[]
            DGL graphs loaded from `espaloma.data.dataset.GraphDataset.load`.
             
        reference_forcefield : str, default=openff-2.0.0
            Reference force field used to compute force field parameters if not present in espaloma.
            The default behavior is to compute the LJ parameters with `reference_forcefield`.
        
        random_seed : int, default=2666
            Random seed used throughout the instance.
        """
        super(CustomGraphDataset, self).__init__()
        self.graphs = graphs
        self.reference_forcefield = reference_forcefield
        self.random_seed = random_seed


    def drop_and_merge_duplicates(self, save_merged_dataset=True, dataset_name='misc', output_directory_path=None):
        """Drop and merge duplicate nonisomeric smiles across different data sources.

        Modifies list of esp.Graph's in place.

        Parameters
        ----------
        save_merged_datest : boolean, default=True
            If True, then merged datasets will be saved as a new dataset.
        
        dataset_name : str, default=misc
            Name of the merged dataset.

        output_directory_path : str, default=None
            Output directory path to save the merged dataset. 
            If None, then the current working directory is used.
        
        Returns
        -------
        None
        """
        import os
        import pandas as pd

        if output_directory_path == None:
            output_directory_path = os.getcwd()

        _logger.info(f'Drop and merge duplicate smiles')
        smiles = [ g.mol.to_smiles(isomeric=False, explicit_hydrogens=True, mapped=False) for g in self.graphs ]
        _logger.info(f'Found {len(smiles)} molecules')

        # Unique entries
        df = pd.DataFrame.from_dict({'smiles': smiles})
        unique_index = df.drop_duplicates(keep=False).index.to_list()
        unique_graphs = [self.graphs[_idx] for _idx in unique_index]
        _logger.info(f'Found {len(unique_index)} unique molecules')

        # Duplicated entries
        index = df.duplicated(keep=False)   # Mark all duplicate entries True
        duplicated_index = df[index].index.to_list()
        _logger.info(f'Found {len(duplicated_index)} duplicated molecules')
        
        # Get unique smiles and assign new molecule name `e.g. mol0001`
        duplicated_df = df.iloc[duplicated_index]
        duplicated_smiles = duplicated_df.smiles.unique().tolist()
        molnames = [ f'mol{i:04d}' for i in range(len(duplicated_smiles)) ]
        _logger.info(f'Found {len(molnames)} unique molecules within duplicate entries')

        # Merge duplicate entries into a new single graph
        duplicated_graphs = []
        molnames_dict = {}
        for molname, duplicated_smile in zip(molnames, duplicated_smiles):
            # Map new molecule name with its unique smiles and dataframe indices
            index = duplicated_df[duplicated_df['smiles'] == duplicated_smile].index.tolist()
            molnames_dict[molname] = {'smiles': duplicated_smiles, 'index': index}
            # Merge graphs
            g = self._merge_graphs([self.graphs[_idx] for _idx in index])
            duplicated_graphs.append(g)
            # Save graphs (optional)
            if save_merged_dataset == True:
                # Notes: Create a temporary directory, `_output_directory_path`, to support pytest in test_utils_graphs.py.
                # Temporary directory needs to be created beforehand for `test_drop_and_merge_duplicates`.
                _output_directory_path = os.path.join(output_directory_path, dataset_name)
                os.makedirs(_output_directory_path, exist_ok=True)
                output_directory_path = os.path.join(_output_directory_path, molname)
                g.save(output_directory_path)

        # Update in place
        new_graphs = unique_graphs + duplicated_graphs
        _logger.info(f'Graph dataset reconstructed: {len(new_graphs)} unique molecules')
        self.graphs = new_graphs
        del unique_graphs, duplicated_graphs, df, duplicated_df


    def subtract_nonbonded_interactions(self, subtract_vdw=False, subtract_ele=True):
        """Subtract nonbonded interactions from QC reference.

        Modifies list of esp.Graph's in place. g.nodes['g'].data['u_qm'] and g.nodes['g'].data['u_qm_prime'] will be 
        cloned from `u_ref` and `u_ref_prime`, respectively, to book keep raw QC data.
    
        Parameters
        ----------
        subtract_vdw : boolean, default=False
            Subtract van der Waals interactions from QM reference.
            If subtracted, vdw parameters should be refitted during espaloma training.
        
        subtract_ele : boolean, default=True
            Subtract electrostatic interactions from QM reference.
            If subtracted, partial charges should be refitted during espaloma training.

        Notes
        -----
        subtract_vdw=False, substract_ele=False:
            Fit valence terms only.

        subtract_vdw=False, subtract_ele=True:
            Fit valence terms and partial charges (electrostatic terms).
        
        subtract_vdw=True, subtract_ele=False:
            Fit valence and vdw terms (THIS IS NOT SUPPORTED).

        subtract_vdw=True, subtract_ele=True:
            Fit both valence and nonbonded terms (THIS IS NOT SUPPORTED).

        Returns
        -------
        None
        """
        new_graphs = []
        from espaloma.data.md import subtract_nonbonded_force

        for i, g in enumerate(self.graphs):
            # `espaloma.data.md.subtract_nonbonded_force` will update g.nodes['g'].data['u_ref'] and g.nodes['g'].data['u_ref_prime'] in place. 
            # Clone QM reference into g.nodes['g'].data['u_qm'] and g.nodes['g'].data['u_qm_prime'], if not exist
            if 'u_qm' not in g.nodes['g'].data.keys():
                g.nodes['g'].data['u_qm'] = g.nodes['g'].data['u_ref'].detach().clone()
            if 'u_qm_prime' not in g.nodes['g'].data.keys():
                g.nodes['n1'].data['u_qm_prime'] = g.nodes['n1'].data['u_ref_prime'].detach().clone()

            if subtract_vdw==False and subtract_ele==True:
                # Note that current (espaloma version <=0.3.2) cannot properly pass reference force field to subtract_coulomb_force() when 
                # called with subtract_nonbonded_force() (see ref[1]). subtract_coulomb_force(g) should be subtract_coulomb_force(g, forcefield).
                # 
                # However, this is not problematic if the partial charges are precomputed and stored for each molecule (see ref[2]).
                # subtract_nonbonded_force() will return the coulomb interactions using the predefined partial charges.
                #
                # Reference:
                # [1] https://github.com/choderalab/espaloma/blob/main/espaloma/data/md.py#L503C19-L503C19
                # [2] https://github.com/openmm/openmmforcefields/blob/637d551a4408cc6145529cd9dc30e267f4178367/openmmforcefields/generators/template_generators.py#L1432
                g = subtract_nonbonded_force(g, forcefield=self.reference_forcefield, subtract_charges=True)
            elif subtract_vdw == False and subtract_ele == False:
                g = subtract_nonbonded_force(g, forcefield=self.reference_forcefield, subtract_charges=False)
            else:
                raise Exception(f'Current option is not supported (subtract_vdw={subtract_vdw}, subtract_ele={subtract_ele})')
            new_graphs.append(g)
        
        # Update in place
        self.graphs = new_graphs
        del new_graphs
            

    def filter_high_energy_conformers(self, relative_energy_threshold=0.1, node_feature=None):
        """Filter high energy conformers.

        Modifies list of esp.Graph's in place.
    
        Parameters
        ----------
        relative_energy_threshold : float, default=0.1 (unit: hartee)
            The maximum relative energy respect to minima.
        
        node_feature : str, default=None
            Node feature name that is referred to when filtering the conformers.
            Usually, this should be `u_ref` or `u_qm` which are stored under node type `g`.
        
        Returns
        -------
        None
        """
        if node_feature == None:
            raise Exception(f'Please specify the node feature name under node type `g`')

        new_graphs = []
        for i, g in enumerate(self.graphs):
            # Get indices smaller than the relative energy threshold
            index = g.nodes['g'].data[node_feature] <= g.nodes['g'].data[node_feature].min() + relative_energy_threshold
            index = index.flatten()
            for key in g.nodes['g'].data.keys():
                # g.nodes['g'].data['u_ref']: (1, n_conformers)
                if key.startswith('u_'):    
                    g.nodes['g'].data[key] = g.nodes['g'].data[key][:, index]
            for key in g.nodes['n1'].data.keys():
                # g.nodes['n1'].data['xyz']: (n_atoms, n_conformers, 3)
                if key.startswith('u_') or key.startswith('xyz'):
                    g.nodes['n1'].data[key] = g.nodes['n1'].data[key][:, index, :]
            new_graphs.append(g)
        
        # Update in place
        self.graphs = new_graphs
        del new_graphs


    def filter_minimum_conformers(self, n_conformer_threshold=3):
        """Filter molecules with conformers below given threshold.

        Modifies list of esp.Graph's in place.
    
        Parameters
        ----------        
        n_conformer_threshold : int, default=3
            The minimium number of conformers per entry.

        Returns
        -------
        None
        """
        new_graphs = []
        for i, g in enumerate(self.graphs):
            n_confs = g.nodes['n1'].data['xyz'].shape[1]
            if n_confs >= n_conformer_threshold:
                new_graphs.append(g)

        # Update in place
        self.graphs = new_graphs
        del new_graphs


    def compute_baseline_energy_force(self, forcefield_list=['openff-2.0.0']):
        """Compute energies and forces using other force fields.

        New node features are added to g.nodes['g']. For example, g.nodes['g'].data['u_openff-2.0.0'] and 
        g.nodes['n1'].data['u_openff-2.0.0_prime'] will be created for energies and forces, respectively.
        
        Parameters
        ----------
        forcefield_list : list, default=['openff-2.0.0']
            Currently supports the following force fields:
            'gaff-1.81', 'gaff-2.11', 'openff-1.2.0', 'openff-2.0.0', 'openff-2.1.0', 
            'amber14-all.xml', 'amber/protein.ff14SBonlysc.xml'

            >>> g = CustomGraphDataset()
            >>> g.available_forcefields

            In general, it supports all force fields that can be loaded by openmmforcefields.generators.SystemGenerator.
                    
        References
        ----------
        [1] https://github.com/choderalab/espaloma/espaloma/data/md.py
        [2] https://github.com/choderalab/refit-espaloma/blob/main/openff-default/02-train/merge-data/script/calc_ff.py  

        Returns
        -------
        None
        """
        import torch
        import numpy as np
        from espaloma import units as espunits
        from openmm import openmm, unit
        from openmm.app import Simulation
        from openmm.unit import Quantity
        from openmmforcefields.generators import SystemGenerator

        # Simulation Specs (not important, just place holders)
        TEMPERATURE = 350 * unit.kelvin
        STEP_SIZE = 1.0 * unit.femtosecond
        COLLISION_RATE = 1.0 / unit.picosecond

        if not all(_ in self.available_forcefields for _ in forcefield_list):
            raise Exception(f'{forcefield} force field not supported. Supported force fields are {SUPPORTED_FORCEFIELD_LIST}.')

        new_graphs = []
        for i, g in enumerate(self.graphs):
            for forcefield in forcefield_list:
                if forcefield.startswith('gaff') or forcefield.startswith('openff'):
                    generator = SystemGenerator(
                        small_molecule_forcefield=forcefield,
                        molecules=[g.mol],
                        forcefield_kwargs={"constraints": None, "removeCMMotion": False},
                    )
                    name = forcefield
                elif forcefield.startswith('amber') or forcefield.startswith('protein'):
                    generator = SystemGenerator(
                        forcefields=[forcefield],
                        molecules=[g.mol],
                        forcefield_kwargs={"constraints": None, "removeCMMotion": False},
                    )
                    if forcefield == 'amber14-all.xml':
                        name = 'amber14sb'
                    elif forcefield == 'amber/protein.ff14SBonlysc.xml':
                        name = 'amber14sb_onlysc'
                else:
                    import warnings
                    warnings.warn(f'{forcefield} not supported for molecule {g.mol.to_smiles()}')
                
                suffix = name

                # Parameterize topology
                topology = g.mol.to_topology().to_openmm()
                # Create openmm system
                system = generator.create_system(topology)
                # Use langevin integrator, although it's not super useful here
                integrator = openmm.LangevinIntegrator(TEMPERATURE, COLLISION_RATE, STEP_SIZE)
                # Create simulation
                simulation = Simulation(topology=topology, system=system, integrator=integrator)
                # Get energy
                us = []
                us_prime = []
                xs = (
                    Quantity(
                        g.nodes["n1"].data["xyz"].detach().numpy(),
                        espunits.DISTANCE_UNIT,
                    )
                    .value_in_unit(unit.nanometer)
                    .transpose((1, 0, 2))
                )
                for x in xs:
                    simulation.context.setPositions(x)
                    us.append(
                        simulation.context.getState(getEnergy=True)
                        .getPotentialEnergy()
                        .value_in_unit(espunits.ENERGY_UNIT)
                    )
                    us_prime.append(
                        simulation.context.getState(getForces=True)
                        .getForces(asNumpy=True)
                        .value_in_unit(espunits.FORCE_UNIT) * -1
                    )

                us = torch.tensor(us, dtype=torch.float64)[None, :]
                us_prime = torch.tensor(
                    np.stack(us_prime, axis=1),
                    dtype=torch.get_default_dtype(),
                )

                g.nodes['g'].data['u_%s' % suffix] = us
                g.nodes['n1'].data['u_%s_prime' % suffix] = us_prime

            new_graphs.append(g)

        # Update in place
        self.graphs = new_graphs
        del new_graphs


    def compute_relative_energy(self):
        """Compute relative MM energy wiht the mean set to zero.

        Relative energy will be overwritten and stored in g.nodes['g'].data['u_ref_relative'].

        Returns
        -------
        None
        """
        new_graphs = []
        for g in self.graphs:
            g.nodes['g'].data['u_ref_relative'] = g.nodes['g'].data['u_ref'].detach().clone()
            g.nodes['g'].data['u_ref_relative'] = (g.nodes['g'].data['u_ref_relative'] - g.nodes['g'].data['u_ref_relative'].mean(dim=-1, keepdims=True)).float()
            new_graphs.append(g)

        # Update in place
        self.graphs = new_graphs
        del new_graphs


    def reshape_conformation_size(self, n_confs=50):
        """Reshape conformation size.

        This is a work around to handle different graph size (shape). DGL requires at least one dimension with same size. 
        Here, we will modify the graphs so that each graph has the same number of conformations instead fo concatenating 
        graphs into heterogenous graphs with the same number of conformations. This allows shuffling and mini-batching per
        graph (molecule). 

        Only g.nodes['g'].data['u_ref'], g.nodes['g'].data['u_ref_relative'], and g.nodes['n1'].data['xyz'] will be updated.

        Parameters
        ----------
        n_confs : int, default=50
            Number of conformations per graph (molecule).

        Returns
        -------
        None
        """
        _logger.info(f'Reshape graph size')
        
        import random
        import copy
        import torch

        # Remove node features that are not used during training
        self._remove_node_features()

        new_graphs = []
        for i, g in enumerate(self.graphs):
            n = g.nodes['n1'].data['xyz'].shape[1]

            if n == n_confs:
                _logger.info(f"Mol #{i} ({n} conformations)")
                new_graphs.append(g)

            elif n < n_confs:
                random.seed(self.random_seed)
                index_random = random.choices(range(0, n), k=n_confs-n)
                _logger.info(f"Randomly select {len(index_random)} conformations from Mol #{i} ({n} conformations)")

                _g = copy.deepcopy(g)
                _g.nodes["g"].data["u_ref"] = torch.cat((_g.nodes['g'].data['u_ref'], _g.nodes['g'].data['u_ref'][:, index_random]), dim=-1)
                _g.nodes["n1"].data["xyz"] = torch.cat((_g.nodes['n1'].data['xyz'], _g.nodes['n1'].data['xyz'][:, index_random, :]), dim=1)
                _g.nodes['n1'].data['u_ref_prime'] = torch.cat((_g.nodes['n1'].data['u_ref_prime'], _g.nodes['n1'].data['u_ref_prime'][:, index_random, :]), dim=1)
                new_graphs.append(_g)

            else:
                _logger.info(f"Shuffling Mol #{i} ({n} conformations) and splitting into {n_confs}")
                random.seed(self.random_seed)
                idx_range = random.sample(range(n), k=n)
                for j in range(n // n_confs + 1):
                    _g = copy.deepcopy(g)

                    if (j+1)*n_confs > n:
                        index = range(j*n_confs, n)
                        random.seed(self.random_seed)
                        index_random = random.choices(range(0, n), k=(j+1)*n_confs-n)
                        _logger.debug(f"Iteration {j}: Randomly select {len(index_random)} conformers")

                        _g.nodes["g"].data["u_ref"] = torch.cat((_g.nodes['g'].data['u_ref'][:, index], _g.nodes['g'].data['u_ref'][:, index_random]), dim=-1)
                        _g.nodes["n1"].data["xyz"] = torch.cat((_g.nodes['n1'].data['xyz'][:, index, :], _g.nodes['n1'].data['xyz'][:, index_random, :]), dim=1)
                        _g.nodes["n1"].data["u_ref_prime"] = torch.cat((_g.nodes['n1'].data['u_ref_prime'][:, index, :], _g.nodes['n1'].data['u_ref_prime'][:, index_random, :]), dim=1)
                    else:            
                        idx1 = j*n_confs
                        idx2 = (j+1)*n_confs
                        index = idx_range[idx1:idx2]
                        _logger.debug(f"Iteration {j}: Extract indice from {idx1} to {idx2}")

                        _g.nodes["g"].data["u_ref"] = _g.nodes['g'].data['u_ref'][:, index]
                        _g.nodes["n1"].data["xyz"] = _g.nodes['n1'].data['xyz'][:, index, :]
                        _g.nodes["n1"].data["u_ref_prime"] = _g.nodes['n1'].data['u_ref_prime'][:, index, :]
                    
                    new_graphs.append(_g)

        # Update in place
        self.graphs = new_graphs
        del new_graphs


    def _remove_node_features(self):
        """Remove node features that are not necessarily during Espaloma training.

        Returns
        -------
        None
        """
        import copy
        
        new_graphs = []
        for g in self.graphs:
            _g = copy.deepcopy(g)
            for key in g.nodes['g'].data.keys():
                if key.startswith('u_') and key != 'u_ref':
                    _g.nodes['g'].data.pop(key)
            for key in g.nodes['n1'].data.keys():
                if key.startswith('u_') and key != 'u_ref_prime':
                    _g.nodes['n1'].data.pop(key)
            new_graphs.append(_g)
        
        # Update in place
        self.graphs = new_graphs
        del new_graphs


    @staticmethod
    def _merge_graphs(ds):
        """Merge multiple Graph instances into a single Graph.

        Parameters
        ----------
        ds : list of espaloma.graphs.graph.Graph
            The list of Graph instances to be merged. All Graphs in the list must be equivalent.

        Returns
        -------
        g : single espaloma.graphs.graph.Graph
            The merged Graph. This is a deep copy of the first Graph in the input list, 
            with its node features updated to include those of the other graphs.
        """
        import numpy as np
        import copy
        import torch

        # Check if graphs are equivalent
        for i in range(1, len(ds)):
            # Openff molecule
            assert ds[0].mol == ds[i].mol
            # Mapped isomeric smiles
            assert ds[0].mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True) == ds[i].mol.to_smiles(isomeric=True, explicit_hydrogens=True, mapped=True)
            # Other node features
            for key in ["sum_q"]:
                np.testing.assert_array_equal(ds[0].nodes['g'].data[key].flatten().numpy(), ds[i].nodes['g'].data[key].flatten().numpy())
            for key in ["q_ref", "idxs", "h0"]:
                np.testing.assert_array_equal(ds[0].nodes['n1'].data[key].flatten().numpy(), ds[i].nodes['n1'].data[key].flatten().numpy())

        # Merge graphs
        g = copy.deepcopy(ds[0])
        for key in g.nodes['g'].data.keys():
            if key not in ["sum_q"]:
                for i in range(1, len(ds)):
                    g.nodes['g'].data[key] = torch.cat((g.nodes['g'].data[key], ds[i].nodes['g'].data[key]), dim=-1)
        for key in g.nodes['n1'].data.keys():
            if key not in ["q_ref", "idxs", "h0"]:
                for i in range(1, len(ds)):
                    if key == "xyz":
                        n_confs = ds[i].nodes['n1'].data['xyz'].shape[1]
                    g.nodes['n1'].data[key] = torch.cat((g.nodes['n1'].data[key], ds[i].nodes['n1'].data[key]), dim=1)
        
        return g


    @property
    def available_forcefields(self):
        """Available force fields to compute baseline energies and forces.

        List of available force fields are hard coded but any force fields that are callable from 
        `openmmforcefields.generators.SystemGenerator` are supported.

        Returns
        -------
        ff_list : list
            List of available force fields.
        """
        ff_list = [
            'gaff-1.81', 'gaff-2.11', 
            'openff-1.2.0', 'openff-2.0.0', 'openff-2.1.0', 
            'amber14-all.xml', 'amber/protein.ff14SBonlysc.xml'
        ]
        return ff_list
