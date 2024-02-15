"""
Create espaloma network model and train the model.

TODO
----
* Add support to use multiple GPUs
* Add support to validate model? (or use independent script?)
* Add support to save model? (or use independent script?)
* Improve how data are parsed using dataclasses or pydantic
"""
import logging

_logger = logging.getLogger(__name__)


class EspalomaModel(object):
    """Espaloma network model and training modules.

    Methods
    -------
    from_toml(filename):
        Load espaloma configuration file in TOML format.
    
    Examples
    --------
    >>> from espfit.app.train import EspalomaModel
    >>> filename = 'espfit/data/config/config.toml'
    >>> # create espaloma network model from toml file
    >>> model = EspalomaModel.from_toml(filename)
    >>> # check espaloma network model
    >>> model.net
    >>> # load training dataset
    >>> model.dataset_train = ds
    >>> model.train()
    """

    def __init__(self, net=None, dataset_train=None, dataset_validation=None, dataset_test=None, random_seed=2666, output_directory_path=None, 
                 epochs=1000, batch_size=128, learning_rate=1e-4, checkpoint_frequency=10):
        """Initialize an instance of the class with an Espaloma network model and a random seed.

        This constructor method sets up the Espaloma network model, the training, validation, test datasets, 
        a configuratino file, and the random seed that will be used throughout the training process. 
        If no model or datasets are provided, the corresponding attributes will be set to None. If no random seed is 
        provided, the `random_seed` attribute will be set to 2666.

        Parameters
        ----------
        net : torch.nn.Sequential, default=None
            The Espaloma network model to be used for training.        
        
        dataset_train : espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The training dataset. espaloma.graphs.graph.Graph. If not provided, the `train_data` attribute will be set to None.

        dataset_validation : espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The validation dataset. If not provided, the `validation_data` attribute will be set to None.

        dataset_test : Dataset, espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The test dataset. If not provided, the `test_data` attribute will be set to None.

        random_seed : int, default=2666
            The random seed used throughout the espaloma training.

        output_directory_path : str, default=None
            The directory where the model checkpoints should be saved. 
            If not provided, the checkpoints will be saved in the current working directory.

        epochs : int, default=1000
            The number of epochs to train the model for.

        batch_size : int, default=128
            The number of samples per batch.

        learning_rate : float, default=1e-4
            The learning rate for the optimizer.

        checkpoint_frequency : int, default=10
            The frequency at which the model should be saved.
        """
        import os
        import torch
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.dataset_test = dataset_test
        self.net = net
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_frequency = checkpoint_frequency
        if output_directory_path is None:
            self.output_directory_path = os.getcwd()
        else:
            self.output_directory_path = output_directory_path
        
        # Check if GPU is available
        if torch.cuda.is_available():
            _logger.info('GPU is available for training.')
        else:
            _logger.info('GPU is not available for training.')

        # Check torch data type
        _logger.info(f'Torch data type is {torch.get_default_dtype()}')


    @classmethod
    def from_toml(cls, filename):
        """Create an instance of the class from a TOML configuration file.

        This method reads a TOML file specified by `filename`, extracts the 'espaloma'
        section of the configuration, and uses it to create a model. It then returns 
        an instance of the class initialized with this model. If the file is not found,
        the method prints the error and re-raises the exception.

        Parameters
        ----------
        filename : str
            Path to the TOML file containing the configuration for the espaloma model.

        Returns
        -------
        object
            An instance of the class initialized with the model created from the 
            espaloma configuration in the TOML file.
        """
        import tomllib
        try:
            with open(filename, 'rb') as f:
                config = tomllib.load(f)
        except FileNotFoundError as e:
            print(e)
            raise

        model = cls()
        net = model.create_model(config['espaloma'])
        model.net = net

        return model


    @staticmethod
    def create_model(espaloma_config):
        """Create an Espaloma network model using the provided configuration.

        This function constructs a PyTorch Sequential model with two stages of Graph Neural Network (GNN) layers,
        JanossyPooling readout layers for various features, and additional layers for energy computation and loss calculation.
        The specifics of the GNN layers and the readout layers are controlled by the `espaloma_config` dictionary.
        If a CUDA-compatible GPU is available, the model is moved to the GPU before being returned.

        Parameters
        ----------
        espaloma_config : dict
            A dictionary containing the configuration for the Espaloma network.
            This includes the method and options for the GNN layers, the configurations for the two stages of the network,
            and optionally the weights for different loss components.

        Returns
        -------
        torch.nn.Sequential
            The constructed Espaloma network model.
        """
        import espaloma as esp
        
        # GNN
        gnn_method = 'SAGEConv'
        gnn_options = {}
        for key in espaloma_config['gnn'].keys():
            if key == 'method':
                gnn_method = espaloma_config['gnn'][key]
            else:
                gnn_options[key] = espaloma_config['gnn'][key]
        layer = esp.nn.layers.dgl_legacy.gn(gnn_method, gnn_options)
        
        # Stage1
        config_1 = espaloma_config['nn']['stage1']
        representation = esp.nn.Sequential(layer, config=config_1)
        
        # Stage2
        config_2 = espaloma_config['nn']['stage2']
        units = config_2[0] 
        # out_features: Define modular MM parameters Espaloma will assign
        # 1: atom hardness and electronegativity
        # 2: bond linear combination, enforce positive
        # 3: angle linear combination, enforce positive
        # 4: torsion barrier heights (can be positive or negative)
        readout = esp.nn.readout.janossy.JanossyPooling(
            in_features=units, config=config_2,
            out_features={
                    1: {'s': 1, 'e': 1},
                    2: {'log_coefficients': 2},
                    3: {'log_coefficients': 2},
                    4: {'k': 6},
            },
        )
        # Improper torsions (multiplicity n=2)
        readout_improper = esp.nn.readout.janossy.JanossyPoolingWithSmirnoffImproper(in_features=units, config=config_2, out_features={"k": 2})

        # Get loss weights
        # TODO: Better way to handle this?
        weights = { 'energy': 1.0, 'force': 1.0, 'charge': 1.0, 'torsion': 1.0, 'improper': 1.0 }
        if 'weights' in espaloma_config.keys():
            for key in espaloma_config['weights'].keys():
                weights[key] = espaloma_config['weights'][key]

        # Define espaloma architecture
        import torch
        from espfit.utils.espaloma.module import GetLoss
        net = torch.nn.Sequential(
            representation,
            readout,
            readout_improper,
            esp.nn.readout.janossy.ExpCoefficients(),
            esp.nn.readout.charge_equilibrium.ChargeEquilibrium(),
            esp.mm.geometry.GeometryInGraph(),
            esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]),
            GetLoss(weights),
        )
        if torch.cuda.is_available():
            return net.cuda()
        else:
            return net


    def _load_checkpoint(self):
        """Load the last checkpoint and restart the training process.

        This method finds all the checkpoint files in the output directory, loads the 
        last checkpoint (e.g. net100.pt), and restarts the training process from the next step. 
        If no checkpoint files are found, the training process starts from the first step.

        Returns
        -------
        int
            The step from which the training process should be restarted.
        """
        import os
        import sys
        import glob
        import torch

        checkpoints = glob.glob("{}/*.pt".format(self.output_directory_path))
        
        if checkpoints:
            n = [ int(c.split('net')[1].split('.')[0]) for c in checkpoints ]
            n.sort()
            restart_epoch = n[-1]
            restart_checkpoint = os.path.join(self.output_directory_path, f"net{restart_epoch}.pt")
            self.net.load_state_dict(torch.load(restart_checkpoint))
            logging.info(f'Restarting from ({restart_checkpoint}).')
        else:
            restart_epoch = 0
        
        if restart_epoch >= self.epochs:
            _logger.info(f'Already trained for {self.epochs} epochs.')
            sys.exit(0)
        elif restart_epoch > 0:
            _logger.info(f'Training for additional {self.epochs-restart_epoch} epochs.')
        else:
            _logger.info(f'Training from scratch for {self.epochs} epochs.')

        return restart_epoch


    def train(self, output_directory_path=None):
        """
        Train the Espaloma network model.

        Parameters
        ----------
        output_directory_path : str, default=None
            The directory where the model checkpoints should be saved. If None, the default output directory is used.

        Returns
        -------
        None
        """
        import os
        import torch
        from espfit.utils.units import HARTREE_TO_KCALPERMOL

        if self.dataset_train is None:
            raise ValueError('Training dataset is not provided.')

        if output_directory_path is not None:
            self.output_directory_path = output_directory_path
            os.makedirs(self.output_directory_path, exist_ok=True)

        # Load checkpoint
        restart_epoch = self._load_checkpoint()

        # Train
        # https://github.com/choderalab/espaloma/blob/main/espaloma/app/train.py#L33
        # https://github.com/choderalab/espaloma/blob/main/espaloma/data/dataset.py#L310            
        ds_tr_loader = self.dataset_train.view(collate_fn='graph', batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(restart_epoch, self.epochs):
                epoch = i + 1    # Start from epoch 1 (not zero-indexing)
                for g in ds_tr_loader:
                    optimizer.zero_grad()
                    
                    # TODO: Better way to handle this?
                    if torch.cuda.is_available():
                        g = g.to("cuda:0")
                    
                    g.nodes["n1"].data["xyz"].requires_grad = True 
                    loss = self.net(g)
                    loss.backward()
                    optimizer.step()
                
                if epoch % self.checkpoint_frequency == 0:
                    # Note: returned loss is a joint loss of different units.
                    _loss = HARTREE_TO_KCALPERMOL * loss.pow(0.5).item()
                    _logger.info(f'epoch {epoch}: {_loss:.3f}')
                    checkpoint_file = os.path.join(output_directory_path, f"net{epoch}.pt")
                    torch.save(self.net.state_dict(), checkpoint_file)
    
    
    def train_sampler(self, output_directory_path=None, 
                      biopolymer_file=None, ligand_file=None, small_molecule_forcefield=None,
                      sampler_patience=800, maxIterations=10, nsteps=10, neff_threshold=0.2):
        import os
        import torch
        from espfit.utils.units import HARTREE_TO_KCALPERMOL
        from espfit.utils.sampler import module

        # Parameters for sampling and reweighting
        self.biopolymer_file = biopolymer_file
        self.ligand_file = ligand_file
        self.sampler_patience = sampler_patience
        self.maxIterations = maxIterations
        self.nsteps = nsteps
        self.neff_threshold = neff_threshold
        self.small_molecule_forcefield = small_molecule_forcefield

        if self.dataset_train is None:
            raise ValueError('Training dataset is not provided.')

        if output_directory_path is not None:
            self.output_directory_path = output_directory_path
            os.makedirs(self.output_directory_path, exist_ok=True)

        # Load checkpoint
        restart_epoch = self._load_checkpoint()

        # Initialize neff to -1 to trigger the first sampling
        neff = -1

        # Train
        ds_tr_loader = self.dataset_train.view(collate_fn='graph', batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(restart_epoch, self.epochs):
                epoch = i + 1    # Start from epoch 1 (not zero-indexing)
                loss = torch.tensor(0.0)
                for g in ds_tr_loader:
                    optimizer.zero_grad()

                    if torch.cuda.is_available():
                        g = g.to("cuda:0")

                    g.nodes["n1"].data["xyz"].requires_grad = True 
                    loss += self.net(g)

                # Run sampling
                if epoch > self.sampler_patience:
                    if neff < self.neff_threshold:
                        _logger.info(f'Effective sample size ({neff}) below threshold ({self.neff_threshold}).')
                        # Create system and run sampling, instead of restarting from previous checkpoint
                        _logger.info(f'Run simulation...')
                        sampler_output_directory_path = os.path.join(self.output_directory_path, "sampler", str(epoch))
                        module.run_sampler(sampler_output_directory_path, self.biopolymer_file, self.ligand_file, self.maxIterations, self.nsteps, self.small_molecule_forcefield)

                    # Compute MD loss
                    _logger.info(f'Compute sampler loss.')
                    sampler_loss = module.compute_loss(input_directory_path=sampler_output_directory_path)

                    # Add MD loss to the joint loss
                    loss += sampler_loss

                # Update weights
                loss.backward()
                optimizer.step()
                
                if epoch % self.checkpoint_frequency == 0:
                    # Note: returned loss is a joint loss of different units.
                    _loss = HARTREE_TO_KCALPERMOL * loss.pow(0.5).item()
                    _logger.info(f'epoch {epoch}: {_loss:.3f}')
                    checkpoint_file = os.path.join(self.output_directory_path, f"net{epoch}.pt")
                    torch.save(self.net.state_dict(), checkpoint_file)


    def validate():
        raise NotImplementedError


    def save_model():
        raise NotImplementedError