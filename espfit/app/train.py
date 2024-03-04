"""
Create espaloma network model and train the model.

TODO
----
* Export loss to a file (e.g. LossReporter class?)
* Add support to use multiple GPUs
* Improve how data are parsed using dataclasses or pydantic
"""
import os
import torch
import logging

_logger = logging.getLogger(__name__)


class EspalomaBase(object):
    def __init__(self):
        # Check if GPU is available
        if torch.cuda.is_available():
            _logger.info('GPU is available for training.')
        else:
            _logger.info('GPU is not available for training.')

        # Check torch data type
        _logger.debug(f'Torch data type is {torch.get_default_dtype()}')


    @classmethod
    def from_toml(cls, filename, **override_espalomamodel_kwargs):
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
        model.configfile = filename

        # Update training settings
        for key, value in config['espaloma']['train'].items():
            if hasattr(model, key):
                setattr(model, key, value)
            else:
                raise ValueError(f'Invalid attribute {key}.')

        # Override training settings
        for key, value in override_espalomamodel_kwargs.items():
            if hasattr(model, key):
                setattr(model, key, value)
            else:
                raise ValueError(f'Invalid attribute {key}.')

        return model


    @staticmethod
    def _get_base_module(espaloma_config):
        """Create base modules for Espaloma network model.
        
        Parameters
        ----------
        espaloma_config : dict
            A dictionary containing the configuration for the Espaloma network.
            This includes the method and options for the GNN layers, the configurations for the two stages of the network,
            and optionally the weights for different loss components.

        Returns
        -------
        list
            A list of modules for the Espaloma network model.
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

        # Initialize loss weights and update if provided
        weights = { 'energy': 1.0, 'force': 1.0, 'charge': 1.0, 'torsion': 1.0, 'improper': 1.0 }
        if 'weights' in espaloma_config.keys():
            for key in espaloma_config['weights'].keys():
                weights[key] = espaloma_config['weights'][key]

        # Append base modules
        modules = []
        modules.append(representation)
        modules.append(readout)
        modules.append(readout_improper)
        modules.append(esp.nn.readout.janossy.ExpCoefficients())
        modules.append(esp.nn.readout.charge_equilibrium.ChargeEquilibrium())

        return modules, weights


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
        from espfit.utils.espaloma.module import GetLoss

        # Get base model
        modules, weights = EspalomaBase._get_base_module(espaloma_config)

        # Define espaloma architecture
        modules.append(esp.mm.geometry.GeometryInGraph())
        modules.append(esp.mm.energy.EnergyInGraph(terms=["n2", "n3", "n4", "n4_improper"]))
        modules.append(GetLoss(weights))

        # Create model
        net = torch.nn.Sequential(*modules)
        if torch.cuda.is_available():
            return net.cuda()
        else:
            return net
        

    def save_model(self, net=None, best_model=None, model_name='espaloma.pt', output_directory_path=None):
        """Save the Espaloma network model to a file.
        
        This method saves the Espaloma network model to a file in the specified output directory.
        
        Parameters
        ----------
        net : torch.nn.Sequential
            The Espaloma network model to be saved.

        best_model : str
            The path to the best model file.

        model_name : str, default='espaloma.pt'
            The name of the file to save the model to.

        output_directory_path : str, default=None
            The directory where the model should be saved. 
            If not provided, the model will be saved in the current working directory.

        Returns
        -------
        None
        """
        import espaloma as esp

        if output_directory_path is not None:
            os.makedirs(output_directory_path, exist_ok=True)
        else:
            output_directory_path = os.getcwd()

        if net:
            modules = []
            for module in net:
                if isinstance(module, esp.mm.geometry.GeometryInGraph):
                    break
                modules.append(module)
            modules.append(esp.nn.readout.janossy.LinearMixtureToOriginal())
            net = torch.nn.Sequential(*modules)
        else:
            raise ValueError('No model provided.')
        
        # Save model
        state_dict = torch.load(best_model, map_location=torch.device('cpu'))
        net.load_state_dict(state_dict)
        torch.save(net, os.path.join(output_directory_path, model_name))


class EspalomaModel(EspalomaBase):
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

    def __init__(self, net=None, dataset_train=None, dataset_validation=None, dataset_test=None, 
                 epochs=1000, batch_size=128, learning_rate=1e-4, checkpoint_frequency=10, 
                 random_seed=2666, output_directory_path=None):
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

        epochs : int, default=1000
            The number of epochs to train the model for.

        batch_size : int, default=128
            The number of samples per batch.

        learning_rate : float, default=1e-4
            The learning rate for the optimizer.

        checkpoint_frequency : int, default=10
            The frequency at which the model should be saved.

        random_seed : int, default=2666
            The random seed used throughout the espaloma training.

        output_directory_path : str, default=None
            The directory where the model checkpoints should be saved. 
            If not provided, the checkpoints will be saved in the current working directory.
        """
        super(EspalomaBase, self).__init__()
        self.net = net
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.dataset_test = dataset_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_frequency = checkpoint_frequency
        self.restart_epoch = 0
        self.random_seed = random_seed
        if output_directory_path is None:
            output_directory_path = os.getcwd()
            self.output_directory_path = output_directory_path


    @property
    def output_directory_path(self):
        """Get output directory path."""
        return self._output_directory_path


    @output_directory_path.setter
    def output_directory_path(self, value):
        """Set output directory path."""
        self._output_directory_path = value
        # Create output directory if it does not exist
        os.makedirs(value, exist_ok=True)


    def report_loss(self, loss_trajecotry):
        """Report loss.

        Parameters
        ----------
        loss : dict
            The loss trajectory that stores individual weighted losses for each epoch.

        Returns
        -------
        None
        """
        import pandas as pd
        df = pd.DataFrame.from_dict(loss_trajecotry, orient='index')
        df.to_csv(os.path.join(self.output_directory_path, 'reporter.log'), sep='\t', float_format='%.4f')


    def train(self):
        """
        Train the Espaloma network model.

        TODO
        ----
        * Export training settings to a file?

        Returns
        -------
        None
        """
        from espfit.utils.units import HARTREE_TO_KCALPERMOL

        if self.dataset_train is None:
            raise ValueError('Training dataset is not provided.')
        
        # Load checkpoint
        self.restart_epoch = self._load_checkpoint()

        # Train
        # https://github.com/choderalab/espaloma/blob/main/espaloma/app/train.py#L33
        # https://github.com/choderalab/espaloma/blob/main/espaloma/data/dataset.py#L310            
        ds_tr_loader = self.dataset_train.view(collate_fn='graph', batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.restart_epoch, self.epochs):
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
                    self._save_checkpoint(epoch)
    
    
    def train_sampler(self, sampler_patience=800, neff_threshold=0.2, sampler_weight=1.0, debug=False):
        """
        Train the Espaloma network model with sampler.

        TODO
        ----
        * Export loss to a file (e.g. LossReporter class?)
        * Should `nsteps` be a variable when calling train_sampler?
        * Should `sampler_patience` and `neff_threshold` be an instance variable of sampler.BaseSimulation?

        """
        from espfit.utils.units import HARTREE_TO_KCALPERMOL
        from espfit.utils.sampler.reweight import SetupSamplerReweight

        # Note: RuntimeError will be raised if copy.deepcopy is used.
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace 
        # operation: [torch.cuda.FloatTensor [512, 1]], which is output 0 of AsStridedBackward0, is at version 2; 
        # expected version 1 instead. Hint: the backtrace further above shows the operation that failed to 
        # compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
        import copy
        net_copy = copy.deepcopy(self.net)

        self.sampler_patience = sampler_patience
        self.neff_threshold = neff_threshold

        if self.dataset_train is None:
            raise ValueError('Training dataset is not provided.')

        # Load checkpoint
        self.restart_epoch = self._load_checkpoint()

        # Train
        neff = -1
        loss_trajectory = {}
        ds_tr_loader = self.dataset_train.view(collate_fn='graph', batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.restart_epoch, self.epochs):
                epoch = i + 1    # Start from 1 (not zero-indexing)
                
                loss = torch.tensor(0.0)
                if torch.cuda.is_available():
                    loss = loss.cuda("cuda:0")

                for g in ds_tr_loader:
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        g = g.to("cuda:0")
                    g.nodes["n1"].data["xyz"].requires_grad = True
                    
                    # Forward pass
                    # Note that returned values are weighted losses.
                    _loss, loss_dict = self.net(g)
                    # Append loss
                    loss += _loss

                # Include sampler loss after certain epochs
                if epoch > self.sampler_patience:
                    # Run sampling for the first time
                    if neff == -1:
                        _logger.info(f'Reached sampler patience epoch={self.sampler_patience}. Run sampler for the first time.')
                        # Initialize
                        SamplerReweight = SetupSamplerReweight()
                        # Create new sampler system using local espaloma model
                        samplers = self._setup_local_samplers(epoch, net_copy, debug)
                        SamplerReweight.update(samplers)
                        SamplerReweight.run()
                    else:
                        # If effective sample size is below threshold, re-run sampler
                        neff = SamplerReweight.get_effective_sample_size()
                        if neff < self.neff_threshold:
                            _logger.info(f'Effective sample size ({neff}) below threshold ({self.neff_threshold}).')
                            samplers = self._setup_local_samplers(epoch, net_copy, debug)
                            SamplerReweight.update(samplers)
                            SamplerReweight.run()

                    # Compute sampler loss
                    _logger.info(f'Compute sampler loss.')
                    loss_list = SamplerReweight.compute_loss()   # list of torch.tensor
                    for sampler_index, _loss in enumerate(loss_list):
                        #loss_dict[f'sampler{sampler_index}'] = _loss.item()
                        _sampler = SamplerReweight.samplers[sampler_index]
                        loss_dict[f'{_sampler.target_name}'] = _loss.item()
                        loss += _loss * sampler_weight

                # Append total and individual loss to loss_trajectory
                loss_dict['loss'] = loss.item()
                loss_trajectory[epoch] = loss_dict

                # Update weights
                loss.backward()
                optimizer.step()
                
                if epoch % self.checkpoint_frequency == 0:
                    # Note: returned loss is a joint loss of different units.
                    _loss = HARTREE_TO_KCALPERMOL * loss.pow(0.5).item()
                    _logger.info(f'epoch {epoch}: {_loss:.3f}')
                    self._save_checkpoint(epoch)

        # Export loss trajectory
        # TODO: Report losses at every epoch
        _logger.info(f'Export loss trajectory to a file.')
        self.report_loss(loss_trajectory)


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
        import sys
        import glob

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


    def _save_checkpoint(self, epoch):
        """Save local model.

        Parameters
        ----------
        epoch : int
            The epoch number.

        Returns
        -------
        None
        """
        checkpoint_file = os.path.join(self.output_directory_path, f"checkpoint{epoch}.pt")
        torch.save(self.net.state_dict(), checkpoint_file)


    def _setup_local_samplers(self, epoch, net_copy, debug):
        from espfit.app.sampler import SetupSampler

        # Save espaloma checkpoint models
        self._save_checkpoint(epoch)
        # Save checkpoint as temporary espaloma model (force field)
        local_model = os.path.join(self.output_directory_path, f"checkpoint{epoch}.pt")
        self.save_model(net=net_copy, best_model=local_model, model_name=f"net{epoch}.pt", output_directory_path=self.output_directory_path)
        
        # Define sampler settings with override arguments
        args = [epoch]
        if debug == True:
            from importlib.resources import files
            small_molecule_forcefield = str(files('espfit').joinpath("data/forcefield/espaloma-0.3.2.pt"))
        else:
            small_molecule_forcefield = os.path.join(self.output_directory_path, f"net{epoch}.pt")

        override_sampler_kwargs = { 
            "atomSubset": 'all',
            "small_molecule_forcefield": small_molecule_forcefield,
            "output_directory_path": self.output_directory_path 
            }
        
        # Create sampler system from configuration file. Returns list of systems.                        
        samplers = SetupSampler.from_toml(self.configfile, *args, **override_sampler_kwargs)

        return samplers
    