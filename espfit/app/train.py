"""
Create espaloma network model and train the model.

TODO
----
"""
import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class EspalomaModel(object):
    """Espaloma network model and training modules.

    Methods
    -------
    from_toml(filename):
        Load espaloma configuration file in TOML format.
    
    Examples
    --------
    >>> from espfit.app.train import EspalomaModel
    >>> filename = 'examples/espaloma_config.toml'
    >>> # create espaloma network model from toml file
    >>> model = EspalomaModel.from_toml(filename)
    >>> # check espaloma network model
    >>> model.net
    """

    def __init__(self, net=None, train_dataset=None, validation_dataset=None, test_dataset=None, random_seed=2666):
        """Initialize an instance of the class with an Espaloma network model and a random seed.

        This constructor method sets up the Espaloma network model, the training, validation, test datasets, 
        a configuratino file, and the random seed that will be used throughout the training process. 
        If no model or datasets are provided, the corresponding attributes will be set to None. If no random seed is 
        provided, the `random_seed` attribute will be set to 2666.

        Parameters
        ----------
        net : torch.nn.Sequential, default=None
            The Espaloma network model to be used for training.

        train_dataset : espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The training dataset. espaloma.graphs.graph.Graph. If not provided, the `train_data` attribute will be set to None.

        validation_dataset : espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The validation dataset. If not provided, the `validation_data` attribute will be set to None.

        test_dataset : Dataset, espfit.utils.data.graphs.CustomGraphDataset or espaloma.data.dataset.GraphDataset, default=None
            The test dataset. If not provided, the `test_data` attribute will be set to None.

        random_seed : int, default=2666
            The random seed used throughout the espaloma training.
        """
        self.net = net
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.random_seed = random_seed
        self.config = None   # TODO: Better way to handle this?


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
        model = cls.create_model(config['espaloma'])
        
        # TODO: Better way to handle this?
        model = cls(model)
        model.config = config
        
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


    def _restart_checkpoint(self, output_prefix):
        """Load the last checkpoint and restart the training process.

        This method finds all the checkpoint files in the directory specified by `output_prefix`, 
        loads the last checkpoint (e.g. net100.pt), and restarts the training process from the next step. If no 
        checkpoint files are found, the training process starts from the first step.

        Parameters
        ----------
        output_prefix : str
            The directory where the checkpoint files are stored.

        Returns
        -------
        int
            The step from which the training process should be restarted.
        """
        import os
        import glob
        import torch

        checkpoints = glob.glob("{}/*.th".format(output_prefix))
        
        if checkpoints:
            n = [ int(c.split('net')[1].split('.')[0]) for c in checkpoints ]
            n.sort()
            last_step = n[-1]
            last_checkpoint = os.path.join(output_prefix, f"net{last_step}.pt")
            self.net.load_state_dict(torch.load(last_checkpoint))
            step = last_step + 1
            logging.info(f'Restarting from ({last_checkpoint}).')
        else:
            step = 1
        
        return step
    

    def train(self, epochs=1000, batch_size=128, learning_rate=1e-4, checkpoint_frequency=10, output_prefix=None):
        """
        Train the Espaloma network model.

        This method trains the Espaloma network model using the training dataset. The training process can be customized 
        by specifying the number of epochs, batch size, learning rate, checkpoint frequency, and an output directory. 
        The method also supports restarting the training from a checkpoint.

        Parameters
        ----------
        epochs : int, default=1000
            The number of epochs to train the model for.

        batch_size : int, default=128
            The number of samples per batch.

        learning_rate : float, default=1e-4
            The learning rate for the optimizer.

        checkpoint_frequency : int, default=10
            The frequency at which the model should be saved.

        output_prefix : str, default=None
            The directory where the model checkpoints should be saved. If not provided, current working directory will be used.

        Returns
        -------
        None
        """
        import os
        import dgl
        import torch
        from pathlib import Path
        output_prefix = Path.cwd()

        # espaloma settings for training
        config = self.config['espaloma']['train']
        epochs = config.get('epochs', epochs)
        batch_size = config.get('batch_size', batch_size)
        learning_rate = config.get('learning_rate', learning_rate)
        checkpoint_frequency = config.get('checkpoint_frequency', checkpoint_frequency)
        output_prefix = config.get('output_prefix', output_prefix)

        # create output directory if not exists
        os.makedirs(output_prefix, exist_ok=True)

        # restart from checkpoint if exists
        step = self._restart_checkpoint(output_prefix)

        # train
        # https://github.com/choderalab/espaloma/blob/main/espaloma/app/train.py#L33
        # https://github.com/choderalab/espaloma/blob/main/espaloma/data/dataset.py#L310
        from espfit.utils.units import HARTEE_TO_KCALPERMOL
        ds_tr_loader = self.train_dataset.view(collate_fn='graph', batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        with torch.autograd.set_detect_anomaly(True):
            for i in range(step, step+epochs):
                for g in ds_tr_loader:
                    optimizer.zero_grad()
                    g = g.to("cuda:0")   # TODO: Better way to handle this?
                    g.nodes["n1"].data["xyz"].requires_grad = True 
                    loss = self.net(g)
                    loss.backward()
                    optimizer.step()
                if i % checkpoint_frequency == 0:
                    # Note: returned loss is a joint loss of different units.
                    _loss = HARTEE_TO_KCALPERMOL * loss.pow(0.5).item()
                    logging.info(f'Epoch {i}: {_loss:.3f}')
                    checkpoint_file = os.path.join(output_prefix, f"net{i}.pt")
                    torch.save(self.net.state_dict(), checkpoint_file)


    def train_val():
        raise NotImplementedError

    
    def train_data():
        raise NotImplementedError


    def validation_data():
        raise NotImplementedError


    def test_data():
        raise NotImplementedError


    def save():
        raise NotImplementedError


    def plot_loss():
        raise NotImplementedError

