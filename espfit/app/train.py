"""
Create espaloma network model and train the model
"""
import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class EspalomaModel(object):
    """Espaloma network model and training modules.

    Methods
    -------
    
    from_toml(filename):
        Load espaloma configuration file in TOML format
    
    
    Notes
    -----


    Examples
    --------

    >>> from espfit.app.train import EspalomaModel
    >>> filename = 'examples/espaloma_config.toml'
    >>> model = EspalomaModel.from_toml(filename)

    """

    def __init__(self, net=None, RANDOM_SEED=2666):
        """Create espaloma network model.

        Parameters
        ----------
        net : torch.nn.Sequential, default=None
            espaloma network architecture

        random_seed : int, default=2666
            random seed used throughout the espaloma training
        """
        self.net = net
        self.random_seed = RANDOM_SEED


    @classmethod
    def from_toml(cls, filename):
        """Load espaloma configure file

        Parameters
        ----------
        filename : str
            Espaloma configuration file in TOML format
        """
        import tomllib
        try:
            with open(filename, 'rb') as f:
                config = tomllib.load(f)
        except FileNotFoundError as e:
            print(e)
            raise
        model = cls.create_model(config['espaloma'])
        return cls(model)


    @staticmethod
    def create_model(espaloma_config):
        """Define espaloma architecture.
        
        Parameters
        ----------

        espaloma_config : dict
            espaloma network configuration in dictionary format
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
        if 'loss_weights' in espaloma_config.keys():
            for key in espaloma_config['loss_weights'].keys():
                weights[key] = espaloma_config['loss_weights'][key]

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


    def train():
        raise NotImplementedError


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

