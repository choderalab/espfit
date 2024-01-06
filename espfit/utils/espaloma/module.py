import torch
import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class ExpCoeff(torch.nn.Module):
    """A PyTorch module that applies the exponential function to the logarithmic coefficients of a graph's nodes.

    Reference
    ---------
    See section `D.1 Training and inference` in 
    https://www.rsc.org/suppdata/d2/sc/d2sc02739a/d2sc02739a1.pdf
    """
    def forward(self, g):
        """Apply the exponential function to the logarithmic coefficients of the 'n2' and 'n3' nodes in the graph.

        Parameters
        ----------
        g : dgl.DGLGraph
            The input graph. The graph should have 'n2' and 'n3' nodes with a 'log_coefficients' attribute.

        Returns
        -------
        g : dgl.DGLGraph
            The modified graph, with the 'coefficients' attribute of the 'n2' and 'n3' nodes updated to be the exponential 
            of the 'log_coefficients' attribute.
        """
        g.nodes['n2'].data['coefficients'] = g.nodes['n2'].data['log_coefficients'].exp()
        g.nodes['n3'].data['coefficients'] = g.nodes['n3'].data['log_coefficients'].exp()
        return g


class GetLoss(torch.nn.Module):
    """Compute joint loss.

    Methods
    -------
    compute_energy_loss(g):
        Compute relative energy loss with mean offset

    compute_force_loss(g):
        Compute force loss

    compute_charge_loss(g):
        Compute charge loss

    compute_torsion_loss(g):
        Compute torsion l2 regularization

    compute_improper_loss(g):
        Compute improper l2 regularization

    """
    def __init__(self, weights={'energy': 1.0, 'force': 1.0, 'charge': 1.0, 'torsion': 1.0, 'improper': 1.0}):
        """Define loss function.

        Parameters
        ----------

        weights : dict
            loss weights in dictionary
        """
        super(GetLoss, self).__init__()
        self.weights = weights


    def compute_energy_loss(self, g):
        """Relative energy loss with mean offset"""
        return torch.nn.MSELoss()(
            g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True),
            g.nodes['g'].data['u_ref_relative'],
        )


    def compute_force_loss(self, g):
        """Force loss"""
        du_dx_hat = torch.autograd.grad(
            g.nodes['g'].data['u'].sum(),
            g.nodes['n1'].data['xyz'],
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        du_dx = g.nodes["n1"].data["u_ref_prime"]

        return torch.nn.MSELoss()(
            du_dx, 
            du_dx_hat
        )
    

    def compute_charge_loss(self, g):
        """Charge loss"""
        return torch.nn.MSELoss()(
            g.nodes['n1'].data['q'],
            g.nodes['n1'].data['q_ref'],
        )


    def compute_torsion_loss(self, g):
        """Torsion loss computed as L2 regularization"""
        return g.nodes['n4'].data['k'].pow(2).mean()


    def compute_improper_loss(self, g):
        """Improper loss computed as L2 regularization"""
        return g.nodes['n4_improper'].data['k'].pow(2).mean()


    def forward(self, g):
        loss_energy = self.compute_energy_loss(g) * self.weights['energy']
        loss_force = self.compute_force_loss(g) * self.weights['force']

        if self.charge_weight > 0:
            loss_charge = self.compute_charge_loss(g) * self.weights['charge']
        if self.torsion_weight > 0 and g.number_of_nodes('n4') > 0:
            loss_torsion = self.compute_torsion_loss(g) * self.weights['torsion']
        if self.improper_weight > 0 and g.number_of_nodes('n4_improper') > 0:
            loss_improper = self.compute_improper_loss(g) * self.weights['improper']

        logging.info(f"# energy: {loss_energy}, force: {loss_force}, charge: {loss_charge}, torsion: {loss_torsion}, improper: {loss_improper}")
        loss = loss_energy + loss_force + loss_charge + loss_torsion + loss_improper
        
        return loss