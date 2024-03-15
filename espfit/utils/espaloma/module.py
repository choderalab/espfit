import torch
import logging

_logger = logging.getLogger(__name__)


class ExpCoeff(torch.nn.Module):
    """A PyTorch module that applies the exponential function to the logarithmic coefficients of a graph's nodes.

    Methods
    -------
    forward(g):
        Apply the exponential function to the logarithmic coefficients of the 'n2' and 'n3' nodes in the graph.

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

    forward(g):
        Compute joint loss
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
        """Relative energy loss with mean offset.
        
        Parameters
        ----------
        g : dgl.DGLGraph

        Returns
        -------
        torch.Tensor
        """
        return torch.nn.MSELoss()(
            g.nodes['g'].data['u'] - g.nodes['g'].data['u'].mean(dim=-1, keepdims=True),
            g.nodes['g'].data['u_ref_relative'],
        )


    def compute_force_loss(self, g):
        """Force loss.
        
        Parameters
        ----------
        g : dgl.DGLGraph

        Returns
        -------
        torch.Tensor
        """
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
        """Charge loss.
        
        Parameters
        ----------
        g : dgl.DGLGraph

        Returns
        -------
        torch.Tensor
        """
        return torch.nn.MSELoss()(
            g.nodes['n1'].data['q'],
            g.nodes['n1'].data['q_ref'],
        )


    def compute_torsion_loss(self, g):
        """Torsion loss computed as L2 regularization.
        
        Parameters
        ----------
        g : dgl.DGLGraph

        Returns
        -------
        torch.Tensor
        """
        return g.nodes['n4'].data['k'].pow(2).mean()


    def compute_improper_loss(self, g):
        """Improper loss computed as L2 regularization.
        
        Parameters
        ----------
        g : dgl.DGLGraph

        Returns
        -------
        torch.Tensor
        """
        return g.nodes['n4_improper'].data['k'].pow(2).mean()


    def forward(self, g):
        """Compute joint loss.

        Parameters
        ----------
        g : dgl.DGLGraph

        Returns
        -------
        loss : torch.Tensor
            Total weighted loss

        loss_dict : dict
            Dictionary of individual weighted losses
        """
        loss_energy = self.compute_energy_loss(g) * self.weights['energy']
        loss_force = self.compute_force_loss(g) * self.weights['force']

        if self.weights['charge'] > 0:
            loss_charge = self.compute_charge_loss(g) * self.weights['charge']
        if self.weights['torsion'] > 0 and g.number_of_nodes('n4') > 0:
            loss_torsion = self.compute_torsion_loss(g) * self.weights['torsion']
        else:
            # if no torsion, set to zero
            loss_torsion = torch.tensor(0.0)
        if self.weights['improper'] > 0 and g.number_of_nodes('n4_improper') > 0:
            loss_improper = self.compute_improper_loss(g) * self.weights['improper']
        else:
            # if no improper, set to zero
            loss_improper = torch.tensor(0.0)

        _logger.debug(f"energy: {loss_energy:.5f}, force: {loss_force:.5f}, charge: {loss_charge:.5f}, torsion: {loss_torsion:.5f}, improper: {loss_improper:.5f}")
        loss = loss_energy + loss_force + loss_charge + loss_torsion + loss_improper

        loss_dict = {
            'loss': None,
            'energy': loss_energy.item(),
            'force': loss_force.item(),
            'charge': loss_charge.item(),
            'torsion': loss_torsion.item(),
            'improper': loss_improper.item(),
        }
        
        return loss, loss_dict