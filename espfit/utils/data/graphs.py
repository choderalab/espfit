import espaloma as esp
from espaloma.data.dataset import GraphDataset

class CustomGraphDataset(GraphDataset):
    """
    Custom GraphDataset with additional support to manipulate DGL graphs

    Methods
    -------
    filter()
        Filter high energy conformers and ensure minimum number of conformers

    compute_baseline_forcefields()
        Compute energies and forces using baseline force fields 

    compute_

    """

    def __init__(self):
        super(CustomGraphs, self).__init__()



    def filter():
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



    def compute_baseline_forcefields():
        """
        Compute energies and forces using baseline force fields
        """
        raise NotImplementedError


    def compute_am1bcc_elf10():
        """
        Compute partial charges
        """
        raise NotImplementedError


    def compute_relative_energy():
        """
        Compute relative energy

        Parameters
        ----------

        """
        raise NotImplementedError


    def subtract_nonbonded_interactions():
        """
        Subtract nonbonded energies and forces
        """

        raise NotImplementedError


    def reshape_conformation_size():
        """
        Reshape conformation size
        """
        raise NotImplementedError
