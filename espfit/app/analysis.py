"""
Compute experimental observables from MD simulations.

Notes
-----

TODO
----
"""
import os
import numpy as np
import logging

_logger = logging.getLogger(__name__)


class BaseDataLoader(object):
    """Base class for data loader.
    
    TODO
    ----
    * Add more methods to check trajectory information (e.g. number of frames, number of atoms, etc.)

    Methods
    -------
    load_traj(reference_pdb='solvated.pdb', trajectory_netcdf='traj.nc', atom_indices=None, stride=1):
        Load MD trajectory.
    """
    def __init__(self, atomSubset='solute', input_directory_path=None, output_directory_path=None):
        """Initialize base data loader object.
        
        Parameters
        ----------
        atomSubset : str, default='solute'
            Subset of atoms to save. Default is 'solute'. Other options 'all' and 'not water'.

        input_directory_path : str, optional
            Input directory path. Default is None.
            If None, the current working directory will be used.

        output_directory_path : str, optional
            Output directory path. Default is None.
            If None, the current working directory will be used.
        """
        self.atomSubset = atomSubset
        if self.atomSubset not in ['solute', 'all', 'not water']:
            raise ValueError(f"Invalid atomSubset: {self.atomSubset}. Expected 'solute', 'all', or 'not water'.")

        if input_directory_path is None:
            input_directory_path = os.getcwd()
        if output_directory_path is None:
            output_directory_path = os.getcwd()
        self.input_directory_path = input_directory_path
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


    # Should this be a classmethod?
    def load_traj(self, reference_pdb='solvated.pdb', trajectory_netcdf='traj.nc', stride=1, input_directory_path=None):
        """Load MD trajectory.
        
        Parameters
        ----------
        reference_pdb : str, optional
            Reference pdb file name. Default is 'solvated.pdb'.

        trajectory_netcdf : str, optional
            Trajectory netcdf file name. Default is 'traj.nc'.

        stride : int, optional
            Stride to load the trajectory. Default is 1.

        input_directory_path : str, optional
            Input directory path. Default is None.
            If None, the current working directory will be used.
        """
        import mdtraj

        if input_directory_path is not None:
            self.input_directory_path = input_directory_path

        # Load reference pdb (solvated system)
        pdb = os.path.join(self.input_directory_path, reference_pdb)
        ref_traj = mdtraj.load(pdb)
        
        # Select atoms to load from trajectory
        if self.atomSubset == 'all':
            self.atom_indices = None
            self.ref_traj = ref_traj
        else:
            self.atom_indices = []
            mdtop = ref_traj.topology
            if self.atomSubset == 'solute':
                res = [ r for r in mdtop.residues if r.name not in ('HOH', 'NA', 'CL', 'K') ]
            elif self.atomSubset == 'not water':
                res = [ r for r in mdtop.residues if r.name not in ('HOH') ]
            # Get atom indices
            for r in res:
                for a in r.atoms:
                    self.atom_indices.append(a.index)
            self.ref_traj = ref_traj.atom_slice(self.atom_indices)
        
        # Load trajectory
        netcdf = os.path.join(self.input_directory_path, trajectory_netcdf)
        traj = mdtraj.load(netcdf, top=self.ref_traj.topology, stride=stride)
        if self.atom_indices:
            self.traj = traj.atom_slice(self.atom_indices)
        else:
            self.traj = traj

        
class RNASystem(BaseDataLoader):
    """RNA system class to compute experimental observables from MD simulations.
    
    Methods
    -------
    radian_to_degree(a):
        Convert angle from radian to degree.

    compute_jcouplings(couplings=None, residues=None):
        Compute J-couplings from MD trajectory.

    get_available_couplings:
        Get available couplings to compute.


    Examples
    --------

    Compute J-couplings from MD trajectory.

    >>> from experiment import RNASystem
    >>> rna = RNASystem(input_directory_path='path/to/input', output_directory_path='path/to/output')
    >>> rna.load_traj()
    >>> couplings = rna.compute_jcouplings(couplings=['H1H2', 'H2H3', 'H3H4'], residues=['A_1_0'])
    """

    def __init__(self, **kwargs):
        """Initialize RNA system object.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to initialize the RNA system object.

            output_directory_path : str, optional
                Output directory path. Default is None.
            input_directory_path : str, optional
                Input directory path. Default is None.
        """
        super(RNASystem, self).__init__(**kwargs)
        

    def radian_to_degree(self, a):
        """Convert angle from radian to degree.
        
        Parameters
        ----------
        a : float
            Angle in radian.

        Returns
        -------
        a : float
            Angle in degree.
        """
        a[np.where(a<0.0)] += 2.*np.pi
        a *= 180.0/np.pi
        return a


    def compute_jcouplings(self, couplings=None, residues=None):
        """Compute J-couplings from MD trajectory.
        
        TODO
        ----
        * Compute confidence interval.

        Parameters
        ----------
        couplings : str, optional
            Name of the couplings to compute. Default is None. 
            If a list of couplings to be chosen from [H1H2,H2H3,H3H4,1H5P,2H5P,C4Pb,1H5H4,2H5H4,H3P,C4Pe,H1C2/4,H1C6/8] 
            is provided, only the selected couplings will be computed. Otherwise, all couplings will be computed.

            Sugar: H1H2, H2H3, H3H4
            Beta:  1H5P, 2H5P, C4Pb
            Gamma: 1H5H4, 2H5H4
            Epsilon: H3P, C4Pe
            Chi: H1C2/4, H1C6/8

        residues : list, optional
            List of residues to compute the couplings. Default is None.
            If None, all residues will be considered. The residue naming convention is RESNAME_RESNUMBER_CHAININDEX.
            For example, A_1_0, G_2_0, etc.

        Returns
        -------
        coupling_dict : dict
            Dictionary containing the computed J-couplings. The keys are the residue names and the values are the 
            computed J-couplings. The values are dictionaries containing the computed J-couplings and their standard 
            deviations. For example, {'A_1_0': {'H1H2': {'avg': 5.0, 'std': 0.1}, 'H2H3': {'avg': 5.0, 'std': 0.1}, ...}, ...}
        """
        import barnaba as bb

        _logger.info("Computing J-couplings from MD trajectory...")

        if couplings is not None:
            # Check if the provided coupling names are valid
            assert all(c in self.get_available_couplings for c in couplings), f"Invalid coupling name: {couplings}"
        else:
            couplings = self.get_available_couplings

        # values: [N: # of frames, M: # of nucleobases, X: # of couplings]
        # residue_list: list of M nucleobases
        values, resname_list = bb.jcouplings_traj(self.traj, couplings=couplings, residues=residues)

        # Loop over residues and couplings to store the computed values
        coupling_dict = dict()
        for i, resname in enumerate(resname_list):
            _values = values[:,i,:]  # Coupling values of i-th residue
            values_by_names = dict()
            for j, coupling_name in enumerate(couplings):
                # Function to replace np.nan with None
                avg = np.round(_values[:,j].mean(), 5)  # Mean value of H1H2 coupling of i-th residue
                std = np.round(_values[:,j].std(), 5)   # Standard deviation of H1H2 coupling of i-th residue

                replace_nan_with_none = lambda x: None if np.isscalar(x) and np.isnan(x) else x
                avg = replace_nan_with_none(avg)
                std = replace_nan_with_none(std)
                if avg:
                    avg = avg.item()
                if std:
                    std = std.item()
                # Convert numpy.float to float to avoid serialization issues
                values_by_names[coupling_name] = {'avg': avg, 'std': std}
            coupling_dict[resname] =  values_by_names

        return coupling_dict
    

    @property
    def get_available_couplings(self):
        """Get available couplings to compute.
        
        Returns
        -------
        available_coupling_names : list
            List of available couplings to compute.
        """
        import barnaba as bb
        available_coupling_names = list(bb.definitions.couplings_idx.keys())
        return available_coupling_names
        

class ProteinSystem(BaseDataLoader):
    def __init__(self, **kwargs):
        super(ProteinSystem, self).__init__(**kwargs)
        raise NotImplementedError("ProteinSystem class is not implemented yet.")


class ProteinLigandSystem(BaseDataLoader):
    def __init__(self, **kwargs):
        super(ProteinLigandSystem, self).__init__(**kwargs)
        raise NotImplementedError("ProteinLigandSystem class is not implemented yet.")
