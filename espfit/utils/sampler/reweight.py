"""
Compute effective sample size and weights for each simulation.

TODO
----
* Check J-coupling experimental error. Currently, fixed to 0.5 Hz.
"""
import os
import logging

_logger = logging.getLogger(__name__)


class SetupSamplerReweight(object):
    """Setup sampler for reweighting simulation.

    This class is responsible for setting up the sampler for reweighting simulation.
    It provides methods to run the simulation, compute the effective sample size,
    compute the loss, and compute the weighted observable.

    Methods
    -------
    run():
        Runs the simulation for each sampler.

    get_effective_sample_size(temporary_samplers):
        Computes the effective sample size and sampling weights for each sampler.

    compute_loss():
        Computes the loss for each sampler.    
    """
    def __init__(self):
        self.samplers = None
        self.weights_neff_dict = dict()   # {'target_name': {'weights': w_i}, {'neff': neff}}
        self.force_group_names = ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NonbondedForce']
        self.exclude_n_frames = 0.1


    def run(self):
        """Runs the simulation for each sampler.
        
        Returns
        -------
        None
        """
        for sampler in self.samplers:
            _logger.info(f'Re-run simulation for {sampler.target_name}')
            sampler.minimize()
            sampler.run()


    def get_effective_sample_size(self, temporary_samplers):
        """Computes the effective sample size and sampling weights for each sampler.

        Parameters
        ----------
        temporary_samplers : list
            List of temporary samplers.

        Returns
        -------
        float
            The minimum effective sample size among all samplers.
        """
        import numpy as np
        from openmm.unit import kilocalories_per_mole as kcalpermol
        from espfit.utils.units import KB_T_KCALPERMOL
        from espfit.app.analysis import BaseDataLoader
        
        _logger.info(f'Compute effective sample size and sampling weights')

        if self.samplers is None:
            _logger.info('No samplers found. Return effective sample size -1.')
            return -1

        for sampler, temporary_sampler in zip(self.samplers, temporary_samplers):
            _logger.info(f'Analyzing {sampler.target_name}...')

            # Get temperature
            temp0 = sampler.temperature._value
            temp1 = temporary_sampler.temperature._value
            assert temp0 == temp1, f'Temperature should be equivalent but got sampler {temp0} K and temporary sampler {temp1} K'
            beta = 1 / (KB_T_KCALPERMOL * temp0)
            #_logger.debug(f'beta temperature in kcal/mol: {beta}')

            # Get position from trajectory
            baseloader = BaseDataLoader(atomSubset=sampler.atomSubset)
            baseloader.load_traj(input_directory_path=sampler.output_directory_path, exclude_n_frames=self.exclude_n_frames)
            _logger.info(f'Found {baseloader.traj.n_frames} frames from {sampler.output_directory_path}')
            
            # Compute weights and effective sample size
            w_arr = []
            for i in range(baseloader.traj.n_frames):
                # U(x0, theta0)
                sampler.simulation.context.setPositions(baseloader.traj.openmm_positions(i))
                for gid, force in enumerate(sampler.simulation.system.getForces()):
                    if force.getName() in self.force_group_names:
                        try:
                            potential_energy += sampler.simulation.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy()
                        except:
                            potential_energy = sampler.simulation.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy()
                # U(x0, theta1)
                temporary_sampler.simulation.context.setPositions(baseloader.traj.openmm_positions(i))
                for gid, force in enumerate(temporary_sampler.simulation.system.getForces()):
                    if force.getName() in self.force_group_names:
                        try:
                            reduced_potential_energy += temporary_sampler.simulation.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy()
                        except:
                            reduced_potential_energy = temporary_sampler.simulation.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy()
                # deltaU = U(x0, theta1) - U(x0, theta0)
                delta = (reduced_potential_energy - potential_energy).value_in_unit(kcalpermol)
                # w = ln(exp(-beta * delta))
                w = -1 * beta * delta
                w_arr.append(w)

                _logger.info(f'U(x0, theta0): {potential_energy.value_in_unit(kcalpermol):10.3f} kcal/mol')
                _logger.info(f'U(x0, theta1): {reduced_potential_energy.value_in_unit(kcalpermol):10.3f} kcal/mol')
                _logger.info(f'deltaU:        {delta:10.3f} kcal/mol')
                _logger.info(f'w:             {w:10.3f}')

            # Compute weights and effective sample size (ratio: 0 to 1)
            # Prevent RuntimeWarning: overflow encountered in exp
            w_arr = np.float128(w_arr)
            w_i = np.exp(w_arr) / np.sum(np.exp(w_arr))
            neff = np.sum(w_i) ** 2 / np.sum(w_i ** 2) / len(w_i)
            _logger.debug(f'w_i_sum:       {np.sum(w_i):10.3f}')
            _logger.debug(f'neff:          {neff:10.3f}')

            # Check if sum of weights is 1
            sum_w = abs(np.sum(w_i))
            assert abs(1 - sum_w) <= 0.001, f"Weight sum {sum_w} is greater than tolerance 0.001"

            self.weights_neff_dict[f'{sampler.target_name}'] = {'neff': neff, 'weights': w_i}
            _logger.debug(f'{self.weights_neff_dict}')
            neffs = [self.weights_neff_dict[key]['neff'] for key in self.weights_neff_dict.keys()]

        return min(neffs)
    
        
    def compute_loss(self):
        """Computes the loss for each sampler.

        Returns
        -------
        list
            List of torch tensors representing the loss for each sampler.
        """
        loss_list = []
        for sampler in self.samplers:
            _logger.debug(f'Compute loss for {sampler.target_name}')
            loss = self._compute_loss_per_system(sampler)  # torch.tensor
            loss_list.append(loss)

        return loss_list
    

    def _compute_loss_per_system(self, sampler):
        """Computes the loss per system for a given sampler.

        Parameters
        ----------
        sampler : object
            The sampler object.

        Returns
        -------
        torch.Tensor
            The loss per system as a torch tensor.
        """
        import torch

        # Compute experimental observable
        exp = self._get_experiment_data(sampler.target_class, sampler.target_name)
        pred = self._compute_weighted_observable(sampler.atomSubset, sampler.target_name, sampler.output_directory_path)

        loss = []
        for resi_index, exp_dict in enumerate(exp.values()):
            for key, value in exp_dict.items():
                # {'1H5P': {'name': 'beta_1', 'value': None, 'operator': None, 'error': None}}
                if value['operator'] in ['>', '<', '>=', '<=', '~'] or value['value'] == None:
                    # Dont use uncertain data
                    pass
                else:
                    exp_value = value['value']
                    exp_error = value['error']
                    if exp_error == None:
                        exp_error = 0.5  # TODO: Check experimental error
                    resi_index = int(resi_index)
                    pred_value = list(pred.values())[resi_index][key]['avg']
                    pred_error = list(pred.values())[resi_index][key]['std']
                    _logger.debug(f'Exp ({resi_index}-{key}): {exp}')
                    _logger.debug(f'Pred ({resi_index}-{key}): {pred}')

                    numerator = (pred_value - exp_value) ** 2
                    dominator = (exp_error ** 2 + pred_error ** 2)
                    loss.append(numerator / dominator)
        # Compute loss
        loss_avg = torch.mean(torch.tensor(loss))
        _logger.info(f'Sampler loss: {loss_avg.item():.3f}')

        return loss_avg


    def _get_experiment_data(self, target_class, target_name):
        """Retrieves the experimental data for a given target.

        Parameters
        ----------
        target_class : str
            The class of the target.

        target_name : str
            The name of the target.
        
        Returns
        -------
        dict : The experimental data for the target.
        """
        import yaml
        from importlib.resources import files

        yaml_file = str(files('espfit').joinpath(f'data/target/{target_class}/{target_name}/experiment.yml'))
        with open(yaml_file, 'r', encoding='utf8') as f:
            d = yaml.safe_load(f)

        # {'resi_1': {'1H5P': {'name': 'beta_1', 'value': None, 'operator': None, 'error': None}}}
        return d['experiment_1']['measurement']


    def _compute_weighted_observable(self, atomSubset, target_name, output_directory_path):
        """Computes the weighted observable for a given target.

        Parameters
        ----------
        atomSubset : str
            The atom subset.

        target_name : str
            The name of the target.

        output_directory_path : str
            The output directory path.

        Returns
        -------
        dict : The computed weighted observable.
        """
        import yaml
        from espfit.app.analysis import RNASystem

        # Load trajectory
        target = RNASystem(atomSubset=atomSubset)
        target.load_traj(input_directory_path=output_directory_path, exclude_n_frames=self.exclude_n_frames)
        
        # Compute observable
        if self.weights_neff_dict.keys():
            pred = target.compute_jcouplings(weights=self.weights_neff_dict[target_name]['weights'])
        else:
            pred = target.compute_jcouplings(weights=None)
        _logger.debug(f'Computed observable: {pred}')

        # Export observable
        with open(os.path.join(output_directory_path, 'pred.yaml'), 'w') as f:
            yaml.dump(pred, f, allow_unicode=True)

        return pred
