import os
import logging
from espfit.app.sampler import SetupSampler, BaseSimulation

_logger = logging.getLogger(__name__)


class SetupSamplerReweight(object):

    def __init__(self):
        self.samplers = None
        self.samplers_old = None
        self.weights = None   # list


    def run(self):
        for sampler in self.samplers:
            _logger.info(f'Running simulation for {sampler.target_name} for {sampler.nsteps} steps...')
            sampler.minimize()
            sampler.run()


    def update(self, samplers):
        # Update sampler
        self.samplers_old = self.samplers
        self.samplers = samplers


    def get_effective_sample_size(self):
        # Compute effective sample size

        # U(x0, theta0)
        old_potential_energy = 0

        # U(x0, theta1)
        reduced_potential_energy = 0

        neff = 0.5
        return neff
    
        
    def compute_loss(self):

        loss_list = []
        for sampler in self.samplers:
            loss = self._compute_loss_per_system(sampler)  # torch.tensor
            loss_list.append(loss)

        # list of torch.tensor        
        return loss_list
    

    def _compute_loss_per_system(self, sampler):

        import torch

        # Compute experimental observable
        exp = self._get_experiment_data(sampler.target_class, sampler.target_name)
        pred = self._compute_observable(sampler.atomSubset, sampler.target_class, sampler.output_directory_path)

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
                    pred_error = list(pred.values())[resi_index][key]['std']  # standard deviation
                    _logger.debug(f'Exp ({resi_index}-{key}): {exp}')
                    _logger.debug(f'Pred ({resi_index}-{key}): {pred}')
                    # Compute loss
                    numerator = (pred_value - exp_value) ** 2
                    dominator = (exp_error ** 2 + pred_error ** 2)
                    loss.append(numerator / dominator)
        # Compute loss
        loss_avg = torch.mean(torch.tensor(loss))
        _logger.info(f'Computed sampler loss: {loss_avg.item()}')

        return loss_avg


    def _get_experiment_data(self, target_class, target_name):
        import yaml
        from importlib.resources import files

        yaml_file = str(files('espfit').joinpath(f'data/target/{target_class}/{target_name}/experiment.yml'))
        with open(yaml_file, 'r', encoding='utf8') as f:
            d = yaml.safe_load(f)

        # {'resi_1': {'1H5P': {'name': 'beta_1', 'value': None, 'operator': None, 'error': None}}}
        return d['experiment_1']['measurement']


    def _compute_observable(self, atomSubset, target_class, output_directory_path):
        if target_class == 'nucleoside':
            from espfit.app.analysis import RNASystem
            target = RNASystem(atomSubset=atomSubset)
            target.load_traj(input_directory_path=output_directory_path)
            pred = target.compute_jcouplings()
            _logger.debug(f'Computed observable: {pred}')
        else:
            raise NotImplementedError(f'Observable for {target_class} is not implemented.')

        import yaml
        with open(os.path.join(output_directory_path, 'pred.yaml'), 'w') as f:
            yaml.dump(pred, f, allow_unicode=True)

        return pred
