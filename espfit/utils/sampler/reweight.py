import os
import logging
from espfit.app.sampler import SetupSampler, BaseSimulation

_logger = logging.getLogger(__name__)


class SetupSamplerReweight(SetupSampler, BaseSimulation):

    def __init__(self, weight=1, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

        
    def _get_experiment_data(self, target_class, target_name):
        import yaml
        from importlib.resources import files

        yaml_file = str(files('espfit').joinpath(f'data/target/{target_class}/{target_name}/experiment.yml'))
        with open(yaml_file, 'r') as f:
            d = yaml.safe_load(f)

        # {'resi_1': {'1H5P': {'name': 'beta_1', 'value': None, 'operator': None, 'error': None}}}
        return d['experiment_1']['measurement']


    def get_effective_sample_size(self):
        # Compute effective sample size
        neff = 0.5        
        return neff


    def _compute_observable(self):
        if self.target_class == 'nucleoside':
            from espfit.app.analysis import RNASystem
            target = RNASystem()
            target.load_traj(input_directory_path=self.output_directory_path)
            pred = target.compute_jcouplings()
            #_logger.debug(f'Computed observable: {pred}')
        else:
            raise NotImplementedError(f'Observable for {self.target_class} is not implemented.')

        import yaml
        with open(os.path.join(self.output_directory_path, 'pred.yaml'), 'w') as f:
            yaml.dump(pred, f, allow_unicode=True)

        return pred


    def compute_loss(self):
        # Compute experimental observable
        exp = self._get_experiment_data(self.target_class, self.target_name)
        pred = self._compute_observable()

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

                    # TODO: change to debug
                    _logger.info(f'Exp ({resi_index}-{key}): {exp}')
                    _logger.info(f'Pred ({resi_index}-{key}): {pred}')

                    # Compute loss
                    numerator = (pred_value - exp_value) ** 2
                    dominator = (exp_error ** 2 + pred_error ** 2)
                    loss.append(numerator / dominator)
                

        # Compute loss
        import torch
        loss_avg = torch.mean(torch.tensor(loss))
        _logger.info(f'Computed sampler loss: {loss_avg.item()}')

        return loss_avg