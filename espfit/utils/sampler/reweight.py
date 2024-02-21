import os
import logging
from espfit.app.sampler import SetupSampler, BaseSimulation

_logger = logging.getLogger(__name__)


class SamplerReweight(SetupSampler, BaseSimulation):

    def __init__(self, weight=1, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight


    def get_effective_sample_size(self):
        # Compute effective sample size
        neff = 0.5
        return neff


    def _compute_observable(self):
        if self.target_class == 'nucleoside':
            from espfit.app.analysis import RNASystem
            target = RNASystem()
            target.load_traj(input_directory_path=self.output_directory_path)
            obs_calc = target.compute_jcouplings()
            _logger.info(f'Computed observable: {obs_calc}')
        else:
            raise NotImplementedError(f'Observable for {self.target_class} is not implemented.')

        import yaml
        with open(os.path.join(self.output_directory_path, 'observable.yaml'), 'w') as f:
            yaml.dump(obs_calc, f, allow_unicode=True)

        return obs_calc


    def compute_loss(self):
        # Compute experimental observable
        obs_calc = self._compute_observable()
        _logger.info(f'Computed observable: {obs_calc}')

        # Compute loss
        import torch
        loss = torch.tensor(0.0)

        return loss