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
            val = target.compute_jcouplings()
            _logger.info(f'Computed observable: {val}')
        else:
            raise NotImplementedError(f'Observable for {self.target_class} is not implemented.')

        import yaml
        with open(os.path.join(self.output_directory_path, 'pred.yaml'), 'w') as f:
            yaml.dump(val, f, allow_unicode=True)

        return val


    def compute_loss(self):
        # Compute experimental observable
        val = self._compute_observable()
        _logger.info(f'Compute loss')

        # Compute loss
        import torch
        loss = torch.tensor(0.0)

        return loss