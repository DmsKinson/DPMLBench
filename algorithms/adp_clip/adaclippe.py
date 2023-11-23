from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from torch import optim
from adaclipoptimizer import AdaClipDPOptimizer
from typing import Union, List

class AdpClipPE(PrivacyEngine):
    def _prepare_optimizer(self, optimizer: optim.Optimizer, *, noise_multiplier: float, max_grad_norm: Union[float, List[float]], expected_batch_size: int, loss_reduction: str = "mean", distributed: bool = False, clipping: str = "flat", noise_generator=None) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        return AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            target_unclipped_quantile=0.7,
            clipbound_learning_rate=0.1,
            max_clipbound=10,
            min_clipbound=0.05,
            unclipped_num_std=expected_batch_size/20,
            expected_batch_size=expected_batch_size,
            max_grad_norm=max_grad_norm
        )
