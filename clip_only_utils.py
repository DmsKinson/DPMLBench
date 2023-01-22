from opacus import PrivacyEngine
from opacus.optimizers.optimizer import DPOptimizer,_check_processed_flag ,_mark_as_processed
from torch import optim ,nn
from typing import List, Union
from torch.utils.data.dataloader import DataLoader
import warnings 

class ClipOnlyOptimizer(DPOptimizer):
    def add_noise(self):
        # set flag but no noise addition 
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            p.grad = p.summed_grad.view_as(p.grad)
            _mark_as_processed(p.summed_grad)

class ClipOnlyPrivacyEngine(PrivacyEngine):
    def make_private_with_epsilon(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        target_epsilon: float,
        target_delta: float,
        epochs: int,
        max_grad_norm: float,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        noise_generator=None,
        **kwargs,
    ):
        if len(self.accountant) > 0:
            warnings.warn(
                "You're calling make_private_with_epsilon with non-zero privacy budget "
                "already spent. Returned noise_multiplier assumes zero starting point, "
                "so your overall privacy budget will be higher."
            )

        return self.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            # set noise_multiplier to zero, not actually used in the following codes 
            noise_multiplier=0,
            max_grad_norm=max_grad_norm,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
        )

    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        return ClipOnlyOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
        )

