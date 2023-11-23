from opacus.scheduler import _NoiseScheduler
from opacus.optimizers import DPOptimizer
import math

time_lambda = lambda sigma0,k=0.05,last_epoch=-1: sigma0*(1+k*last_epoch)
step_lambda = lambda sigma0,k=0.6,period=10,last_epoch=-1 : sigma0*k**(last_epoch//period)
exp_lambda = lambda sigma0,k=0.01,last_epoch=-1 : sigma0*math.exp(-k*last_epoch)
poly_lambda = lambda sigma0,sigma1=2,k=3,period=100,last_epoch=-1 : (sigma0 - sigma1) * (1-last_epoch/period)**k + sigma1

lambda_dict = {
    'time':time_lambda,
    'step':step_lambda,
    'exp':exp_lambda,
    'poly':poly_lambda
}

def get_lambda(fn_name):
    return lambda_dict[fn_name]

class TimeBasedDecay(_NoiseScheduler):
    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        self.sigma0 = optimizer.noise_multiplier
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_noise_multiplier(self):
        return time_lambda(self.sigma0,last_epoch=self.last_epoch)

class StepDecay(_NoiseScheduler):
    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        self.sigma0 = optimizer.noise_multiplier
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_noise_multiplier(self):
        return step_lambda(self.sigma0,last_epoch=self.last_epoch)

class ExpDecay(_NoiseScheduler):
    def __init__(self, optimizer: DPOptimizer, *,last_epoch=-1):
        self.sigma0 = optimizer.noise_multiplier
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_noise_multiplier(self):
        return exp_lambda(self.sigma0,last_epoch=self.last_epoch)

class PolyDecay(_NoiseScheduler):
    def __init__(self, optimizer: DPOptimizer, *, last_epoch=-1):
        self.sigma0 = optimizer.noise_multiplier
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_noise_multiplier(self):
        return poly_lambda(self.sigma0,last_epoch=self.last_epoch)

