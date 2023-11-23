import numpy as np

from rdp_accountant import compute_rdp, get_privacy_spent

DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
DEFAULT_SIGMA_MIN_BOUND = 0.01
DEFAULT_SIGMA_MAX_BOUND = 10
MAX_SIGMA = 2000

SIGMA_PRECISION = 0.01

def get_sigma(sample_rate, epochs, target_eps, delta, rgp=True):
    orders = DEFAULT_ALPHAS

    eps = float("inf")
    sigma_min = DEFAULT_SIGMA_MIN_BOUND
    sigma_max = DEFAULT_SIGMA_MAX_BOUND
    steps = int(epochs/sample_rate)

    while eps > target_eps:
        sigma_max = 2*sigma_max
        if(rgp):
            rdp = compute_rdp(sample_rate, sigma_max, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(sample_rate, sigma_max, steps, orders)
        eps, _, _ = get_privacy_spent(orders=orders, rdp=rdp, target_delta=delta)
        if sigma_max > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while sigma_max - sigma_min > SIGMA_PRECISION:

        sigma = (sigma_min + sigma_max)/2
        if(rgp):
            rdp = compute_rdp(sample_rate, sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(sample_rate, sigma, steps, orders)
        eps, _, _ = get_privacy_spent(orders=orders, rdp=rdp, target_delta=delta)

        if eps < target_eps:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma, eps