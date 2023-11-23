from opacus.accountants import RDPAccountant

MAX_SIGMA = 2000
DEFAULT_SIGMA_MIN_BOUND = 0.01
DEFAULT_SIGMA_MAX_BOUND = 10
SIGMA_PRECISION = 1e-3

def search_sigma(target_eps, n_epoch, decay_fn, delta, sample_rate,sigma_min=0.01,sigma_max=100,**kwargs):
    '''
    binary search initial sigma for Adaptive alloc 
    '''
    eps = float("inf")
    sigma_min = DEFAULT_SIGMA_MIN_BOUND
    sigma_max = DEFAULT_SIGMA_MAX_BOUND

    while eps > target_eps:
        sigma_max = 2*sigma_max
        if sigma_max > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")
        eps = forward(sigma0=sigma_max,n_epoch=n_epoch,decay_fn=decay_fn,sample_rate=sample_rate,delta=delta)

    print('initial:eps=',eps,'sigma_max=',sigma_max)

    while sigma_max - sigma_min > SIGMA_PRECISION:
        sigma = (sigma_min + sigma_max)/2
        eps = forward(sigma0=sigma,n_epoch=n_epoch,decay_fn=decay_fn,sample_rate=sample_rate,delta=delta)
        print('iter:eps=',eps,'sigma=',sigma)
        if eps < target_eps:
            sigma_max = sigma
        else:
            sigma_min = sigma

    return sigma, eps

def forward(sigma0, n_epoch, decay_fn, delta,sample_rate, **kwargs):
    acc = RDPAccountant()
    for t in range(n_epoch):
        total = 0
        sigma = decay_fn(sigma0=sigma0,last_epoch=t,**kwargs)
        while total < 1:
            acc.step(noise_multiplier=sigma,sample_rate=sample_rate)
            total += sample_rate
    eps = acc.get_epsilon(delta)
    return eps