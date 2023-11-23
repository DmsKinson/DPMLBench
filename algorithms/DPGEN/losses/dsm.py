from logging import exception
import torch
import random, numpy
from datetime import datetime

# +
def private_select(perturbed_samples, samples, sigmas, labels, used_sigmas, k, eps, batch_size, p_sample_buff=None, sample_buff=None, label_buff=None, sigma_buff=None):
#     import ipdb; ipdb.set_trace()
    batch_idx = random.sample(range(len(samples)), batch_size)
    perturbed_samples = perturbed_samples[batch_idx]
    
    n = perturbed_samples.shape[0]
    m = samples.shape[0]
    
    if m < k:
        k = m
        
    D = torch.cdist(perturbed_samples.view(n, -1), samples.view(m, -1), p=float("inf")) / used_sigmas.squeeze()
#     print(datetime.now(), "D shape: " , D.shape)

    for i in range(n):        
        private_sigmas = used_sigmas.squeeze().clone()
        private_labels = labels.clone()
        
        ideal_sigmas = torch.max(torch.abs(perturbed_samples[i].view(1,-1) - samples.view(m, -1)), dim=1)[0] / 5

            
        weight = numpy.ones(k)
        
        # Avoid overflow when EPS = 1000, and distinguish it from EPS = 100
        if(eps > 100):
            eps = 101
        
        weight[batch_idx[i]] = numpy.exp(eps)
        sample_ix = random.choices(range(k), weights=weight)[0]
        
        if sample_ix == batch_idx[i]:
            new_sigma = private_sigmas[sample_ix]
            label_ix = private_labels[sample_ix]
        else:
            if D[i, sample_ix] <= 5:
                new_sigma = private_sigmas[sample_ix]
                label_ix = private_labels[sample_ix]
            elif D[i, sample_ix] > 5:
                new_label = torch.nonzero(ideal_sigmas[sample_ix] > sigmas)[0] - 1
                new_sigma = sigmas[new_label][0]
                label_ix = new_label[0]

#         import ipdb; ipdb.set_trace()
        p_sample_buff.append(perturbed_samples[i])
        sample_buff.append(samples[sample_ix])
        label_buff.append(label_ix)
        sigma_buff.append(new_sigma)
                
    return 0
# -



def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None,
            p_sample_buff=None, sample_buff=None, label_buff=None, sigma_buff=None, config=None):

    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
#     import ipdb; ipdb.set_trace()
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise

    try:
        k = config.training.k
        eps = config.training.epsilon
        batch_size = config.training.batch_size
                
        # mini-batch
        if len(p_sample_buff) < config.training.queue_size:
            private_select(perturbed_samples, samples, sigmas, labels, used_sigmas, k, eps, batch_size, p_sample_buff, sample_buff, label_buff, sigma_buff)
    
        if len(p_sample_buff) < batch_size:
            print(len(p_sample_buff))
            return 'pass'
        
        
        sample_idx = random.sample(range(len(p_sample_buff)), batch_size)
        
        perturbed_samples = torch.stack(list(p_sample_buff))[sample_idx]
        samples = torch.stack(list(sample_buff))[sample_idx]
        used_sigmas = torch.stack(list(sigma_buff))[sample_idx].view(samples.shape[0], *([1] * len(samples.shape[1:])))
        labels = torch.stack(list(label_buff))[sample_idx]
        noise = perturbed_samples - samples
        
    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
        print(e)
       
    
    
    target = - 1 / (used_sigmas ** 2) * noise

    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
    
    return loss




