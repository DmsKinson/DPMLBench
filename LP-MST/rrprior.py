import torch
import math
import random

def rr(epsilon:float, y:int, k, others:list):
    # too large epsilon to compute e^eps, set p as 1 approximately 
    if(epsilon > 100):
        p = 1
    else:
        p = math.exp(epsilon) / (math.exp(epsilon)+k-1)
    randp = torch.rand(1).item()
    # output y with probability e^eps/(e^eps+k-1)
    if randp <= p :
        return y
    # output y' in Y_k/{y} with probability 1/(e^eps+k-1)
    else:
        return random.choice(others)

# pr:[K]  y:1  K:1
def rr_topk(pr:torch.Tensor, y:torch.Tensor, k:torch.Tensor, epsilon):
    n_label = pr.shape[0]
    top = torch.argsort(pr,descending=True)[:k].tolist()  # [K]
    others = list(range(n_label))
    others.remove(y)
    # if y in Y_k, then output y with probability e^eps/(e^eps+k-1) and output y' in Y_k/{y} with probability 1/(e^eps+k-1)
    if y in top:
        out = rr(epsilon=epsilon, k=k, y=y, others=others)
    # if y not in Y_k, output an element from Y_k uniformly at random
    else :
        out = random.choice(others)
    return out

def rr_prior(pr:torch.Tensor, y, K, epsilon):
    w = []
    out = []
    for k in range(1,K+1):
        # same reason as rr
        if(epsilon > 100):
            p = 1
        else:
            p = math.exp(epsilon) / (math.exp(epsilon)+k-1)
        top = torch.argsort(pr,descending=True,dim=1)[:,:k]  # [B,k]
        top_pr = pr.index_select(dim=1,index=top[1])
        w.append(top_pr.sum(dim=1) * p)     # [B,1]
        
    w = torch.stack(w,dim=1)
    best_k = torch.argmax(w,dim=1) + 1  # [B]
    
    for i,k in enumerate(best_k):
        o = rr_topk(pr[i],y[i],k.item(),epsilon)
        out.append(o)

    return best_k.float().mean(), torch.Tensor(out).long()