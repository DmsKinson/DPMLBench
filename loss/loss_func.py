import torch

def MSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out =  0.5*(((logits-target)**2).sum(dim=1)).mean()
    return out

def MSE_vec(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out =  0.5*(((logits-target)**2).sum(dim=1))
    return out

def CE(logits, label):
    CE_loss = torch.nn.CrossEntropyLoss()
    loss = CE_loss(logits, label)
    return loss

def L2(features):
    features = features.view(features.size(0),-1)
    features_L2 =  0.5*(features**2)
    features_L2_per_example =  features_L2.mean(dim=1)
    return features_L2_per_example

def MSE_Focal_L2(logits, label, epoch, features, threshold_on_epoch, Alpha_Emb):

    # Compute the probability of the ground-truth label for Focal loss
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    ce_loss = loss(logits, label)
    pt = torch.exp(-ce_loss)
    gamma=5



    # Compute the coefficient of losses
    sigmoid_coef = torch.nn.Sigmoid()
    eta=sigmoid_coef(torch.tensor(epoch-threshold_on_epoch,  dtype=torch.float64).cuda())
    

    Focal_loss=((1-pt) ** gamma) * ce_loss
    
    MSE_loss= MSE_vec(logits, label)

    L2_loss=0


    for i in range(len(features)):
        L2_loss=L2_loss+L2(features[i])

    Total_loss= (eta*Focal_loss+ (1-eta)*MSE_loss + ((1-eta)/Alpha_Emb)*L2_loss).mean()

    return Total_loss 

def MSE_Focal(logits, label, epoch, thrs_epoch=0, alpha_emb=1):
    
    # Compute the probability of the ground-truth label for Focal loss
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    ce_loss = loss(logits, label)
    pt = torch.exp(-ce_loss)
    gamma=5

    # Compute the coefficient of losses
    sigmoid_coef = torch.nn.Sigmoid()
    eta=sigmoid_coef(torch.tensor(epoch-thrs_epoch,  dtype=torch.float64).cuda())
    

    Focal_loss=((1-pt) ** gamma) * ce_loss
    
    MSE_loss= MSE_vec(logits, label)


    Total_loss= (eta*Focal_loss+ (1-eta)*MSE_loss).mean()  

    return Total_loss 