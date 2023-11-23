import torch

def train(model, criterion, optimizer, label_loader, unlabel_loader,unlabel_aug_loader, lambda_u, mask_threshold, temperature_T, device):
    l_loss = ul_loss = 0
    idx = 0
    model.train()

    for  (l_data, l_label),(ul_data,_),(ul_aug,_) in zip(label_loader,unlabel_loader,unlabel_aug_loader):
        idx += 1
        l_size= len(l_data)

        inputs = torch.cat((l_data, ul_data, ul_aug)).to(device)
        targets_x = l_label.to(device)
        logits = model(inputs)
        logits_x = logits[:l_size]
        logits_ul_w, logits_ul_s = logits[l_size:].chunk(2)
        
        del logits
        Lx = criterion(logits_x, targets_x)
        l_loss += Lx.item()

        targets_u = torch.softmax(logits_ul_w.detach()/temperature_T, dim=-1)
        # print(targets_u)
        max_probs, _ = torch.max(targets_u, dim=-1)
        mask = max_probs.ge(mask_threshold).float()
        # print(mask)
        Lu = (-(targets_u * torch.log_softmax(logits_ul_s, dim=-1)).sum(dim=-1) * mask).mean()
        ul_loss += Lu.item()
        # print('ul_loss:',ul_loss)
        loss = Lx + lambda_u * Lu

        # assert False
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print('train loss:',(l_loss+ul_loss)/idx,'label loss:',l_loss/idx,'unlabel loss:',ul_loss/idx)
    return l_loss/idx, ul_loss/idx


# def main():
#     dataset = 'mnist'
#     net = 'resnet'
#     lr = 0.01
#     nesterov = True
#     epoch = 100
#     batchsize = 128

#     device = torch.device('cuda')
#     df = DataFactory(dataset)
#     trainset = df.getTrainSet('full')
#     model = get_model(net,dataset)
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr,
#                           momentum=0.9, nesterov=nesterov)
#     criterion = torch.nn.CrossEntropyLoss()

    

    for e in range(epoch):
        train(e,model,criterion,optimizer,)