import torch
from time import time
from tools import MemoryManagerProxy

def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train(model, train_loader, scattering, optimizer, criterion, K, w, h,max_physical_bs,**kwargs):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0
    start = time()
    scatter_total = 0

    with MemoryManagerProxy(is_private=kwargs.get('private',False),data_loader=train_loader, max_physical_batch_size=max_physical_bs, optimizer=optimizer) as new_loader:
        # new_loader = train_loader
        for batch_idx, (data, target) in enumerate(new_loader):

            data, target = data.to(device), target.to(device)
            t_start = time()
            with torch.no_grad():
                data = scattering(data).view(-1,K,w,h)
            scatter_total += time() - t_start
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss += loss.item()
            num_examples += len(data)

        train_loss /= num_examples
        train_acc = 100. * correct / num_examples

        print(f'Time cost: total={time()-start:.2f}s, scattering={scatter_total:.2f}s, Train set: Average loss: {train_loss:.4f}, '
                f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc


def test(model, test_loader, criterion, scattering, K, w, h):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0
    start = time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = scattering(data).view(-1,K,w,h)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    print(f'Time cost:{time()-start}s, Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)')

    return test_loss, test_acc
