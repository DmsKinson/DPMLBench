import torch 
import torchvision.transforms as transforms
from torchvision.transforms import RandAugment
from torch.utils.data import DataLoader,TensorDataset

class TransformUDA(object):
    def __init__(self, n_ops=2, mag=10):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugment(num_ops=n_ops, magnitude=mag),
            ])
        
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong

def rand_aug_process(unlabel_set):
    dataloader = DataLoader(unlabel_set, batch_size=64)
    rand_aug = transforms.RandAugment(2,10)
    
    aug_list = []
    label_list = []
    for data,label in dataloader:
        assert torch.max(data)<=1 and torch.min(data)>=0, f"input data should be origin data transfer to (0,1)"
        assert data.shape[1] in (1,3), f"input data's dimension should be (...,1 or 3,H,W), :{data.shape}"

        data = (data*255).to(dtype=torch.uint8)
        afterAugData = rand_aug(data)
        data = afterAugData.to(dtype=torch.float32)/255
        aug_list.append(data)
        label_list.append(label)

    rand_aug_data = torch.cat(aug_list)
    labels = torch.cat(label_list)
    augset = TensorDataset(rand_aug_data, labels)

    return augset
    