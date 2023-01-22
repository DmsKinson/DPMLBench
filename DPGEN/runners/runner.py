import sys,os
pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(pwd,'..','..'))

from time import time
import numpy as np

from DataFactory import DataFactory
from losses.dsm import anneal_dsm_score_estimation
from collections import deque
import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from helper.net import RefineNet, MultiClassifier
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from helper import anneal_Langevin_dynamics
from helper import get_sigmas
from helper.ema import EMAHelper
from tools import load_model
from db_models import DB_Model

def get_model(config):
    if config.data.dataset == 'cifar10' or config.data.dataset == 'svhn' or config.data.dataset == 'fmnist' or config.data.dataset == 'mnist':
        return RefineNet(config).to(config.device)

def label_sample(sample, label_model):
    label_model.eval()
    outputs = label_model(sample)
    predicts = torch.argmax(outputs,dim=1)
    return predicts
# +
class Runner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)
        
        N = self.config.training.queue_size
        self.p_sample_buff = deque(maxlen=N)
        self.sample_buff = deque(maxlen=N)
        self.label_buff = deque(maxlen=N)
        self.sigma_buff = deque(maxlen=N)

    def train(self):
        df = DataFactory(self.config.data.dataset)
        dataset = df.getTrainSet('target')
        test_dataset = df.getTestSet('target')
        dataloader = DataLoader(dataset, batch_size=self.config.training.k, shuffle=True,
                                num_workers=self.config.data.num_workers, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        # import ipdb; ipdb.set_trace()

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            ### Make sure we can resume with different eps
            states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)

        for epoch in range(start_epoch, self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                score.train()
                step += 1

                X = X.to(self.config.device)
                X = data_transform(self.config, X)
                # import ipdb; ipdb.set_trace()
                
                kwargs = {'p_sample_buff': self.p_sample_buff, 'sample_buff': self.sample_buff, 
                        'label_buff':self.label_buff, 'sigma_buff':self.sigma_buff, 'config': self.config}

                losses = anneal_dsm_score_estimation(score, X, sigmas, None,
                                                   self.config.training.anneal_power,
                                                   hook=None, **kwargs)
                
                if losses == 'pass':
                    print(len(self.sample_buff))
                    continue
                    
                loss = losses.mean(dim=0)
                logging.info("step: {}, loss: {}".format(step, losses.mean().item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(score)

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    if self.config.model.ema:
                        test_score = ema_helper.ema_copy(score)
                    else:
                        test_score = score

                    test_score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    test_X = data_transform(self.config, test_X)

                    # with torch.no_grad():
                    #     # test_dsm_loss = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                    #     #                                             self.config.training.anneal_power,
                    #     #                                             hook=test_hook)
                    #     test_dsm_losses = anneal_dsm_score_estimation(test_score, test_X, sigmas, None,
                    #                                                 self.config.training.anneal_power)
                    #     # tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=step)
                    #     # logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))
                    #     logging.info("step: {}, test_loss: {}".format(step, test_dsm_losses.mean().item()))

                    #     del test_score

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    # torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                    if self.config.training.snapshot_sampling:
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()
                        init_samples = torch.rand(36, self.config.data.channels,
                                                  self.config.data.image_size, self.config.data.image_size,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)

                        all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                               self.config.sampling.n_steps_each,
                                                               self.config.sampling.step_lr,
                                                               final_only=True, verbose=True,
                                                               denoise=self.config.sampling.denoise)

                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, 6)
                        save_image(image_grid,
                                   os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))

                        del test_score
                        del all_samples

    def sample(self, max_try = 10):
        amount = self.config.sampling.amount
        batch_size = self.config.sampling.batch_size
        is_balance = self.config.sampling.balance
        print(f'''
        Start sampling.
        Amount:{amount}
        is_balance:{is_balance}
        ''')

        # label_model = MultiClassifier()
        # state = torch.load(os.path.join(pwd,'..','label_net',self.config.data.dataset,label_model._get_name()+'.pt'))
        # label_model.load_state_dict(state)
        print("Load label model.")
        label_model = load_model(
            DB_Model.type=='shadow',
            DB_Model.eps==None,
            DB_Model.dataset==self.config.data.dataset,
            DB_Model.net=='inception',
        )

        save_dir = os.path.join(pwd,'..','label_dataset',self.config.data.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir,exist_ok=True)

        score = get_model(self.config)
        # import ipdb; ipdb.set_trace()
        score = torch.nn.DataParallel(score)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
        score.load_state_dict(states[0])

        if self.config.model.ema:
            ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)

        if self.config.model.ema:
            test_score = ema_helper.ema_copy(score)
        else:
            test_score = score

        test_score.eval()
    
        if(is_balance):
            # 10 classes default
            per_class_cnt = torch.zeros(10,dtype=torch.long)
            per_class_amount = amount//10
            # eliminate float error
            amount = per_class_amount*10
        all_samples = []
        all_labels = []
        cur_cnt = 0
        try_cnt = 0
        while(cur_cnt < amount):
            time_start = time()
            init_samples = torch.rand(batch_size, self.config.data.channels,
                                        self.config.data.image_size, self.config.data.image_size,
                                        device=self.config.device)
            init_samples = data_transform(self.config, init_samples)

            samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                    self.config.sampling.n_steps_each,
                                                    self.config.sampling.step_lr,
                                                    final_only=True, verbose=False,
                                                    denoise=self.config.sampling.denoise)

            sample = samples[-1].reshape(samples[-1].shape[0], self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_size)

            sample = inverse_data_transform(self.config, sample)
            labels = label_sample(sample, label_model)
            if(is_balance):
                cnt = torch.bincount(labels,minlength=10)
                indexes = []
                print('distribution:',cnt)
                for i in range(10):
                    if(cnt[i]==0):
                        continue
                    if(per_class_cnt[i] >= per_class_amount):
                        continue
                    else:
                        index = (labels==i).nonzero().squeeze_(dim=1)
                        if(cnt[i] + per_class_cnt[i] > per_class_amount):
                            redundant = int(cnt[i] + per_class_cnt[i] - per_class_amount)
                            index = index[:-redundant]
                        per_class_cnt[i] += len(index)
                        indexes.append(index)
                
                if(len(indexes)==0):
                    print("No valid sample generated.")
                    try_cnt += 1
                    if(try_cnt==max_try):
                        print("Can't generate balance dataset.")
                        sys.exit(1)
                    continue
                else:
                    try_cnt = try_cnt//2

                print('per_class_cnt:',per_class_cnt)
                indexes = torch.cat(indexes)
                all_samples.append(sample[indexes])
                all_labels.append(labels[indexes])
                cur_cnt += len(indexes)
            else:   
                all_samples.append(sample)
                all_labels.append(labels)
                cur_cnt += len(labels)
            print(f'Already Generate {cur_cnt}/{amount} samples')
            print('Time cost:',time()-time_start)

            # save
            all_samples_tensor = torch.cat(all_samples)
            all_labels_tensor = torch.cat(all_labels)
            torch.save((all_samples_tensor,all_labels_tensor), os.path.join(save_dir,f'{self.config.training.epsilon}.pt'))
            del all_samples_tensor, all_labels_tensor

        

