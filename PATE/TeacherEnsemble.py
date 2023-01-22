from genericpath import exists
import os
import pathlib
import random
import re
import shutil

import numpy as np
from numpy.core.fromnumeric import size
import torch
from torch import nn
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.types import Device
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from Teacher import Teacher

PREDS_NAME = '/preds.npy'
LABELS_NAME = '/labels.npy'

class TeacherEnsemble():

    def __init__(self, numTeachers:int = 0, net:str = 'simple', dataset:str = 'mnist', device:torch.device=torch.device('cpu'), root:str = None):
        self.teachers = []
        self.numTeachers = numTeachers
        if(root != None):
            self.loadTeachers(root)
        self.net = net
        self.device = device
        self.dataset = dataset

    def trainTeachers(self, trainData, criterion, device, batchSize=64, epoch=10, save_inplace=False, **kargs):
        print("\n**********************")
        print("Partitioning dataset")
        # partition dataset to subset
        teacherLoaders = []
        dataSize = len(trainData) // self.numTeachers
        # shuffle dataset before partition
        full_indices = list(range(len(trainData)))
        random.shuffle(full_indices)
        for i in range(self.numTeachers):
            indices = full_indices[i*dataSize:(i+1)*dataSize]
            # indices = list(range(i*dataSize, (i+1)*dataSize))
            subsetData = Subset(trainData, indices)
            loader = DataLoader(subsetData, batch_size=batchSize)
            teacherLoaders.append(loader)

        print("\n**********************")
        print("Training teachers")
        # train teacher with subset seperately
        self.teachers.clear()
        for i in tqdm(range(self.numTeachers)):
            teacher = Teacher(i, net=self.net, dataset=self.dataset)
            teacher.train(teacherLoaders[i], optimizer=optim.SGD(teacher.model.parameters(),lr=kargs.get("lr")), criterion=criterion, device=device, epochs=epoch)
            if(save_inplace):
                dir = kargs.get('save_root')
                if not exists(dir): 
                    os.mkdir(dir)
                save_dir = pathlib.Path(dir).joinpath(f'{self.net}_{self.dataset}_teachers')
                if not save_dir.is_dir():
                    os.mkdir(save_dir)
                torch.save(teacher, f'{save_dir}/teacher_{i}.pt')
            else:
                self.teachers.append(teacher)
        # if modelDir != None:
        #     self.saveTeachers(modelDir)

    def predict(self, dataLoader: DataLoader, device,) -> torch.Tensor:
        print('Ensemble with {} teachers is predicting {} datas'.format(
            self.numTeachers, len(dataLoader.dataset)))
        preds = torch.zeros((self.numTeachers, len(
            dataLoader.dataset)), dtype=torch.long)
        for i, teacher in tqdm(enumerate(self.teachers)):
            results = teacher.predict(dataLoader, device=device)
            preds[i] = results  # nTeacher*nDataset

        return preds

    def get(self,index) -> Teacher:
        assert index >= 0 and index < len(self.teachers)
        return self.teachers[index]

    def test_teachers(self, dataloader, device):
        print("\nTesting teachers' accrucy.")
        top = 0.
        low = 1.
        avg = 0.
        for teacher in self.teachers:
            acc = teacher.test(dataloader, device)
            if acc > top:
                top = acc
            elif acc < low:
                low = acc
            avg += acc
        avg /= len(self.teachers)
        print(f"{len(self.teachers)} teachers test on {len(dataloader.dataset)} datas")
        print(f"Top accuracy is {top*100}%")
        print(f"Lowest accuracy is {low*100}%")
        print(f"Average accuracy is {avg*100}%")

    @staticmethod
    def savePreds(predsDir, preds, labels):
        if exists(predsDir) : shutil.rmtree(predsDir)
        os.mkdir(predsDir)
        np.save(file=predsDir+PREDS_NAME, arr=preds.numpy())
        np.save(file=predsDir+LABELS_NAME, arr=labels)
        print(f"Save preds to {predsDir}")

    def saveTeachers(self, dir: str):
        if not exists(dir): 
            os.mkdir(dir)
        save_dir = pathlib.Path(dir).joinpath(f'{self.net}_{self.dataset}_teachers')
        if not save_dir.is_dir():
            os.mkdir(save_dir)
        for i, teacher in enumerate(self.teachers):
            torch.save(teacher, f"{save_dir}/teacher_{i}.pt")
        print("Save teacher models to {}".format(save_dir))
    
    def loadTeachers(self, dir: str):
        print(f"Loading teacher models from {dir}")
        self.teachers.clear()
        fileStrs = os.listdir(dir)
        cnt = 0
        for file in fileStrs:
            if file.endswith(".pt") and file.startswith('teacher'):
                self.teachers.append(torch.load(dir+'/'+file))
                cnt += 1
        self.numTeachers = cnt
        print(f"Load {cnt} models successfully.")

