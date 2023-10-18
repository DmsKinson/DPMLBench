import sys
import os

import torch
from tqdm import tqdm

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

from db_models import DB_Attack, DB_Privacy
from name_map import *
from sklearn import metrics
import numpy as np
from peewee import *

DATABASE_PATH = '/data2/zmh/workplace/paper-impl/DP-ML/database/main.db'
main_db = SqliteDatabase(DATABASE_PATH)

class DB_Attack(DB_Attack):
    class Meta:
        database = main_db
        table_name = 'Attack'

class DB_Privacy(DB_Privacy):
    class Meta:
        database = main_db
        table_name = 'Privacy'

types = ['black','white']

DB_Privacy.delete().execute()

ents = DB_Attack.select().where(
    DB_Attack.type << types,
)

value_list = []
for ent in tqdm(ents):
    loc = ent.location
    bundles = torch.load(loc)
    y_score, y_true = bundles[:2]
    if(np.isnan(y_score).all()):
        auc = 0.5
        ma = 0
        precision = 0
        recall = 0
        f1 = 0
        asr = 0
    else:
        y_pred = (y_score>0.5).long()
        fpr, tpr, thresholds = metrics.roc_curve(y_score=y_score, y_true=y_true)
        auc = metrics.auc(fpr,tpr)
        
        tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pred,y_true=y_true).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        recall = tpr
        ma = recall - fpr
        precision = tp/(tp+fp+1e-6)
        f1 = 2*tp/(2*tp+fp+fn)
        asr = (tp+tn)/len(y_pred)
    value = (
        ent.func,
        ent.net,
        ent.dataset,
        ent.eps,
        ent.type,
        ent.shadow_dp,
        ent.extra,

        auc,
        ma,
        precision,
        recall,
        f1,
        asr
        
    )
    value_list.append(value)

print(len(value_list),'records in total.')


with main_db.atomic():
    DB_Privacy.insert_many(value_list, 
        fields=[
            DB_Privacy.func,
            DB_Privacy.net,
            DB_Privacy.dataset,
            DB_Privacy.eps,
            DB_Privacy.type,
            DB_Privacy.shadow_dp,
            DB_Privacy.extra,

            DB_Privacy.auc,
            DB_Privacy.ma,
            DB_Privacy.precision,
            DB_Privacy.recall,
            DB_Privacy.f1,
            DB_Privacy.asr
        ]
    ).execute()
