import sys
import os

pwd = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pwd+"/..") 

import pandas as pd
from db_models import DB_Csv,DB_Utility,db

DB_Utility.delete().execute()
# db.create_tables(DB_Utility)

ents = DB_Csv.select()

COLS_MAP = {
    'lp-2st':['idx','epoch','train_loss','train_acc','test_loss','test_acc','train_cost','test_cost'],
    'knn':['idx','epoch','train_loss','train_acc','test_loss','test_acc','time_cost'],
    'adp_alloc':['epoch','train_loss','train_acc','test_loss','test_acc','noise','total_eps','train_cost','test_cost'],
}

PATE_UDA_COL = ['epoch','label_loss','unlabel_loss','test_loss','test_acc']
KNN_UDA_COL = ['idx','epoch','label_loss','unlabel_loss','test_loss','test_acc','time_cost']
DEFAULT_COL = ['epoch','train_loss','train_acc','test_loss','test_acc','train_cost','test_cost']

value_list = []
for ent in ents:
    if(ent.extra == 'uda' and ent.func != None):
        if(ent.func == 'pate'):
            csv = pd.read_csv(ent.location,names=PATE_UDA_COL)
        elif(ent.func == 'knn'):
            csv = pd.read_csv(ent.location,names=KNN_UDA_COL)
        else:
            raise NotImplementedError(f'ent.func:{ent.func}')
        train_loss = csv['label_loss'].iat[-1]
        train_acc = 0
        test_loss = csv['test_loss'].iat[-1]
        test_acc = csv['test_acc'].iat[-1]
    else:
        csv = pd.read_csv(ent.location,names=COLS_MAP.get(ent.func,DEFAULT_COL))
        train_loss = csv['train_loss'].iat[-1]
        train_acc = csv['train_acc'].iat[-1]
        test_loss = csv['test_loss'].iat[-1]
        test_acc = csv['test_acc'].iat[-1]
    if(ent.func is None):
        func_name = model_type = 'shadow'
    else:
        func_name = ent.func
        model_type = 'target'
    
    value = (
        func_name,
        ent.net,
        ent.dataset,
        ent.eps,
        model_type,
        ent.extra,
        train_loss,
        train_acc,
        test_loss,
        test_acc
    )
    value_list.append(value)

with db.atomic():
    DB_Utility.insert_many(value_list, 
        fields=[
            DB_Utility.func,
            DB_Utility.net,
            DB_Utility.dataset,
            DB_Utility.eps,
            DB_Utility.type,
            DB_Utility.extra,
            DB_Utility.train_loss,
            DB_Utility.train_acc,
            DB_Utility.test_loss,
            DB_Utility.test_acc
        ]
    ).execute()
