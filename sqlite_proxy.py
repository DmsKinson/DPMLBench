import logging
import os
from datetime import datetime

pwd = os.path.split(os.path.realpath(__file__))[0]

import logging
from db_models import DB_Model, DB_Csv, DB_Func, DB_Attack

sql_logger = logging.getLogger('sqlite_proxy')

TYPE_TARGET = 'target'
TYPE_SHADOW = 'shadow'
TYPE_ATTACK = 'attack'

def _upsert_model(func:str=None,
    net:str=None,
    dataset:str=None, 
    eps:float=None, 
    other_param:dict=None, 
    model_loc:str=None, 
    model_type:str=TYPE_SHADOW, 
    extra:str = None
):
    new_model, _ = DB_Model.get_or_create(
        func = func,
        net = net,
        dataset = dataset,
        eps = eps,
        type = model_type,
        extra = extra
    )
    new_model.location = model_loc
    new_model.other_param = other_param
    new_model.date = datetime.now()
    new_model.save()
    id = new_model.get_id()
    sql_logger.info(f'upsert model with id:{id}')
    return id

def _upsert_csv(
    func:str=None, 
    net:str=None, 
    dataset:str=None, 
    eps:float=None, 
    csv_loc:str=None, 
    extra:str = None
):
    new_csv, _ = DB_Csv.get_or_create(
        func = func,
        net = net,
        dataset = dataset,
        eps = eps,
        extra = extra
    )
    new_csv.location = csv_loc
    new_csv.date = datetime.now()
    new_csv.save() 
    id = new_csv.get_id()
    sql_logger.info(f'upsert csv with id:{id}')
    return id
    
def _upsert_mia(
    attack_type:str='black:', 
    func:str=None, 
    net:str=None, 
    dataset:str=None, 
    eps:float=None, 
    auc:float=0,
    prob_loc:str=None, 
    shadow_dp:bool=False,
    extra:str = None
):
    new_mia, _ = DB_Attack.get_or_create(
        func = func,
        net = net,
        dataset = dataset,
        eps = eps,
        type = attack_type,
        shadow_dp = shadow_dp,
        extra = extra
    )
    new_mia.location = prob_loc
    new_mia.date = datetime.now()
    new_mia.auc = auc
    new_mia.save() 
    attack_id = new_mia.get_id()
    sql_logger.info(f'upsert mia with id:{attack_id}')
    return attack_id

def _insert_shadow_model(
    net:str=None,
    dataset:str=None,
    eps:float=None,
    other_param:dict=None, 
    exp_loc:str=None,
    model_loc:str=None,
    model_type = TYPE_SHADOW,
    extra:str = None
):
    arg_bundle = locals()
    _upsert_csv(func=None,net=net, dataset=dataset,eps=eps,csv_loc=exp_loc,extra=extra)
    _upsert_model(func=None, net=net, dataset=dataset, eps=eps,other_param=other_param,model_loc=model_loc,model_type=model_type,extra=extra)
    sql_logger.info(f'finish insert shadow net')
    return arg_bundle

def _insert_target_model(
    func:str = None, 
    net:str=None, 
    dataset:str=None, 
    eps:float=None, 
    other_param:dict=None, 
    exp_loc:str=None, 
    model_loc:str=None, 
    model_type = TYPE_TARGET,
    extra:str = None
):
    arg_bundle = locals()
    func_ent, _ = DB_Func.get_or_create(
        name = func,
        net = net, 
        dataset = dataset,
        eps = eps,
    )
    csv_id = _upsert_csv(
        func=func,
        net=net,
        dataset=dataset,
        eps=eps,
        csv_loc=exp_loc,
        extra=extra
    )
    model_id = _upsert_model(
        func=func,
        net=net,
        dataset=dataset,
        eps=eps,
        other_param=other_param,
        model_loc=model_loc,
        model_type=model_type,
        extra=extra
    )
    func_ent.model_id = model_id
    func_ent.csv_id = csv_id
    func_ent.save()
    return arg_bundle

def _insert_attack_model(
    func:str=None,
    net:str=None, 
    eps:float=None,
    dataset:str=None, 
    other_param:dict=None,
    model_loc:str=None, 
    model_type = TYPE_ATTACK,
    extra:str = None
):
    arg_bundel = locals()
    _upsert_model(func=func, net=net, dataset=dataset, eps=eps,other_param=other_param,model_loc=model_loc,model_type=model_type,extra=extra)
    sql_logger.info(f'finish insert attack net')
    return arg_bundel

def insert_net(
    func:str = None, 
    net:str=None, 
    dataset:str=None, 
    eps:float=None, 
    other_param:dict=None, 
    exp_loc:str=None, 
    model_loc:str=None, 
    model_type:str=TYPE_TARGET, 
    extra:str = None
):
    sql_logger.debug(f'insert {locals()}')
    if(model_type == TYPE_TARGET):
        return _insert_target_model(
            func=func, 
            net=net, 
            dataset=dataset, 
            eps=eps, 
            other_param=other_param, 
            exp_loc=exp_loc,
            model_loc=model_loc,
            extra=extra
        )
    elif(model_type == TYPE_SHADOW):
        return _insert_shadow_model(
            net=net,
            dataset=dataset,
            eps=eps,
            other_param=other_param,
            exp_loc=exp_loc,
            model_loc=model_loc,
            extra=extra
        )
    elif(model_type == TYPE_ATTACK):
        return _insert_attack_model(
            func=func,
            net=net,
            eps=eps,
            dataset=dataset,
            other_param=other_param,
            model_loc=model_loc,
            extra=extra
        )

def insert_mia(
        type:str='black',
        func:str=None,
        net:str=None,
        dataset:str=None,
        eps:float=None,
        auc:float=0,
        prob_loc:str=None,
        shadow_dp=False,
        extra=None
    ):
    param = locals()
    attack_id = _upsert_mia(attack_type=type, func=func, net=net, dataset=dataset, eps=eps, auc=auc, prob_loc=prob_loc, shadow_dp=shadow_dp,extra=extra)
    func_ent = DB_Func.get_or_none(
        DB_Func.name == func,
        DB_Func.net == net,
        DB_Func.dataset == dataset,
        DB_Func.eps == eps, 
    )
    if(func_ent == None):
        sql_logger.error(f'cannot find model:func={func},net={net},dataset={dataset},eps={eps}')
        return
    if(type == 'black'):
        func_ent.black_id = attack_id
    elif(type == 'white'):
        func_ent.white_id = attack_id
    func_ent.save()
    return param

def query_mia(type:str='black:', func:str=None, net:str=None, dataset:str=None, eps:float=None,):
    func_ent = DB_Func.get_or_none(
        DB_Func.func == func,
        DB_Func.net == net,
        DB_Func.dataset == dataset,
        DB_Func.eps == eps, 
    )
    if(func_ent == None):
        sql_logger.error(f'cannot find model:func={func},net={net},dataset={dataset},eps={eps}')
        return
    if(type == 'white'):
        attack_id = func_ent.white_id
    elif(type == 'black'):
        attack_id = func_ent.black_id

    attack_ent = DB_Attack.get_by_id(attack_id)
    return attack_ent

def query_csv(*query):
    if(len(query) == 0):
        list = DB_Csv.select()
    else:
        list = DB_Csv.select().where(*query)
    return list