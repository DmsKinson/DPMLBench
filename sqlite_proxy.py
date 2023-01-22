import logging
import os
from datetime import datetime

pwd = os.path.split(os.path.realpath(__file__))[0]

from config import HOST_IP
import config
import zerorpc
import logging
from db_models import DB_Model, DB_Csv, DB_Func, DB_Attack

sql_logger = logging.getLogger('sqlite_proxy')

TYPE_TARGET = 'target'
TYPE_SHADOW = 'shadow'
TYPE_ATTACK = 'attack'

def rpc_insert_net(params:dict):
    c = zerorpc.Client()
    try:
        c.connect(f'tcp://{config.REMOTE_IP}:{config.REMOTE_RPC_PORT}')
        c.insert_net(params)
    except Exception as e:
        sql_logger.error(e)
    finally:
        c.close()
        
def rpc_insert_mia(params:dict):
    c = zerorpc.Client()
    try:
        c.connect(f'tcp://{config.REMOTE_IP}:{config.REMOTE_RPC_PORT}')
        c.insert_mia(params)
    except Exception as e:
        sql_logger.error(e)
    finally:
        c.close()

def _upsert_model(func:str=None,
    net:str=None,
    dataset:str=None, 
    eps:float=None, 
    other_param:dict=None, 
    model_loc:str=None, 
    model_checksum:str=None, 
    model_type:str=TYPE_SHADOW, 
    host_ip:str=HOST_IP,
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
    new_model.checksum = model_checksum
    new_model.other_param = other_param
    new_model.date = datetime.now()
    new_model.host_ip = host_ip
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
    csv_checksum:str=None, 
    host_ip:str=HOST_IP,
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
    new_csv.checksum = csv_checksum
    new_csv.date = datetime.now()
    new_csv.host_ip = host_ip
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
    prob_checksum:str=None, 
    shadow_dp:bool=False,
    host_ip:str=HOST_IP,
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
    new_mia.checksum = prob_checksum
    new_mia.date = datetime.now()
    new_mia.host_ip = host_ip
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
    exp_checksum:str=None,
    model_loc:str=None,
    model_checksum:str=None,
    model_type = TYPE_SHADOW,
    host_ip=HOST_IP,
    extra:str = None
):
    arg_bundle = locals()
    _upsert_csv(func=None,net=net, dataset=dataset,eps=eps,csv_loc=exp_loc,csv_checksum=exp_checksum,host_ip=host_ip,extra=extra)
    _upsert_model(func=None, net=net, dataset=dataset, eps=eps,other_param=other_param,model_loc=model_loc,model_checksum=model_checksum,model_type=model_type,host_ip=host_ip,extra=extra)
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
    exp_checksum:str=None, 
    model_checksum:str=None,
    model_type = TYPE_TARGET,
    host_ip = HOST_IP,
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
        csv_checksum=exp_checksum,
        host_ip=host_ip,
        extra=extra
    )
    model_id = _upsert_model(
        func=func,
        net=net,
        dataset=dataset,
        eps=eps,
        other_param=other_param,
        model_loc=model_loc,
        model_checksum=model_checksum,
        model_type=model_type,
        host_ip=host_ip,
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
    model_checksum:str=None,
    model_type = TYPE_ATTACK,
    host_ip = HOST_IP,
    extra:str = None
):
    arg_bundel = locals()
    _upsert_model(func=func, net=net, dataset=dataset, eps=eps,other_param=other_param,model_loc=model_loc,model_checksum=model_checksum,model_type=model_type,host_ip=host_ip,extra=extra)
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
    exp_checksum:str=None, 
    model_checksum:str=None,
    host_ip = HOST_IP,
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
            exp_checksum=exp_checksum,
            model_loc=model_loc,
            model_checksum=model_checksum,
            host_ip=host_ip,
            extra=extra
        )
    elif(model_type == TYPE_SHADOW):
        return _insert_shadow_model(
            net=net,
            dataset=dataset,
            eps=eps,
            other_param=other_param,
            exp_loc=exp_loc,
            exp_checksum=exp_checksum,
            model_loc=model_loc,
            model_checksum=model_checksum,
            host_ip=host_ip,
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
            model_checksum=model_checksum,
            host_ip=host_ip,
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
        prob_checksum:str=None,
        host_ip=HOST_IP,
        shadow_dp=False,
        extra=None
    ):
    param = locals()
    attack_id = _upsert_mia(attack_type=type, func=func, net=net, dataset=dataset, eps=eps, auc=auc, prob_loc=prob_loc, prob_checksum=prob_checksum,host_ip=host_ip, shadow_dp=shadow_dp,extra=extra)
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
    elif(type == 'label'):
        func_ent.label_id = attack_id
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
    elif(type == 'label'):
        attack_id = func_ent.label_id

    attack_ent = DB_Attack.get_by_id(attack_id)
    return attack_ent

def query_csv(*query):
    if(len(query) == 0):
        list = DB_Csv.select()
    else:
        list = DB_Csv.select().where(*query)
    return list