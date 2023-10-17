from peewee import *
import os
import yaml

pwd = os.path.split(os.path.realpath(__file__))[0]

DATABASE_DIR = os.path.join(pwd, 'database')
if(not os.path.isdir(DATABASE_DIR)):
    os.makedirs(DATABASE_DIR, exist_ok=True)
MAIN_DB_PATH = os.path.join(DATABASE_DIR, 'main.db')
MERGE_DB_PATH = os.path.join(DATABASE_DIR, 'merge.db')

main_db = SqliteDatabase(MAIN_DB_PATH)
merge_db = SqliteDatabase(MERGE_DB_PATH)

class DB_AccStat(Model):
    func = CharField()
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)
    
    mean = FloatField(null=True)
    std = FloatField(null=True)
    class Meta:
        table_name = 'AccStat'
        database = merge_db

class DB_PrivacyStat(Model):
    func = CharField()
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)
    type = CharField()
    
    auc_mean = FloatField(null=True)
    auc_std = FloatField(null=True)
    prop_mean = FloatField(null=True)
    prop_std = FloatField(null=True)
    class Meta:
        table_name = 'PrivacyStat'
        database = merge_db



class YamlField(CharField):
    def db_value(self, value):
        if(value != None):    
            value = yaml.safe_dump(value)
        return value 

    def python_value(self, value):
        if(value != None):
            value = yaml.safe_load(value)
        return value 

class BaseModel(Model):
    class Meta:
        database = main_db

class Property(BaseModel):
    location = CharField(null=True)
    date = DateTimeField(null=True)
    extra = CharField(null=True)

class DB_Func(BaseModel):
    name = CharField()
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)

    model_id = IntegerField(null=True)
    csv_id = IntegerField(null=True)
    black_id = IntegerField(null=True)
    white_id = IntegerField(null=True)
    label_id = IntegerField(null=True)
    class Meta:
        table_name = 'Func'

class DB_Csv(Property):
    func = CharField(null=True)
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)
    class Meta:
        table_name = 'Csv'

class DB_Model(Property):
    func = CharField(null=True)
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)
    type = CharField()
    other_param = YamlField(null=True)
    class Meta:
        table_name = 'Model'

class DB_Attack(Property):
    func = CharField()
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)
    type = CharField()
    auc = FloatField(null=True)
    shadow_dp = BooleanField(default=False)
    class Meta:
        table_name = 'Attack'

class DB_Utility(BaseModel):
    func = CharField()
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)
    type = CharField()
    extra = CharField(null=True,default=None)
    
    train_acc = FloatField(null=True)
    train_loss = FloatField(null=True)
    test_acc = FloatField(null=True)
    test_loss = FloatField(null=True)

    class Meta:
        table_name = 'Utility'

class DB_Privacy(BaseModel):
    func = CharField()
    net = CharField()
    dataset = CharField()
    eps = FloatField(null=True)
    type = CharField()
    shadow_dp = BooleanField(default=False)
    extra = CharField(null=True,default=None)
    
    auc = FloatField(null=True)
    ma = FloatField(null=True)
    precision = FloatField(null=True)
    recall = FloatField(null=True)
    f1 = FloatField(null=True)
    asr = FloatField(null=True)

    class Meta:
        table_name = 'Privacy'

merge_db.create_tables([
    DB_AccStat,
    DB_PrivacyStat,
])

main_db.create_tables([
    DB_Func,
    DB_Model,
    DB_Csv,
    DB_Attack,
    DB_Utility,
    DB_Privacy,
    ])