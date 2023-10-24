from peewee import *
import os

pwd = os.path.split(os.path.realpath(__file__))[0]

MERGE_DB_PATH = os.path.join('origin_data', 'merge.db')
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

merge_db.create_tables([
    DB_AccStat,
    DB_PrivacyStat,
])
