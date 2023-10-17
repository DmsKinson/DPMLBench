FUNC_DB2TABLE = {
    'dpgen':'\\dpgen',
    'gswgan':'\\gswgan',
    'handcraft':'\\wavelet',
    'tanh':'\\sigmoid',
    'loss':'\\loss',
    'relu':'\\dpsgd',
    'adp_alloc':'\\alloc',
    'adpclip':'\\adpclip',
    'gep':'\\gep',
    'rgp':'\\rgp',
    'pate':'\\pate',
    'knn':'\\knn',
    'lp-2st':'\\lpmst',
    'alibi':'\\alibi'
}

FUNC_DB2FIGURE = {
    'dpgen':'DPGEN',
    'gswgan':'GS-WGAN',
    'handcraft':'Handcraft-DP',
    'tanh':'TanhActv',
    'loss':'FocalLoss',
    'relu':'DP-SGD',
    'adp_alloc':'AdpAlloc',
    'adpclip':'AdpClip',
    'gep':'GEP',
    'rgp':'RGP',
    'pate':'PATE',
    'knn':'Private-kNN',
    'lp-2st':'LP-MST',
    'alibi':'ALIBI'
}

NETS_MAP = {
    'simplenn':'SimpleCNN',
    'resnet':'ResNet',
    'inception':'InceptionNet',
    'vgg':'VGG'
}

DATASETS_MAP = {
    'mnist':'MNIST',
    'fmnist':'FMNIST',
    'svhn':'SVHN',
    'cifar10':'CIFAR-10'
}

FUNCS_CFG_MAP = {
    'adp_alloc':{
        'db_name':'adp_alloc',
        'color':'#1F77B4',
        'display_name':FUNC_DB2FIGURE['adp_alloc'],
        'table_name':FUNC_DB2TABLE['adp_alloc'],
        'marker':"o"
    },
    'adpclip':{
        'db_name':'adpclip',
        'color':'#F8C549',
        'display_name':FUNC_DB2FIGURE['adpclip'],
        'table_name':FUNC_DB2TABLE['adpclip'],
        'marker':"D"
    },
    'gep':{
        'db_name':'gep',
        'color':'#FF7F0E',
        'display_name':FUNC_DB2FIGURE['gep'],
        'table_name':FUNC_DB2TABLE['gep'],
        'marker':'v'
    },
    'handcraft':{
        'db_name':'handcraft',
        'color':'#2CA02C',
        'display_name':FUNC_DB2FIGURE['handcraft'],
        'table_name':FUNC_DB2TABLE['handcraft'],
        'marker':'^'
    },
    'loss':{
        'db_name':'loss',
        'color':'#D62728',
        'display_name':FUNC_DB2FIGURE['loss'],
        'table_name':FUNC_DB2TABLE['loss'],
        'marker':'<'
    },
    'relu':{
        'db_name':'relu',
        'color':'#9467BD',
        'display_name':FUNC_DB2FIGURE['relu'],
        'table_name':FUNC_DB2TABLE['relu'],
        'marker':'>'
    },
    'tanh':{
        'db_name':'tanh',
        'color':'#8C564B',
        'display_name':FUNC_DB2FIGURE['tanh'],
        'table_name':FUNC_DB2TABLE['tanh'],
        'marker':'8'
    },
    'rgp':{
        'db_name':'rgp',
        'color':'#E377C2',
        'display_name':FUNC_DB2FIGURE['rgp'],
        'table_name':FUNC_DB2TABLE['rgp'],
        'marker':'s'
    },
    'knn':{
        'db_name':'knn',
        'color':'#72D884',
        'display_name':FUNC_DB2FIGURE['knn'],
        'table_name':FUNC_DB2TABLE['knn'],
        'marker':'+',
        'extra':'uda',
    },
    'pate':{
        'db_name':'pate',
        'color':'#7F7F7F',
        'display_name':FUNC_DB2FIGURE['pate'],
        'table_name':FUNC_DB2TABLE['pate'],
        'marker':'p',
        'extra':'uda',
    },
    'lp-2st':{
        'db_name':'lp-2st',
        'color':'#BCBD22',
        'display_name':FUNC_DB2FIGURE['lp-2st'],
        'table_name':FUNC_DB2TABLE['lp-2st'],
        'marker':'P'
    },
    'alibi':{
        'db_name':'alibi',
        'color':'#17BECF',
        'display_name':FUNC_DB2FIGURE['alibi'],
        'table_name':FUNC_DB2TABLE['alibi'],
        'marker':'*'
    },
    'dpgen':{
        'db_name':'dpgen',
        'color':'#F8C549',
        'display_name':FUNC_DB2FIGURE['dpgen'],
        'table_name':FUNC_DB2TABLE['dpgen'],
        'marker':'X'
    },
    'gswgan':{
        'db_name':'gswgan',
        'color':'#97057F',
        'display_name':FUNC_DB2FIGURE['gswgan'],
        'table_name':FUNC_DB2TABLE['gswgan'],
        'marker':'s'
    }
}
