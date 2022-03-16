from data_prepare import *
from data import *
from model import *
from training import *
from evaluate import *

import pandas as pd
from tqdm import tqdm

def get_setup():
    setup_file = None
    with open('./setup/setup.txt','r') as file:
        setup_file = file.read().split('\n')
    try:    
        setup_file = {line.split(':')[0] : line.split(':')[1] for line in setup_file}
        setup_file['Classes'] = [classname for classname in setup_file['Classes'].split(',')]
        setup_file['FromLatest'] = bool(setup_file['FromLatest'])
        setup_file['Epoch'] = int(setup_file['FromLatest']) if setup_file['FromLatest'].isnumeric() else setup_file['FromLatest']
        setup_file['MaxEpoch'] = int(setup_file['MaxEpoch'])
        return setup_file
    except Exception as e:
        print(e)
        raise FileNotFoundError('Unable to read setup file.')

def prepare_data():
    return []

def prepare_models(setup):
    policy_model = SaccadicNetEye()
    classifier = SaccadicNetClassifier()
    epoch = 0
    if setup['FromLatest']:
        epoch = max(list(pd.read_csv('./weights/index.csv')['Epoch']))
        policy_model.load_latest()
        classifier.load_latest()
    elif type(setup['Epoch']) == int:
        policy_model.load_intermediate(setup['Epoch'])
        classifier.load_intermediate(setup['Epoch'])
        epoch = setup['Epoch']
    return policy_model, classifier, epoch

def validate():
    pass

def evaluate():
    pass

def train_epoch(sn_eye,sn_classifier,dataset,epoch):
    do_epoch(sn_eye,sn_classifier,dataset,epoch)
    sn_eye.store_intermediate(epoch)
    sn_classifier.store_intermediate(epoch)
#     evaluate()


if __name__ == '__main__':
    setup = get_setup()
#     dataset = prepare_data()
    saccadic_net_eye, saccadic_net_classifier, epoch = prepare_models(setup)
#     validate()
    num_epochs = setup['MaxEpoch']
    if epoch < num_epochs:
        for i in range(len(num_epochs)-epoch):
            print('Beginning training epoch' + str(i+epoch)+'.')
            train_epoch(saccadic_net_eye, saccadic_net_classifier,dataset,i+epoch)