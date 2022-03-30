from data import *
from model import *
from training import *
from evaluate import *

import pandas as pd
from tqdm import tqdm

import os

def get_setup():
    setup_file = None
    with open('./setup/setup.txt','r') as file:
        setup_file = file.read().split('\n')
    try:
        setup_file = {line.split('=')[0] : line.split('=')[1] for line in setup_file if line != ""}
        setup_file['Classes'] = [classname for classname in setup_file['Classes'].split(',')]
        setup_file['FromLatest'] = bool(setup_file['FromLatest'])
        setup_file['Epoch'] = int(setup_file['FromLatest']) if str(setup_file['FromLatest']).isnumeric() else setup_file['FromLatest']
        setup_file['MaxEpoch'] = int(setup_file['MaxEpoch'])
        setup_file['GPU'] = bool(setup_file['FromLatest'])
        setup_file['DataPath'] = setup_file['DataPath']
        setup_file['Multiprocessing'] = bool(setup_file['Multiprocessing'])
        setup_file['Modus'] = setup_file['Modus']
        return setup_file
    except Exception as e:
        print(e)
        raise FileNotFoundError('Unable to read setup file.')

def prepare_data(setup):
    # download(setup['Classes'], setup['DataPath'],multiprocessing=setup['Multiprocessing'])
    print('Preparing dataset pipeline.')
    dataset = load_manual_alternative(setup['DataPath'])
    dataset['train'] = preprocess_data(dataset['train'], len(setup['Classes'])).take(20)
    dataset['test'] = preprocess_data(dataset['test'], len(setup['Classes'])).take(20)
    return dataset

def prepare_models(setup):
    policy_model = SaccadicNetEye()
    classifier = SaccadicNetClassifier()
    epoch = 0
    if setup['FromLatest'] and len(os.listdir('./weights/'))>2:
        epoch = max(list(pd.read_csv('./weights/index.csv')['Epoch']))
        policy_model.load_latest()
        classifier.load_latest()
    elif type(setup['Epoch']) == int and len(os.listdir('./weights/'))>2:
        policy_model.load_intermediate(setup['Epoch'])
        classifier.load_intermediate(setup['Epoch'])
        epoch = setup['Epoch']
    return policy_model, classifier, epoch

if __name__ == '__main__':
    setup = get_setup()
    dataset = prepare_data(setup)
    saccadic_net_eye, saccadic_net_classifier, epoch = prepare_models(setup)

    if setup['Modus'] == 'Training':
        num_epochs = setup['MaxEpoch']
        if epoch < num_epochs:
            for i in range(num_epochs-epoch):
                print('Beginning training epoch' + str(i+epoch)+'.')
                if setup['GPU']:
                    with tf.device("gpu"):
                        train_epoch(saccadic_net_eye, saccadic_net_classifier,dataset,i+epoch)
                else:
                    train_epoch(saccadic_net_eye, saccadic_net_classifier,dataset,i+epoch)
    elif setup['Modus'] == 'Evaluation':
        pass
    elif setup['Modus'] == '':
        pass
