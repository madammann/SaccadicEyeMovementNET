from data import *
from model import *
from training import *
from evaluate import *

import pandas as pd
from tqdm import tqdm

import os

def get_setup():
    '''Loads in and returns the dictionary given in the setup file'''
    setup_file = None
    try:
        '''Reading the file lines'''
        with open('./setup/setup.txt','r') as file:
            setup_file = file.read().split('\n')
        
        '''Creating a dictionary for each key'''
        setup_file = {line.split('=')[0] : line.split('=')[1] for line in setup_file if line != ""}
        
        '''Converting certain keys into more appropriate data type'''
        setup_file['Classes'] = [classname for classname in setup_file['Classes'].split(',')]
        setup_file['MaxEpoch'] = int(setup_file['MaxEpoch'])
        setup_file['GPU'] = bool(setup_file['FromLatest'])
        
        return setup_file
    except Exception as e:
        '''Raise an error if something went wrong'''
        print(e)
        raise FileNotFoundError('Unable to read or load in the setup file.')

def prepare_data(setup):
    '''Work in progress'''
    # TODO: Add a condition to start the download
    # if already_downloaded():
        # print('Preparing to download the COCO dataset to be stored in ' + setup['DataPath'])
        # download(setup['Classes'], setup['DataPath'])
        
    # TODO: Switch from duplicate removal to only downloading images uniquely
    print('Scanning for duplicate images.')
    search_and_delete_duplicates(setup['DataPath']+"/coco_dataset_subclass")
    
    '''Loading and constructing the dataset pipelin'''
    print('Preparing dataset pipeline.')
    dataset = load_manual_alternative(setup['DataPath']) 
    dataset['train'] = preprocess_data(dataset['train'], len(setup['Classes']))
    dataset['test'] = preprocess_data(dataset['test'], len(setup['Classes']))
    return dataset

def prepare_models(setup):
    '''Loads in the model class objects with the selected setup.'''
    
    sn_eye = SaccadicNetEye()
    sn_classifier = SaccadicNetClassifier()
    
    '''If statements to check selected setup option'''
    if setup['Mode'] == 'Training':
        '''In training mode we start with the latest model available'''
        epoch = []
        epoch += [sn_eye.load_latest()]
        epoch += [sn_classifier.load_latest()]
        
        '''Raise an error if the two latest available epoch weights are not from the same epoch'''
        if epoch[0] != epoch[1]:
            raise FileNotFoundError('Latest available epochs differ between used models.')
        
        return sn_eye, sn_classifier, epoch[0]
    
    elif setup['Mode'] == 'Evaluation':
        '''In evaluation mode we start with the initialized model again'''
        return sn_eye, sn_classifier, 0
    else:
        '''Raises an error if the selected mode option does not exist'''
        raise KeyError('The dictionary based on the setup.txt file does not support the value '+str(setup['Mode'])+' for Mode.')

if __name__ == '__main__':
    setup = get_setup()
    dataset = prepare_data(setup)
    saccadic_net_eye, saccadic_net_classifier, epoch = prepare_models(setup)

    if setup['Mode'] == 'Training':
        num_epochs = setup['MaxEpoch']
        if epoch < num_epochs:
            for i in range(num_epochs-epoch):
                print('Beginning training epoch' + str(i+epoch)+'.')
                if setup['GPU']:
                    with tf.device("gpu"):
                        train_epoch(saccadic_net_eye, saccadic_net_classifier,dataset,i+epoch)
                else:
                    train_epoch(saccadic_net_eye, saccadic_net_classifier,dataset,i+epoch)
    elif setup['Mode'] == 'Evaluation':
        pass
    elif setup['Mode'] == '':
        pass
