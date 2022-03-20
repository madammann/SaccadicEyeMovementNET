import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from model import *

def update_weight_index(model,epoch):
    try:
        df = pd.read_csv('./weights/index.csv')
        df.append(model.name,epoch,str(datetime.now()),columns=['Model','Epoch','Datetime'])
        df.to_csv('./weights/index.csv',index=False)
    except Exception as e:
        print(e)
        print('Could not write model weight to index file.')

def update_performance(model,loss,accuracy,epoch):
    try:
        df = pd.read_csv('./eval/loss_accuracy.csv')
        df.append(model.name,epoch,loss,accuracy,str(datetime.now()),columns=['Model','Epoch','Loss','Accuracy','Datetime'])
        df.to_csv('./eval/loss_accuracy.csv',index=False)
    except Exception as e:
        print(e)
        print('Could not write model loss and accuracy to file.')

def classifier_performance_eval(sn_classifier, data_split):
    loss, accuracy = [],[]
    for (input_batch, target_batch) in tqdm(data_split,desc='Evaluating accuracy of classifier on train and test data.'):
        prediction = sn_classifier(input_batch)
        loss += sn_classifier.loss(target_batch, prediction).numpy()
        accuracy +=  np.mean(np.argmax(target, axis=1) == np.argmax(prediction, axis=1))
    loss = tf.reduce_mean(loss_aggregator)
    accuracy = tf.reduce_mean(accuracy_aggregator)
    return loss, accuracy

def do_epoch(sn_eye,sn_classifier,dataset,epoch):
    if epoch == 0:
        '''Before epoch, get the performance of the classifier'''
        for data, name in zip(list(dataset.values()),list(dataset.keys())):
            loss, accuracy = classifier_performance_eval(sn_classifier, data)
            update_performance(sn_classifier,loss,accuracy,0)
    '''Training loop'''
    for input_batch, target_batch in tqdm(dataset['train'],desc='Training epoch '+str(epoch+1)):
        with tf.GradientTape() as eye_tape, tf.GradientTape() as class_tape:
            '''Prediction/Focus point output of models'''
            class_prediction = sn_classifier(input_batch)
            eye_focus = sn_eye(input_batch)
            
            '''Calculating the loss'''
            eye_loss = sn_eye.loss()
            class_loss = sn_classifier.loss(target_batch, class_prediction)
            
            '''Applying gradients'''
            eye_gradient = eye_tape.gradient(eye_loss, sn_eye.trainable_variables)
            sn_eye.optimizer.apply_gradients(zip(eye_gradient, sn_eye.trainable_variables))
            class_gradient = class_tape.gradient(class_loss, sn_classifier.trainable_variables)
            sn_classifier.optimizer.apply_gradients(zip(class_tape, sn_classifier.trainable_variables))
    
    '''Storing model weights'''
    sn_eye.store_intermediate(epoch)
    update_weight_index(sn_eye,epoch+1)
    sn_classifier.store_intermediate(epoch)
    update_weight_index(sn_classifier,epoch+1)
    
    '''After epoch classifier evaluation'''
    for data, name in zip(list(dataset.values()),list(dataset.keys())):
        loss, accuracy = classifier_performance_eval(sn_classifier, data)
        update_performance(sn_classifier,loss,accuracy,epoch+1)