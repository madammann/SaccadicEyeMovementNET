import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from multiprocessing.pool import ThreadPool
from model import *

from scipy import stats

'''CSV File Writer Functions'''
def update_weight_index(model,epoch):
    try:
        df = pd.read_csv('./weights/index.csv')
        df.append(model.name,epoch,str(datetime.now()),columns=['Model','Epoch','Datetime'])
        df.to_csv('./weights/index.csv',index=False)
    except Exception as e:
        print(e)
        print('Could not write model weight to index file.')

def update_performance(model,loss,accuracy,reward,epoch):
    try:
        df = pd.read_csv('./eval/loss_accuracy.csv')
        df.append(model.name,epoch,loss,accuracy,reward,str(datetime.now()),columns=['Model','Epoch','Loss','Accuracy','Reward','Datetime'])
        df.to_csv('./eval/loss_accuracy.csv',index=False)
    except Exception as e:
        print(e)
        print('Could not write model loss and accuracy to file.')

'''Training and evaluation loops'''
def train_epoch(sn_eye,sn_classifier,dataset,epoch,batch_size=64,sequence_len=20,gamma=0.98):
    '''Function description'''
    
    '''Generate list of losses and accuracy measures over the entire epoch'''
    epoch_eye_loss,epoch_classifier_loss,epoch_accuracy = [], [], []
    
    '''Iterate over the dataset'''
    for data, target in tqdm(dataset,desc='Training epoch '):
        '''Generate first sequence element'''
        env = Environment(data,batch_size=batch_size,seq_len=sequence_len)
        
        '''Generate loss, accuracy and reward measures for batch sequence'''
        with tf.GradientTape(persistent=True) as tape:
            '''Generate necessary lists and variables for tracing'''
            classifier_loss,accuracy,eye_loss = [],[],[]
            
            '''Generate necessary variables for stroing the initial LSTM states'''
            class_hidden_h,class_hidden_c,eye_hidden_h,eye_hidden_c = None,None,None,None
            
            '''Until the environment reaches terminal state (sequence length) do:'''
            for i in range(sequence_len):
                '''Feed the current sequence through the network'''
                distributions,eye_hidden_h,eye_hidden_c = sn_eye(env.sequence[-1],initial_state=[eye_hidden_h,eye_hidden_c])
                prediction,class_hidden_h,class_hidden_c = sn_classifier(env.sequence[-1],initial_state=[class_hidden_h,class_hidden_c])
                
                '''Use the sn_eye output to generate the next sequence element if the sequence length is not yet reached'''
                if not env.terminal():
                    env.action(distributions)
                
                '''Classifier loss and accuracy calculations'''
                accuracy += [tf.reduce_mean(tf.cast(tf.argmax(prediction) == tf.argmax(target),'float32'))]
                classifier_loss += [sn_classifier.loss(target, prediction)]
            
            '''Create the eye loss from the classifier loss before reducing to mean'''
            policy_gradient = PolicyGradient(classifier_loss[1:])
            eye_loss = policy_gradient.get_eye_loss(env.get_states())
            
            # parameter_grad_zipper = [(None, param) for param in sn_eye.trainable_variables[:-4]]
            # parameter_grad_zipper += [(policy_gradient.generate_gradient(param), param) for param in sn_eye.trainable_variables[-4:]]
            # print(parameter_grad_zipper[-5:])
            
            '''Taking the mean'''
            eye_loss = tf.reduce_mean(eye_loss)
            classifier_loss = tf.reduce_mean(classifier_loss)
            accuracy = tf.reduce_mean(accuracy)
            
            '''Appending results to epoch metric'''
            epoch_classifier_loss += [classifier_loss]
            epoch_accuracy += [accuracy]
            epoch_eye_loss += [eye_loss]
            
        '''Applying gradients'''
        class_gradient = tape.gradient(classifier_loss, sn_classifier.trainable_variables)
        sn_classifier.optimizer.apply_gradients(zip(class_gradient, sn_classifier.trainable_variables))
            
        '''Calculating and applying gradients with Policy Class'''
        # eye_gradient = tape.gradient(eye_loss, sn_eye.trainable_variables)
        # sn_eye.optimizer.apply_gradients(parameter_grad_zipper)
    
    '''Taking the mean'''
    epoch_accuracy = tf.reduce_mean(epoch_accuracy).numpy()
    epoch_classifier_loss = tf.reduce_mean(epoch_classifier_loss).numpy()
    epoch_eye_loss = tf.reduce_mean(epoch_eye_loss).numpy()
    '''Updating the performance csv files'''
    # update_performance(sn_eye.nameinfo,epoch_eye_loss,None,epoch_reward,epoch)
    # update_performance(sn_classifier.nameinfo,epoch_classifier_loss,epoch_accuracy,None,epoch)
    
    '''Storing model weights for future use'''
    # update_weight_index(sn_eye.nameinfo,epoch)
    # update_weight_index(sn_classifier.nameinfo,epoch)
    
    return epoch_classifier_loss,epoch_eye_loss,epoch_reward,epoch_accuracy
            
def evaluation_loop():
    pass