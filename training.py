import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from multiprocessing.pool import ThreadPool
from model import *

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

'''Patch Series generator and reward functions'''
def get_image_patch(image, coordinates=None):
        size = image.shape[:2]
        center = coordinates
        if type(center) == type(None):
            center = [int(size[0]/2),int(size[1]/2)]
        else:
            center = np.array(coordinates)
        if center[0] < 16:
            center[0] = 16
        elif center[0] > size[0] - 15:
            center[0] = size[0]-15
        if center[1] < 16:
            center[1] = 16
        elif center[1] > size[1] - 15:
            center[1] = size[1]-15
        return image[center[0]-16:center[0]+16,center[1]-16:center[1]+16]

    
'''Functions for REINFORCE plus MVG entropy'''
def advance_sequence(images,coordinates=None,batch_size=64):
    if type(coordinates) != type(None):
        return tf.expand_dims(tf.constant(np.array([patch for patch in ThreadPool(batch_size).starmap(get_image_patch, [(images[i],coordinates[i]) for i in range(batch_size)])])),axis=0)
    else:
        return tf.expand_dims(tf.constant(np.array([patch for patch in ThreadPool(batch_size).starmap(get_image_patch, [(images[i],None) for i in range(batch_size)])])),axis=0)

def sample_coordinates(distributions):
    return [int(np.random.normal(distributions[0],distributions[2])),int(np.random.normal(distributions[1],distributions[3]))]

def action_probability(sample,distribution):
    df_x = lambda x: (1/(2*np.pi*distribution[2]))+np.exp((-1/(2*(distribution[2]**2)))*(x-distribution[0])**2)
    df_y = lambda y: (1/(2*np.pi*distribution[3]))+np.exp((-1/(2*(distribution[3]**2)))*(y-distribution[1])**2)
    return df_x(sample[0])*df_y(sample[1])

def capped_univariate_entropy_mean(distribution):
    entropy = lambda sigma: 0.5*(np.log(2*np.pi*sigma**2)+0.5) if sigma > 0.4 else 0.25
    x = entropy(np.array(distribution[2]))
    y = entropy(np.array(distribution[3]))
    return np.mean([x,y])
    
'''Training and evaluation loops'''
def train_epoch(sn_eye,sn_classifier,dataset,epoch,batch_size=64,sequence_len=20,gamma=0.98):
    epoch_eye_loss,epoch_classifier_loss,epoch_reward,epoch_accuracy = [], [], [], []
    '''Iterate over the dataset'''
    for data, target in tqdm(dataset,desc='Training epoch '):
        '''Generate first sequence element'''
        sequence = advance_sequence(data)
        
        '''Generate loss, accuracy and reward measures for batch sequence'''
        with tf.GradientTape() as eye_tape, tf.GradientTape() as class_tape:
            '''Generate necessary lists and variables for tracing'''
            classifier_loss,accuracy,reward = [],[],[]
            average_action_prob,average_entropy = [],[]
            eye_loss = None
            
            '''Loop over sequence length to generate sequence losses'''
            for step in range(sequence_len):
                '''Generate necessary variables'''
                class_hidden_h,class_hidden_c,eye_hidden_h,eye_hidden_c = None,None,None,None
                
                '''Feed the current sequence through the network'''
                distributions,eye_hidden_h,eye_hidden_c = sn_eye(sequence[-1],initial_state=[eye_hidden_h,eye_hidden_c])
                prediction,class_hidden_h,class_hidden_c = sn_classifier(sequence[-1],initial_state=[class_hidden_h,class_hidden_c])
                
                '''Generate next sequence element from sample if not at sequence end'''
                if step != sequence_len:
                    coordinates = np.array([coord for coord in ThreadPool(batch_size).map(sample_coordinates,[distributions[i].numpy() for i in range(batch_size)])])
                    sequence = tf.concat([sequence,advance_sequence(data,coordinates)],axis=0)
                    average_action_prob += [tf.reduce_mean([prob for prob in ThreadPool(batch_size).starmap(action_probability,[(coordinates[i],distributions[i]) for i in range(batch_size)])])]
                
                '''Calculate average entropy of the distribution'''
                average_entropy += [tf.reduce_mean([ent for ent in ThreadPool(batch_size).map(capped_univariate_entropy_mean,[distributions[i] for i in range(batch_size)])])]

                '''Classifier loss and accuracy calculations'''
                accuracy += [tf.reduce_mean(tf.cast(tf.argmax(prediction) == tf.argmax(target),'float32'))]
                classifier_loss += [sn_classifier.loss(target, prediction)]
        
            '''Calculate the discounted total reward'''
            reward = np.sum([-classifier_loss[0]*average_action_prob[0]]+[(gamma**step)*(-classifier_loss[step])*average_action_prob[step] for step in range(1,sequence_len)])

            '''Calculate average-based eye loss'''
            average_action_prob = tf.reduce_mean(average_action_prob)
            average_entropy = tf.reduce_mean(average_entropy)
            eye_loss = sn_eye.loss(reward,average_entropy)
            
            '''Taking the mean'''
            classifier_loss = tf.reduce_mean(classifier_loss)
            accuracy = tf.reduce_mean(accuracy)
            reward = tf.reduce_mean(reward)
            print('REWARD',reward)
            print('EYE LOSS',eye_loss)
            print('AVG ACTION PROB',average_action_prob)
            print('AVG ENTROPY',average_entropy)
            '''Appending results to epoch metric'''
            epoch_classifier_loss += [classifier_loss]
            epoch_eye_loss += [eye_loss]
            epoch_reward += [reward]
            epoch_accuracy += [accuracy]
            
            '''Applying gradients'''
            eye_gradient = eye_tape.gradient(eye_loss, sn_eye.trainable_variables)
            sn_eye.optimizer.apply_gradients(zip(eye_gradient, sn_eye.trainable_variables))
            class_gradient = class_tape.gradient(classifier_loss, sn_classifier.trainable_variables)
            sn_classifier.optimizer.apply_gradients(zip(class_gradient, sn_classifier.trainable_variables))
    
    '''Taking the mean'''
    epoch_reward = tf.reduce_mean(epoch_reward).numpy()
    epoch_accuracy = tf.reduce_mean(epoch_accuracy).numpy()
    epoch_classifier_loss = tf.reduce_mean(epoch_classifier_loss).numpy()
    epoch_eye_loss = tf.reduce_mean(epoch_eye_loss).numpy()
    
    '''Updating the performance csv files'''
    update_performance(sn_eye.nameinfo,epoch_eye_loss,None,epoch_reward,epoch)
    update_performance(sn_classifier.nameinfo,epoch_classifier_loss,epoch_accuracy,None,epoch)
    
    '''Storing model weights for future use'''
    update_weight_index(sn_eye.nameinfo,epoch)
    update_weight_index(sn_classifier.nameinfo,epoch)
    
    return epoch_classifier_loss,epoch_eye_loss,epoch_reward,epoch_accuracy
            
def evaluation_loop():
    pass