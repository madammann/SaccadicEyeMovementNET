import numpy as np
import pandas as pd

from multiprocessing.pool import ThreadPool

import tensorflow as tf
from tensorflow import keras

class PatchSeries:
    def __init__(self, data, sn_eye, length=20):
        self.batch_size = data.shape[0]
        self.sequence_len = length
        
        self.sn_eye = sn_eye
        self.images = data

        self.series = np.zeros((self.sequence_len,self.batch_size,32,32,3))
        self.distributions = np.zeros((self.sequence_len-1,self.batch_size,4))
        self.coordinates = np.zeros((self.sequence_len-1,self.batch_size,2),dtype='int32')
        
        for t in range(self.sequence_len):
            if t == 0:
                self.series[t] = np.array([patch for patch in ThreadPool(self.batch_size).starmap(self.get_image_patch, [(img,None) for img in self.images])])
            else:
                self.distributions[t-1] = self.sn_eye(tf.constant(self.series[0:t])).numpy()
                self.coordinates[t-1] = np.array([coord for coord in ThreadPool(self.batch_size).starmap(self.sample_coordinates,[(self.distributions[t-1][i],t) for i in range(self.batch_size)])])
                self.series[t] = np.array([patch for patch in ThreadPool(self.batch_size).starmap(self.get_image_patch, [(self.images[i],self.coordinates[t-1][i]) for i in range(self.batch_size)])])

    def get_image_patch(self, image, coordinates=None):
        size = image.shape[:2]
        center = coordinates
        if type(center) == type(None):
            center = [int(size[0]/2),int(size[1]/2)]
        if center[0] < 16:
            center[0] = 16
        elif center[0] > size[0] - 15:
            center[0] = size[0]-15
        if center[1] < 16:
            center[1] = 16
        elif center[1] > size[1] - 15:
            center[1] = size[1]-15
        return image[center[0]-16:center[0]+16,center[1]-16:center[1]+16]

    def sample_coordinates(self,dist,step):
        coordinates = [int(np.random.normal(dist[0],dist[2])),int(np.random.normal(dist[1],dist[3]))]
        return coordinates
    
    def create_distributions(self,series):
        input_tensor = tf.constant(series)
        return self.sn_eye(input_tensor).numpy()

    def return_patch_series(self):
        return self.series
    
    def return_distributions(self):
        return self.distributions
    
    def return_coordinates(self):
        return self.coordinates
    
class FeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same')
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.avgpool(x)
        return x

class SaccadicNetEye(tf.keras.Model):
    def __init__(self):
        super(SaccadicNetEye, self).__init__()
        '''Model structure'''
        self.feature_extractor = FeatureExtractor()

        self.LSTM = tf.keras.layers.LSTM(30,time_major=True)

        self.dense = tf.keras.layers.Dense(30,activation='relu')

        self.out_mu = tf.keras.layers.Dense(2,activation='linear')
        self.out_sigma = tf.keras.layers.Dense(2,activation='softplus')
        
        self.concat_layer = tf.keras.layers.Concatenate(0)

        '''Utility'''
        self.nameinfo = 'SaccadicNetEye'
        self.built = True
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = None
    
    @tf.function
    def call(self, x):
        sequence_len = x.shape[0]
        lst = []
        for step in range(sequence_len):
            lst.append(tf.expand_dims(self.feature_extractor(x[step]),axis=0))
        x = self.concat_layer(lst)
        
        x = self.LSTM(x)
        
        x = self.dense(x)
        a = self.out_mu(x)
        b = self.out_sigma(x)
        
        return tf.concat([a,b],axis=1)

    def store_intermediate(self, epoch, path='./weights/'):
        '''Stores the model weights after the training epoch'''
        name = self.nameinfo+str(epoch)
        self.save_weights('./weights/'+name+'.h5',save_format='HDF5')

    def load_intermediate(self, epoch, path='./weights/'):
        '''Loads the weights from a specific state'''
        name = self.nameinfo+str(epoch)
        self.load_weights('weights/'+name+'.h5')

    def load_latest(self, path='./weights/'):
        '''Loads the weights of the final model.'''
        try:
            df = pd.read_csv('./weights/index.csv')
            df.where('Model' == self.nameinfo).dropna()
            current_epoch = max(list(df['Epoch']))
            name = self.nameinfo+str(current_epoch)
            self.load_weights('weights/'+name+'.h5')
        except Exception as e:
            print(e)
            print('Could not load latest model from index.csv information.')
            print('You can ignore this error if the model has never been trained.')

class SaccadicNetClassifier(tf.keras.Model):
    def __init__(self, classes=11):
        super(SaccadicNetClassifier, self).__init__()
        '''Model structure'''
        self.feature_extractor = FeatureExtractor()

        self.LSTM = tf.keras.layers.LSTM(30,time_major=True)

        self.dense = tf.keras.layers.Dense(30,activation='relu')

        self.out = tf.keras.layers.Dense(classes,activation='softmax')
        
        self.concat_layer = tf.keras.layers.Concatenate(0)
        
        '''Utility'''
        self.nameinfo = 'SaccadicNetClassifier'
        self.built = True
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def call(self, x):
        sequence_len = x.shape[0]
        lst = []
        for step in range(sequence_len):
            lst.append(tf.expand_dims(self.feature_extractor(x[step]),axis=0))
        x = self.concat_layer(lst)
        
        x = self.LSTM(x)
        
        x = self.dense(x)
        x = self.out(x)
        return x

    def store_intermediate(self, epoch, path='./weights/'):
        '''Stores the model weights after the training epoch'''
        name = self.nameinfo+str(epoch)
        self.save_weights('./weights/'+name+'.h5',save_format='HDF5')

    def load_intermediate(self, epoch, path='./weights/'):
        '''Loads the weights from a specific state'''
        name = self.nameinfo+str(epoch)
        self.load_weights('weights/'+name+'.h5')

    def load_latest(self, path='./weights/'):
        '''Loads the weights of the final model.'''
        try:
            df = pd.read_csv('./weights/index.csv')
            df.where('Model' == self.nameinfo).dropna()
            current_epoch = max(list(df['Epoch']))
            name = self.nameinfo+str(current_epoch)
            self.load_weights('weights/'+name+'.h5')
        except Exception as e:
            print(e)
            print('Could not load latest model from index.csv information.')
            print('You can ignore this error if the model has never been trained.')
