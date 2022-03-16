import tensorflow as tf
from tensorflow import keras

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
        
        self.LSMT = tf.keras.layers.LSTM(30)
        
        self.dense = tf.keras.layers.Dense(30,activation='relu')
        
        self.out_mu = tf.keras.layers.Dense(2,activation='linear')
        self.out_sigma = tf.keras.layers.Dense(2,activation='softplus')
        
        '''Utility'''
        self.name = 'SaccadicNetClassifier'
        self.built = True
        self.optimizer = None
        self.loss = None
        
    @tf.function
    def call(self, x):
        x = self.feature_extractor(x)
        x = self.LSTM(x)
        x = self.dense(x)
        a = self.out_mu(x)
        b = self.out_sigma(x)
        return tf.concat([a,b],axis=1)
    
    def store_intermediate(self, epoch, path='./weights/'):
        '''Stores the model weights after the training epoch'''
        name = self.name+str(epoch)
        self.save_weights('./weights/'+name+'.h5',save_format='HDF5')

    def load_intermediate(self, epoch, path='./weights/'):
        '''Loads the weights from a specific state'''
        name = self.name+str(epoch)
        self.load_weights('weights/'+name+'.h5')
        
    def load_latest(self, path='./weights/'):
        '''Loads the weights of the final model.'''
        try:
            df = pd.read_csv('./weights/index.csv')
            df.where('Model' == self.name).dropna()
            current_epoch = max(list(df['Epoch']))
            name = self.name+str(current_epoch)
            self.load_weights('weights/'+name+'.h5')
        except Exception as e:
            print(e)
            print('Could not load latest model from index.csv information.')
            print('You can ignore this error if the model has never been trained.')

class SaccadicNetClassifier(tf.keras.Model):
    def __init__(self, classes=10):
        super(SaccadicNetClassifier, self).__init__()
        '''Model structure'''
        self.feature_extractor = FeatureExtractor()
        
        self.LSMT = tf.keras.layers.LSTM(30)
        
        self.dense = tf.keras.layers.Dense(30,activation='relu')
        
        self.out = tf.keras.layers.Dense(classes,activation='softmax')

        '''Utility'''
        self.name = 'SaccadicNetClassifier'
        self.built = True
        self.optimizer = None
        self.loss = None
    
    @tf.function
    def call(self, x):
        x = self.feature_extractor(x)
        x = self.LSTM(x)
        x = self.dense(x)
        x = self.out(x)
        return x
    
    def store_intermediate(self, epoch, path='./weights/'):
        '''Stores the model weights after the training epoch'''
        name = self.name+str(epoch)
        self.save_weights('./weights/'+name+'.h5',save_format='HDF5')

    def load_intermediate(self, epoch, path='./weights/'):
        '''Loads the weights from a specific state'''
        name = self.name+str(epoch)
        self.load_weights('weights/'+name+'.h5')
        
    def load_latest(self, path='./weights/'):
        '''Loads the weights of the final model.'''
        try:
            df = pd.read_csv('./weights/index.csv')
            df.where('Model' == self.name).dropna()
            current_epoch = max(list(df['Epoch']))
            name = self.name+str(current_epoch)
            self.load_weights('weights/'+name+'.h5')
        except Exception as e:
            print(e)
            print('Could not load latest model from index.csv information.')
            print('You can ignore this error if the model has never been trained.')
        