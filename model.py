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

class ClassifierHead(tf.keras.Model):
    def __init__(self,classes=10):
        super(ClassifierHead, self).__init__()
        self.dense = tf.keras.layers.Dense(20,activation='relu')
        self.out = tf.keras.layers.Dense(classes,activation='softmax')

    @tf.function
    def call(self, x):
        x = self.dense(x)
        x = self.out(x)
        return x

class ActionHead(tf.keras.Model):
    def __init__(self):
        super(ActionHead, self).__init__()
        self.dense = tf.keras.layers.Dense(20,activation='relu')
        self.out = tf.keras.layers.Dense(4,activation='linear')

    @tf.function
    def call(self, x):
        x = self.dense(x)
        x = self.out(x)
        return x

class SaccadicNetEye(tf.keras.Model):
    def __init__(self, destinct_id):
        super(SaccadicNetEye, self).__init__()
        '''Model parts'''
        self.feature_extractor = FeatureExtractor()
        self.LSMT = tf.keras.layers.LSTM(20)
        self.action_head = ActionHead()
        '''Optimizer and loss parts'''
        self.optimizer = None
        self.loss = None
        '''other'''
        self.nameinfo = ['SaccadicNetClassifier','classifier',destinct_id]
        
    @tf.function
    def call(self, x):
        x = self.feature_extractor(x)
        x = self.LSTM(x)
        x = self.action_head(x)
        return x
    
    def store_intermediate(self, path='./weights/'):
        '''Stores the model weights after the training epoch'''
        name = self.nameinfo[1]+self.nameinfo[2]+str(epoch)
        self.save_weights('./weights/'+name+'.h5',save_format='HDF5')

    def load_intermediate(self, epoch, path='./weights/'):
        '''Loads the weights from a specific state'''
        name = self.nameinfo[1]+self.nameinfo[2]+str(epoch)
        self.load_weights('weights/'+name+'.h5')
        
    def load_final(self, path='./weights/'):
        '''Loads the weights of the final model.'''
        try:
            df = pd.read_csv('./weights/index.csv')
            df.where('Part' == self.nameinfo[1]).dropna()
            df.where('Id' == self.nameinfo[2]).dropna()
            current_epoch = max(df['Epoch'])
            name = self.nameinfo[1]+self.nameinfo[2]+str(current_epoch)
            self.load_weights('weights/'+name+'.h5')
        except Exception as e:
            print(e)
            print('Could not load latest model from index.csv information.')

class SaccadicNetClassifier(tf.keras.Model):
    def __init__(self, destinct_id):
        super(SaccadicNetClassifier, self).__init__()
        '''Model parts'''
        self.feature_extractor = FeatureExtractor()
        self.LSMT = tf.keras.layers.LSTM(20)
        self.classifier_head = ActionHead()
        '''Optimizer and loss parts'''
        self.optimizer = None
        self.loss = None
        '''other'''
        self.built = True
        self.nameinfo = ['SaccadicNetClassifier','classifier',destinct_id]
    
    @tf.function
    def call(self, x):
        x = self.feature_extractor(x)
        x = self.LSTM(x)
        x = self.ClassifierHead(x)
        return x
    
    def store_intermediate(self, path='./weights/'):
        '''Stores the model weights after the training epoch'''
        name = self.nameinfo[1]+self.nameinfo[2]+str(epoch)
        self.save_weights('./weights/'+name+'.h5',save_format='HDF5')

    def load_intermediate(self, epoch, path='./weights/'):
        '''Loads the weights from a specific state'''
        name = self.nameinfo[1]+self.nameinfo[2]+str(epoch)
        self.load_weights('weights/'+name+'.h5')
        
    def load_final(self, path='./weights/'):
        '''Loads the weights of the final model.'''
        try:
            df = pd.read_csv('./weights/index.csv')
            df.where('Part' == self.nameinfo[1]).dropna()
            df.where('Id' == self.nameinfo[2]).dropna()
            current_epoch = max(df['Epoch'])
            name = self.nameinfo[1]+self.nameinfo[2]+str(current_epoch)
            self.load_weights('weights/'+name+'.h5')
        except Exception as e:
            print(e)
            print('Could not load latest model from index.csv information.')
        