import tensorflow as tf

class FeatureExtractor(tf.keras.Model):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=5,strides=1,padding='same',activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64,kernel_size=5,strides=1,padding='same',activation='relu')
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same')
        self.conv3 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=2,strides=2,padding='same')
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
#         self.output_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

    def call(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.maxpool2(x)
        print(x.shape)
        x = self.avgpool(x)
        print(x.shape)
        return x

class LstmLayer(tf.keras.Model):
    def __init__(self):
        super(LstmLayer, self).__init__()

    def call(self, x):
        return x

class ClassifierHead(tf.keras.Model):
    def __init__(self,classes=10):
        super(ClassifierHead, self).__init__()
        self.output = tf.keras.layers.Dense(classes,activation='softmax')

    def call(self, x):
        x = self.output(x)
        return x

class ActionHead(tf.keras.Model):
    def __init__(self):
        super(ActionHead, self).__init__()
        self.output = tf.keras.layers.Dense(2,activation='linear')

    def call(self, x):
        x = self.output(x)
        return x

class SaccadicNet(tf.keras.Model):
    def __init__(self):
        super(SaccadicNet, self).__init__()
        '''Model parts'''
        self.feature_extractor = FeatureExtractor()
        self.LSMT_part = None
        self.classifier_head = ClassifierHead()
        self.action_head = ActionHead()
        '''Optimizer and loss parts'''
        self.optimizer = None
        self.loss = {'FE' : None, 'LSTM' : None, 'CH' : None, 'AH' : None}
        
    def store_intermediate(self, path='./weights/intermediate/'):
        '''Stores the model weights after the training epoch'''
        pass

    def load_intermediate(self, epoch, path='./weights/intermediate/'):
        '''Loads the weights from a specific state'''
        pass

    def store_final(self, path='./weights/'):
        '''Stores the weights of the final model when terminating training.'''
        pass

    def load_final(self, path='./weights/'):
        '''Loads the weights of the final model.'''
        pass

    def call(self, x):
        return x
