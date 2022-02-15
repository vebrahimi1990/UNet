import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model



class CNNBlock(layers.Layer):
    def __init__(self,num_filters, kernel_size=3):
        super(CNNBlock,self).__init__()
        self.conv = layers.Conv2D(num_filters,kernel_size,padding='same')
        self.bn   = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x,training=training)
        return x

class ResCNNBlock(layers.Layer):
    def __init__(self,num_filters,kernel_size=3):
        super(ResCNNBlock,self).__init__()
        self.conv1 = CNNBlock(num_filters)
        self.conv2 = CNNBlock(num_filters)
        self.add   = layers.Conv2D(num_filters,kernel_size = 3,padding='same')
    def call(self,input_tensor,training=False):
        x = self.conv1(input_tensor,training=training)
        x = layers.LeakyReLU()(x)
        x = self.conv2(x,training=training)
        y = self.add(x)
        x = layers.add([x,y])
        return x

class UpSampleBlock(layers.Layer):
    def __init__(self,num_filters,kernel_size=3):
        super(UpSampleBlock,self).__init__()
        self.conv1 = layers.Conv2DTranspose(num_filters,kernel_size,strides=(2,2),padding='same')
    def call(self,input_tensor,training=False):
        x = self.conv1(input_tensor)
        return x
class UpCNNBlock(layers.Layer):
    def __init__(self,num_filters,kernel_size=3):
        super(UpCNNBlock,self).__init__()
        self.conv1 = CNNBlock(num_filters,kernel_size)
        self.conv2 = CNNBlock(num_filters,kernel_size)
    def call(self,input_tensor,training=False):
        x = self.conv1(input_tensor,training=training)
        x = layers.LeakyReLU()(x)
        x = self.conv2(x,training=training)
        x = layers.LeakyReLU()(x)
        return x





