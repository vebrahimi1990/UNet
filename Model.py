
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
from Blocks import CNNBlock, ResCNNBlock, UpCNNBlock,UpSampleBlock


class myUNet_ResLearning(keras.Model):
    def __init__(self,filters):
        super(myUNet_ResLearning,self).__init__()
        self.depth = len(filters)
        self.resblock =[]
        self.upblock  =[]
        self.upcnnblock = []
        self.fCNN = CNNBlock(filters[0])
        self.lCNN = CNNBlock(filters[0])
        for i in range(len(filters)):
            self.resblock.append(ResCNNBlock(filters[i]))
        self.cnnblock1 = CNNBlock(filters[self.depth-1])
        filters.reverse()
        for j in range(len(filters)-1):
            self.upblock.append(UpSampleBlock(filters[j+1]))
            self.upcnnblock.append(UpCNNBlock(filters[j+1]))
        self.cnn = CNNBlock(1)
    def call(self,input_tensor,training=False):
        skip_connection =[]
        x = input_tensor
        x = self.fCNN(x,training=training)
        x = layers.LeakyReLU()(x)
        for i in range(self.depth-1):
            x = self.resblock[i](x,training=training)
            skip_connection.append(x)
            x = layers.MaxPool2D()(x)
        x = self.resblock[self.depth-1](x,training=training)

        x = self.cnnblock1(x,training=training)
        x = layers.LeakyReLU()(x)
        x = self.cnnblock1(x,training=training)
        x = layers.LeakyReLU()(x)

        skip_connection.reverse()

        for i in range(self.depth-1):
            x = self.upblock[i](x,training=training)
            x = layers.concatenate([x,skip_connection[i]])
            x = self.upcnnblock[i](x,training=training)
        x = self.lCNN(x,training=training)
        x = layers.LeakyReLU()(x)
        x = self.cnn(x,training=training)
        return x
    def model_shape(self):
        x = keras.Input(shape=(256,256,1))
        return keras.Model(inputs=[x],outputs=self.call(x))

    
