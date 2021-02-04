import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations


class PreActBlock(tf.keras.Model):
    def __init__(self, out_c, k=3, s=1, use_bias=False):
        super(PreActBlock, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(out_c, kernel_size=k, strides=(s,s),padding='same', use_bias=use_bias)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(out_c, kernel_size=k, strides=(1,1),padding='same', use_bias=use_bias)

        if s!=1:
            self.skip=layers.Conv2D(out_c, kernel_size=1, strides=(s,s), padding='valid')
        else:
            #identity
            self.skip=layers.Lambda(lambda x:x)

    def call(self, x):
        out = self.conv1(activations.relu(self.bn1(x)))
        out = self.conv2(activations.relu(self.bn2(out)))
        return out+self.skip(x)