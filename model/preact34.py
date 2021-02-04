import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
from layer_utils import PreActBlock

class PreAct34(tf.keras.Model):
    def __init__(self, num_classes):
        super(PreAct34, self).__init__()
        self.blc1 = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7, strides=(2,2), padding='same', use_bias=False),
            layers.BatchNormalization(),
        ])
        self.blc2 = tf.keras.Sequential([
            PreActBlock(64),
            PreActBlock(64),
            PreActBlock(64)
        ])
        self.blc3 = tf.keras.Sequential([
            PreActBlock(128, s=2),
            PreActBlock(128),
            PreActBlock(128),
            PreActBlock(128)
        ])
        self.blc4 = tf.keras.Sequential([
            PreActBlock(256, s=2),
            PreActBlock(256),
            PreActBlock(256),
            PreActBlock(256),
            PreActBlock(256),
            PreActBlock(256)
        ])
        self.blc5 = tf.keras.Sequential([
            PreActBlock(512, s=2),
            PreActBlock(512),
            PreActBlock(512)
        ])
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)

    def call(self, x):
        out = tf.nn.avg_pool2d(activations.relu(self.blc1(x)), ksize=3, strides=(2,2), padding='SAME')
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        print(out.shape)
        out = self.gap(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    b,h,w,c = 4, 224, 224, 3
    x = tf.random.normal((b, h, w, c))
    net = PreAct34(10)
    out = net(x)
    net.summary()
    print(out.shape)