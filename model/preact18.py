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

class PreAct18(tf.keras.Model):
    def __init__(self, num_classes):
        super(PreAct18, self).__init__()
        self.blc1 = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=7,strides=(2,2), use_bias=False,padding='same'),
            layers.BatchNormalization()
        ]
        )
        self.blc2 = tf.keras.Sequential([
            PreActBlock(64),
            PreActBlock(64)
        ])

        self.blc3 = tf.keras.Sequential([
            PreActBlock(128, s=2),
            PreActBlock(128)
        ])

        self.blc4 = tf.keras.Sequential([
            PreActBlock(256, s=2),
            PreActBlock(256)
        ])

        self.blc5 = tf.keras.Sequential([
            PreActBlock(512, s=2),
            PreActBlock(512)
        ])

        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)


    def call(self, x):
        out = tf.nn.max_pool2d(activations.relu(self.blc1(x)), ksize=3,strides=(2,2),padding="SAME")
        out = self.blc2(out)
        out = self.blc3(out)
        out = self.blc4(out)
        out = self.blc5(out)
        out = self.gap(out)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    b, h, w, c = 4, 224, 224, 3
    # net = PreActBlock(c, 16, s=2)
    net = PreAct18(100)
    x = tf.random.normal((b,h,w,c))
    # net.build((b,h,w,c))
    out = net(x)
    # conv1 = layers.Conv2D(16, kernel_size=3, strides=(2,2),padding='same', use_bias=False)
    # out = conv1(x)
    net.summary()
    print(out.shape)

