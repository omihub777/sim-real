import tensorflow as tf
import tensorflow.keras.layers as layers

class PreAct50(tf.keras.Model):
    def __init__(self, num_classes, weights='imagenet'):
        super(PreAct50, self).__init__()
        self.net = tf.keras.applications.ResNet50V2(include_top=False, weights=weights, pooling="avg")
        self.fc = layers.Dense(num_classes)
    def call(self, x):
        out = self.net(x)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    b,h,w,c = 4,224, 224, 3
    x = tf.random.normal((b,h,w,c))
    net = PreAct50(10, weights=None)
    import IPython ; IPython.embed() ; exit(1)
    out =net(x)
    print(out.shape)
    net.summary()