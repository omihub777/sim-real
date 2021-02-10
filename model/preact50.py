import tensorflow as tf
import tensorflow.keras.layers as layers

class PreAct50(tf.keras.Model):
    def __init__(self, num_classes, weights='imagenet', freeze=False):
        """PreAct50. If `freeze`, all blocks before stage4 will be frozen."""
        super(PreAct50, self).__init__()
        self.net = tf.keras.applications.ResNet50V2(include_top=False, weights=weights, pooling="avg")
        self.fc = layers.Dense(num_classes)
        self.freeze_upto = "conv4_block1_preact_bn"
        if freeze:
            self._freeze()

    def _freeze(self):
        trainable = False
        for layer in self.net.layers:
            if layer.name == self.freeze_upto:
                print(f"Freezed layers upto {layer.name}")
                trainable = True
            layer.trainable=trainable               


    def call(self, x):
        out = self.net(x)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    b,h,w,c = 4,224, 224, 3
    x = tf.random.normal((b,h,w,c))
    net = PreAct50(10, weights=None, freeze=True)
    # import IPython ; IPython.embed() ; exit(1)
    out =net(x)
    print(out.shape)
    net.net.summary()