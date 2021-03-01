import tensorflow as tf
import tensorflow.keras.layers as layers

class PreAct50(tf.keras.Model):
    def __init__(self, num_classes, weights='imagenet', freeze=False, freeze_upto="4"):
        """PreAct50. If `freeze`, all blocks before stage4 will be frozen."""
        super(PreAct50, self).__init__()
        self.net = tf.keras.applications.ResNet50V2(include_top=False, weights=weights, pooling="avg")
        self.fc = layers.Dense(num_classes)
        if freeze_upto in ["2", "3", "4", "5"]:
            self.freeze_upto = f"conv{freeze_upto}_block1_preact_bn"
        elif freeze_upto=='full':
            self.freeze_upto = "_"
        else:
            raise ValueError()

        if freeze:
            self._freeze()

    def _freeze(self):
        trainable = False
        for layer in self.net.layers:
            # print(layer.name)
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
    for i in [2,3,4,5, 'full']:
        print(f"==========Frozen upto {i}===========")
        net = PreAct50(10, weights=None, freeze=True, freeze_upto=i)
        # import IPython ; IPython.embed() ; exit(1)
        out =net(x)
        print(out.shape)
        net.summary()