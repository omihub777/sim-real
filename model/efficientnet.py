import tensorflow as tf

class EfficientNet(tf.keras.Model):
    def __init__(self, version, num_classes, weights='noisy', freeze=True, freeze_upto='3'):
        super(EfficientNet, self).__init__()
        if weights is not None:
            weights = f"efficientnetb{version}_notop.h5"
        if version==3:
            self.base = tf.keras.applications.EfficientNetB3(include_top=False, weights=weights, pooling='avg')
        elif version==4:
            self.base = tf.keras.applications.EfficientNetB4(include_top=False, weights=weights, pooling='avg')
        else:
            ValueError(f"What is EfficientNet-B{version}?")

        self.fc = tf.keras.layers.Dense(num_classes)

        # num_blocks = version*

        if freeze_upto=='full':
            self.freeze_upto = "_"
        elif freeze_upto in ["2", "3", "4", "5","6", "7"]:
            self.freeze_upto = f"block{freeze_upto}a_expand_conv"
        else:
            raise ValueError()

        if freeze:
            self._freeze()

    def _freeze(self):
        trainable = False
        for layer in self.base.layers:
            if layer.name == self.freeze_upto:
                print(f"Freezed layers upto {layer.name}")
                trainable = True
            layer.trainable=trainable               

    def call(self, x):
        out = self.base(x)
        out = self.fc(out)
        return out




if __name__ == "__main__":
    b,h,w,c = 2,128,128,3
    x = tf.random.normal((b,h,w,c))
    net = EfficientNet(version=4, num_classes=10, weights=None, freeze=True, freeze_upto='full')
    # net = tf.keras.applications.EfficientNetB7(False, weights=None)
    out = net(x)
    net.summary()
    # for layer in net.base.layers:
    #     print(layer.name)
    
