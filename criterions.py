import tensorflow as tf

class AugMixSparseCategoricalCrossEntropy(tf.keras.Model):
    def __init__(self, lambda_=12.):
        super(AugMixSparseCategoricalCrossEntropy, self).__init__()
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kld = tf.keras.losses.KLDivergence()
        self.jsd = lambda p_mix, p_clean, p_aug1, p_aug2 : (self.kld(p_mix, p_clean) + self.kld(p_mix, p_aug1) + self.kld(p_mix, p_aug2))/3.
        self.lambda_ = lambda_

    def call(self, out_clean, out_aug1, out_aug2, label):
        loss = self.ce(label, out_clean) + self.lambda_ * self._js_div(out_clean, out_aug1, out_aug2)
        return loss

    def _js_div(self, out_clean, out_aug1, out_aug2):
        p_clean, p_aug1, p_aug2 = tf.keras.activations.softmax(out_clean), tf.keras.activations.softmax(out_aug1), tf.keras.activations.softmax(out_aug2)
        p_mix = tf.math.log(tf.clip_by_value((p_clean+p_aug1+p_aug2)/3., 1e-7, 1))
        jsd = self.jsd(p_mix, p_clean, p_aug1, p_aug2)
        return jsd

if __name__=="__main__":
    num_class = 10
    bs = 4

    out_clean = tf.random.normal((bs, num_class))
    out_aug1 = tf.random.normal((bs, num_class))
    out_aug2 = tf.random.normal((bs, num_class))

    label = tf.random.uniform((bs,1), maxval=num_class, dtype=tf.int32)
    criterion = AugMixSparseCategoricalCrossEntropy()
    loss = criterion(out_clean, out_aug1, out_aug2, label)
    print(loss)
