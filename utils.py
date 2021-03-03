import sys
import os
import glob
import pickle

sys.path.append(os.path.abspath("data"))

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    Callback,
    LearningRateScheduler,
    TensorBoard
)
import tensorflow_addons as tfa



from criterions import AugMixSparseCategoricalCrossEntropy

# Code: https://gist.github.com/scorrea92/b9485cbe26cb010e81af02b6c5d0c2ab
class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """
    def __init__(self,
                 args,
                 learning_rate_base,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_epoch=0,
                 hold_base_rate_steps=0,
                 learning_rate_final=None,
                 stop_epoch=None,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
        Arguments:
            learning_rate_base {float} -- base learning rate.
            total_steps {int} -- total number of training steps.
        Keyword Arguments:
            global_step_init {int} -- initial global step, e.g. from previous checkpoint.
            warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
            warmup_steps {int} -- number of warmup steps. (default: {0})
            hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                        before decaying. (default: {0})
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.args = args
        self.learning_rate_base = learning_rate_base
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_epoch = warmup_epoch
        self.hold_base_rate_steps = hold_base_rate_steps
        self.learning_rates = []
        self.verbose = verbose
        self.stop_epoch = stop_epoch
        self.learning_rate_final = learning_rate_final
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        # import IPython; IPython.embed(); exit(1)
        total_steps = int(
            self.params['epochs'] * self.args.total_train_images / self.args.batch_size)
        warmup_steps = int(
            self.warmup_epoch * self.args.total_train_images / self.args.batch_size)
        lr = self.cosine_decay_with_warmup(
            global_step=self.global_step,
            learning_rate_base=self.learning_rate_base,
            total_steps=total_steps,
            warmup_learning_rate=self.warmup_learning_rate,
            warmup_steps=warmup_steps,
            hold_base_rate_steps=self.hold_base_rate_steps)
        if self.stop_epoch is not None and self.stop_epoch > 0 and self.epoch >= self.stop_epoch:
            if self.learning_rate_final is not None:
                K.set_value(self.model.optimizer.lr, self.learning_rate_final)
            else:
                self.learning_rate_final = lr
                K.set_value(self.model.optimizer.lr, self.learning_rate_final)
        else:
            K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
#
    def cosine_decay_with_warmup(self, global_step,
                                 learning_rate_base,
                                 total_steps,
                                 warmup_learning_rate=0.0,
                                 warmup_steps=0,
                                 hold_base_rate_steps=0):
        """Cosine decay schedule with warm up period.
        Cosine annealing learning rate as described in
            Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
            ICLR 2017. https://arxiv.org/abs/1608.03983
        In this schedule, the learning rate grows linearly from warmup_learning_rate
        to learning_rate_base for warmup_steps, then transitions to a cosine decay
        schedule.
        Arguments:
            global_step {int} -- global step.
            learning_rate_base {float} -- base learning rate.
            total_steps {int} -- total number of training steps.
        Keyword Arguments:
            warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
            warmup_steps {int} -- number of warmup steps. (default: {0})
            hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                        before decaying. (default: {0})
        Returns:
            a float representing learning rate.
        Raises:
            ValueError: if warmup_learning_rate is larger than learning_rate_base,
            or if warmup_steps is larger than total_steps.
        """
        if total_steps < warmup_steps:
            raise ValueError('total_steps must be larger or equal to '
                             'warmup_steps.')
        learning_rate = 0.5 * learning_rate_base * (
            1 + np.cos(
                np.pi * (global_step - warmup_steps - hold_base_rate_steps) /
                float(total_steps - warmup_steps - hold_base_rate_steps)
                )
            )
        if hold_base_rate_steps > 0:
            learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                     learning_rate, learning_rate_base)
        if warmup_steps > 0:
            if learning_rate_base < warmup_learning_rate:
                raise ValueError('learning_rate_base must be larger or equal to '
                                 'warmup_learning_rate.')
            slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
            warmup_rate = slope * global_step + warmup_learning_rate
            learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                     learning_rate)
        return np.where(global_step > total_steps, 0.0, learning_rate)




def get_dataset(args):
    operations = [
        lambda image: tf.image.random_brightness(image, 0.2),
        lambda image: tf.image.random_contrast(image, lower=0.7, upper=1.3),
        lambda image: tf.image.random_hue(image, .1),
        lambda image: tfa.image.sharpness(image, tf.random.uniform(shape=(1,), minval=0.8, maxval=1.2)),
        lambda image: tfa.image.shear_x(image, 0.05, replace=1.),
        lambda image: tfa.image.shear_y(image, 0.05, replace=1.),
        tfa.image.gaussian_filter2d,
        lambda image: tfa.image.random_cutout(image[tf.newaxis,], (args.size//8, args.size//8), constant_values=0)[0],
        lambda image: tfa.image.rotate(image, angles=tf.random.uniform(shape=(1,), minval=-args.angle*np.pi, maxval=args.angle*np.pi, dtype=tf.float32)),
        tfa.image.equalize,
    ]
    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image = tf.io.decode_png(image_string, channels=3)
        # image = tf.io.decode_image(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [args.size, args.size])
        return image, label

    def train_preprocess(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.resize_with_crop_or_pad(image, args.size+args.padding*2, args.size+args.padding*2)
        image = tf.image.random_crop(image, [args.size, args.size, 3])
        return image, label

    def augmix_preprocess(image, label):
        weights = np.random.dirichlet((args.augmix_alpha, args.augmix_alpha, args.augmix_alpha))
        image_aug = tf.zeros_like(image)

        for w in weights:
            op1, op2, op3 = np.random.choice(operations, size=3, replace=False)
            op12, op123 = lambda image: op2(op1(image)), lambda image: op3(op2(op1(image)))
            chain = np.random.choice([op1, op12, op123])
            image_aug += w*chain(image)
        m = np.random.beta(args.augmix_alpha, args.augmix_alpha)
        image = m*image + (1.-m)*image_aug
        return image, label


    if args.dataset == 'sim_real':
        # list_ds = tf.data.Dataset.list_files("data/trainB")
        # train_img_paths_all = glob.glob(f"{args.data_path}/mask/*.png")
        train_img_paths_all = glob.glob(f"{args.data_path}/new_mask/*/*.png")

        # test_img_paths = glob.glob(f"{args.data_path}/valB/*.jpg")
        # test_img_paths = glob.glob(f"{args.data_path}/trainB/*.jpg")
        # test_img_paths = glob.glob(f"{args.data_path}/trainB_mask/*.png")
        # test_img_paths = glob.glob(f"{args.data_path}/trainB_mask_bk/*.png")
        test_img_paths = glob.glob(f"{args.data_path}/new_test_mask/*/*.jpg")
        args.total_test_images = len(test_img_paths)
        with open(f'{args.data_path}/id_to_object.txt','r') as f:
            id_to_object = f.readlines()

        id_to_object_dict = {l.split('  ')[0]: int(l.split('  ')[1].replace('\n','')) for l in id_to_object}
        # object_to_label_dict = {object_:i for i,object_ in enumerate(id_to_object_dict.values())}

        # test_ids = [img_path.split('/')[-1].split('_')[0] for img_path in test_img_paths]
        test_ids = [img_path.split('/')[-2] for img_path in test_img_paths]

        # for test_id in set(test_ids):
        #     print(f"{test_id}: {id_to_object_dict.get(test_id)}")
        test_object_set = [id_to_object_dict.get(test_id) for test_id in set(test_ids)]
        if not os.path.exists("data/object_to_label_dict.dict"):
            object_to_label_dict = {object_:i for i,object_ in enumerate(test_object_set)}
            with open("data/object_to_label_dict.dict", 'wb') as fw:
                pickle.dump(object_to_label_dict, fw, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open("data/object_to_label_dict.dict", 'rb') as f:
                object_to_label_dict = pickle.load(f)
        args.num_classes = len(test_object_set)

        train_img_paths = []
        for img_path in train_img_paths_all:
            # object_ = int(img_path.split('-')[-1].replace('object','').replace('.png','')) # old
            object_ = int(img_path.split('/')[-2])
            if object_ in test_object_set:
                train_img_paths.append(img_path)
        args.total_train_images = len(train_img_paths)
        # train_labels = [object_to_label_dict[int(img_path.split('-')[-1].replace('object','').replace('.png',''))] for img_path in train_img_paths] # old
        train_labels = [object_to_label_dict[int(img_path.split('/')[-2])] for img_path in train_img_paths]
        test_labels = [object_to_label_dict[id_to_object_dict[id_]] for id_ in test_ids]
        # import IPython ; IPython.embed();exit(1)

        train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))
        train_ds = train_ds.shuffle(len(train_labels))
        train_ds = train_ds.map(parse_function, num_parallel_calls=4)
        train_ds = train_ds.map(train_preprocess, num_parallel_calls=4)
        if args.augmix:
            train_ds = train_ds.map(augmix_preprocess, num_parallel_calls=4)
        train_ds = train_ds.batch(args.batch_size)
        train_ds= train_ds.prefetch(1)


        # import IPython; IPython.embed();exit(1)
        test_ds = tf.data.Dataset.from_tensor_slices((test_img_paths, test_labels))
        test_ds = test_ds.map(parse_function, num_parallel_calls=4)
        test_ds = test_ds.batch(args.eval_batch_size)
        test_ds = test_ds.prefetch(1)
    elif args.dataset=='c10' or args.dataset=='mnist':
        args.num_classes = 10
        if args.dataset=='c10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        elif args.dataset=='mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train[..., tf.newaxis].astype("float32")
            x_test = x_test[..., tf.newaxis].astype("float32")
        args.total_train_images = x_train.shape[0]
        args.total_test_images = x_test.shape[0]

        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(args.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(args.eval_batch_size)
        train_ds = train_ds.prefetch(1)
        test_ds = test_ds.prefetch(1)
    else:
        raise NotImplementedError(f"{args.dataset} is NOT existing.")
    return train_ds, test_ds
    
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    args = argparse.Namespace()
    args.dataset = 'sim_real'
    args.batch_size = 8
    args.eval_batch_size = 64
    args.data_path = 'data'
    args.size=128
    args.padding=4
    args.augmix=True
    args.augmix_alpha=0.5
    args.angle=np.pi*0.05
    train_ds, test_ds = get_dataset(args)
    img, label = next(iter(train_ds))
    for i in range(args.batch_size):
        plt.imshow(img[i])
        plt.show()
    # plt.show()
    # import IPython; IPython.embed();exit(1)


def get_model(args):
    if args.model_name=='preact18':
        from model.preact18 import PreAct18
        net = PreAct18(args.num_classes)
    elif args.model_name=='preact34':
        from model.preact34 import PreAct34
        net = PreAct34(args.num_classes)
    elif args.model_name=='preact50':
        from model.preact50 import PreAct50
        net = PreAct50(args.num_classes, freeze=args.freeze, freeze_upto=args.freeze_upto)
    elif 'effb' in args.model_name:
        from model.efficientnet import EfficientNet
        version = int(args.model_name.replace('effb',''))
        assert version == 3 or version == 4
        net = EfficientNet(version, num_classes=args.num_classes, weights='noisy', freeze=args.freeze, freeze_upto=args.freeze_upto)
    else:
        raise NotImplementedError(f"{args.model_name} is NOT implemented yet.")

    return net

def get_criterion(args):
    if args.criterion=='crossentropy':
        # label should be integer.(NOT ONE-HOT)
        if args.jsd:
            criterion = AugMixSparseCategoricalCrossEntropy()
        else:
            criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise NotImplementedError(f"{args.criterion} is NOT existing.")

    return criterion

def get_optimizer(args):
    if args.optimizer=='adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.beta_1, beta_2=args.beta_2)
    else:
        raise NotImplementedError(f"{args.optimizer} is NOT existing.")

    return optimizer


def get_lr_scheduler(args):
    if args.lr_scheduler=='cosine':
        lr_scheduler = WarmUpCosineDecayScheduler(args, learning_rate_base=args.learning_rate, warmup_epoch=args.warmup_epoch)

    return lr_scheduler


def get_experiment_name(args):
    experiment_name = f"{args.model_name}"
    if args.freeze:
        experiment_name += f"_freeze_{args.freeze_upto}"
    if args.augmix:
        experiment_name += f"_augmix"
    return experiment_name


def image_grid(x, size=4):
    t = tf.unstack(x[:size * size], num=size*size, axis=0)
    rows = [tf.concat(t[i*size:(i+1)*size], axis=0) 
            for i in range(size)]
    image = tf.concat(rows, axis=1)
    return image[None]