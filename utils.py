import tensorflow as tf
import sys
import os
import glob

sys.path.append(os.path.abspath("data"))

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    Callback,
    LearningRateScheduler,
    TensorBoard
)


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



def parse_function(filename, label, size=224):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [size, size])
    return image, label

def train_preprocess(image, label):
    image = tf.image.random_flip_left_right(image)

    return image, label

def get_dataset(args):
    if args.dataset == 'sim_real':
        # list_ds = tf.data.Dataset.list_files("data/trainB")
        train_img_paths_all = glob.glob(f"{args.data_path}/mask/*.png")
        # test_img_paths = glob.glob(f"{args.data_path}/valB/*.jpg")
        test_img_paths = glob.glob(f"{args.data_path}/trainB/*.jpg")
        args.total_test_images = len(test_img_paths)
        with open(f'{args.data_path}/id_to_label.txt','r') as f:
            id_to_label = f.readlines()

        id_to_object_dict = {l.split('  ')[0]: int(l.split('  ')[1].replace('\n','')) for l in id_to_label}
        object_to_label_dict = {object_:i for i,object_ in enumerate(id_to_object_dict.values())}

        test_ids = [img_path.split('/')[-1].split('_')[0] for img_path in test_img_paths]
        test_labels = [object_to_label_dict[id_to_object_dict[id_]] for id_ in test_ids]
        test_object_set = [id_to_object_dict.get(test_id) for test_id in set(test_ids)]
        args.num_classes = len(test_object_set)

        # train_img_paths = [img_path for img_path in train_img_paths_all if int(img_path.split('-')[-1].replace('object','').replace('.png','')) in test_object_set]
        train_img_paths = []
        for img_path in train_img_paths_all:
            object_ = int(img_path.split('-')[-1].replace('object','').replace('.png',''))
            if object_ in test_object_set:
                train_img_paths.append(img_path)
        args.total_train_images = len(train_img_paths)
        train_labels = [object_to_label_dict[int(img_path.split('-')[-1].replace('object','').replace('.png',''))] for img_path in train_img_paths]
        # import IPython ; IPython.embed();exit(1)

        train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))
        train_ds = train_ds.shuffle(len(train_labels))
        train_ds = train_ds.map(parse_function, num_parallel_calls=4)
        train_ds = train_ds.map(train_preprocess, num_parallel_calls=4)
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
    args = argparse.Namespace()
    args.dataset = 'sim_real'
    args.batch_size = 16
    args.eval_batch_size = 64
    args.data_path = 'data'
    train_ds, test_ds = get_dataset(args)


def get_model(args):
    if args.model_name=='preact18':
        from model.preact18 import PreAct18
        net = PreAct18(args.num_classes)
    elif args.model_name=='preact34':
        from model.preact34 import PreAct34
        net = PreAct34(args.num_classes)
    else:
        raise NotImplementedError(f"{args.model_name} is NOT implemented yet.")

    return net

def get_criterion(args):
    if args.criterion=='crossentropy':
        # label should be integer.(NOT ONE-HOT)
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