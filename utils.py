import tensorflow as tf
import sys
import os
import glob

sys.path.append(os.path.abspath("data"))


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
        args.num_classes = 24
        # list_ds = tf.data.Dataset.list_files("data/trainB")
        train_img_paths = glob.glob(f"{args.data_path}/mask/*.png")
        test_img_paths = glob.glob(f"{args.data_path}/valB/*.jpg")
        with open(f'{args.data_path}/id_to_label.txt','r') as f:
            id_to_label = f.readlines()

        id_to_object_dict = {l.split('  ')[0]: int(l.split('  ')[1].replace('\n','')) for l in id_to_label}
        object_to_label_dict = {object_:i for i,object_ in enumerate(id_to_object_dict.values())}

        train_labels = [object_to_label_dict[int(img_path.split('-')[-1].replace('object','').replace('.png',''))] for img_path in train_img_paths]
        test_labels = [object_to_label_dict[id_to_object_dict[img_path.split('/')[-1].split('_')[0]]] for img_path in test_img_paths]
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


