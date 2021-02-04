import tensorflow as tf

def get_dataset(args):
    if args.dataset == 'sim_real':
        args.num_classes = 23
    else:
        raise NotImplementedError(f"{args.dataset} is NOT existing.")
    return None
    

def get_model(args):
    if args.model_name=='preact18':
        from model.preact18 import PreAct18
        net = PreAct18(args.num_classes)
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


