import sys
import os
sys.path.append(os.path.abspath("model"))

import comet_ml
import tensorflow as tf
import argparse
from utils import get_model, get_dataset, get_criterion,get_optimizer, get_lr_scheduler
#
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="sim_real", help='[sim_real, c10, mnist]', type=str)
parser.add_argument("--model-name", required=True, help='[preact18, preact34, preact50 ]', type=str)
parser.add_argument("--criterion", default="crossentropy", help="[crossentropy,]",type=str)
parser.add_argument("--optimizer", default="adam", help="[adam,]", type=str)
parser.add_argument("--learning-rate", default=1e-3, type=float)
parser.add_argument("--beta-1", default=0.9, type=float)
parser.add_argument("--beta-2", default=0.999, type=float)
parser.add_argument("--batch-size",default=16, type=int)
parser.add_argument("--eval-batch-size", default=64, type=int)
parser.add_argument("--data-path",required=True)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--mixed-precision", action="store_true")
parser.add_argument("--lr-scheduler", default="cosine", type=str, help=["cosine"])
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--patience", default=5, type=int)
args = parser.parse_args()

with open("data/api_key.txt",'r') as f:
    api_key = f.readline()

logger = comet_ml.Experiment(
    api_key=api_key,
    project_name="sim_real",
    auto_metric_logging=True,
    auto_param_logging=True,
)

if args.mixed_precision:
    print("Applied: Mixed Precision")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

train_ds, test_ds = get_dataset(args)
model = get_model(args)
criterion = get_criterion(args)
optimizer = get_optimizer(args)
lr_scheduler = get_lr_scheduler(args)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=args.patience, restore_best_weights=True)
model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
logger.set_name(f"{args.model_name}")
with logger.train():
    # import IPython; IPython.embed() ; exit(1)
    logger.log_parameters(vars(args))
    filename =f'{args.model_name}.hdf5'
    mc = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_accuracy', mode='max', save_best_only=True, verbose=True)
    model.fit(train_ds, validation_data=test_ds, epochs=args.epochs, callbacks=[lr_scheduler, early_stop, mc])
    model.save_weights(filename)
    logger.log_asset(filename)

    # Load model weights.
    model = get_model(args)
    model.build((2, 224,224,3)) # Build
    model.load_weights(filename) # Load
    # Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.evaluate(test_ds) # Evaluate

# if __name__ == "__main__":
#     b, h, w, c = 4, 224, 224, 3
#     x = tf.random.normal((b,h,w,c))
#     out = model(x)
#     model.summary()
#     print(out.shape)