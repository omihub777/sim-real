import sys
import os
sys.path.append(os.path.abspath("model"))

import tensorflow as tf
import argparse
from utils import get_model, get_dataset, get_criterion,get_optimizer


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="sim_real", help='[sim_real,]', type=str)
parser.add_argument("--model-name", required=True, help='[preact18, ]', type=str)
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
args = parser.parse_args()

if args.mixed_precision:
    print("Applied: Mixed Precision")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

train_ds, test_ds = get_dataset(args)
model = get_model(args)
criterion = get_criterion(args)
optimizer = get_optimizer(args)
model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
model.fit(train_ds, validation_data=test_ds, epochs=args.epochs)

# if __name__ == "__main__":
#     b, h, w, c = 4, 224, 224, 3
#     x = tf.random.normal((b,h,w,c))
#     out = model(x)
#     model.summary()
#     print(out.shape)