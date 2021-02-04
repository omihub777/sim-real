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
args = parser.parse_args()

ds = get_dataset(args)
model = get_model(args)
criterion = get_criterion(args)
optimizer = get_optimizer(args)
model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
# model.fit()

if __name__ == "__main__":
    b, h, w, c = 4, 224, 224, 3
    x = tf.random.normal((b,h,w,c))
    out = model(x)
    model.summary()
    print(out.shape)