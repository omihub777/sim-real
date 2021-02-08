# ==============================
# Show acc. for each class.
# ==============================

import sys
import os
sys.path.append(os.path.abspath("model"))

import tensorflow as tf
import argparse
from utils import get_model, get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--weight-path", required=True, type=str)
parser.add_argument("--model-name", required=True, type=str, help=['preact50'])
parser.add_argument("--data-path",default="data")
parser.add_argument("--dataset", default="sim_real",type=str)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--batch-size",default=16, type=int)
parser.add_argument("--eval-batch-size", default=64, type=int)

args = parser.parse_args()

_, test_ds = get_dataset(args)
model = get_model(args)

if __name__=='__main__':
    model.build((2, 224,224,3)) # Build
    model.load_weights(args.weight_path) # Load
    # Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.evaluate(test_ds) # Evaluate
