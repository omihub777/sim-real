# ==============================
# Show acc. for each class.
# ==============================

import sys
import os
sys.path.append(os.path.abspath("model"))

import tensorflow as tf
import argparse
from utils import get_model, get_dataset
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--weight-path", required=True, type=str)
parser.add_argument("--model-name", required=True, type=str, help=['preact50'])
parser.add_argument("--data-path",default="data")
parser.add_argument("--dataset", default="sim_real",type=str)
parser.add_argument("--size", default=224, type=int)
parser.add_argument("--batch-size",default=16, type=int)
parser.add_argument("--eval-batch-size", default=64, type=int)

args = parser.parse_args()

train_ds, test_ds = get_dataset(args)
model = get_model(args)

if __name__=='__main__':
    model.build((2, 224,224,3)) # Build
    model.load_weights(args.weight_path) # Load
    # Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # import IPython; IPython.embed();exit(1)
    labels, preds = [], []
    for img,label in test_ds:
        labels += list(label.numpy())
        preds += list(model.predict(img).argmax(1))
    cm = tf.math.confusion_matrix(labels, preds)
    print(cm)
    with open("data/label_to_id_dict.dict", "rb") as f:
        label_to_id_dict = pickle.load(f)
    label_ids = [label_to_id_dict[i] for i in range(args.num_classes)]
    df_cm = pd.DataFrame(cm.numpy(), index=label_ids, columns=label_ids)
    plt.figure(figsize=(10,7))
    sns.heatmap(df_cm, annot=True)
    hist = model.evaluate(test_ds) # Evaluate
    # import IPython; IPython.embed();exit(1)
    plt.title(f"Loss:{round(hist[0],4)}, Acc:{round(hist[1]*100, 4)}")
    plt.show()
