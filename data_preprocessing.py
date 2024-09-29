import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf 
import pickle
import os

## preprocess the data and save data into tf.data.Dataset format
path = "."

data_path="temp.txt"
df = pd.read_csv(data_path, sep="\t", header=None, names=["Seq", "Expression"])

dataset = tf.data.TextLineDataset(data_path)

seq_list=[]
exp_list=[]
pbar = tqdm(total=len(df))

for i in dataset.as_numpy_iterator():
    seq, exp = i.decode("utf-8").split("\t")
    seq_list.append(list(seq))
    exp_list.append(float(exp))
    pbar.update(1)

pbar.close()

## preprocess raw data
## pad sequence to 112 
pad_seq_list = tf.keras.preprocessing.sequence.pad_sequences(seq_list, maxlen=112, padding="post", truncating='post', dtype="str", value="N")

## save the data
pickle.dump(pad_seq_list, open(path+"/preprocessed_data/seq_list", "wb"))
pickle.dump(exp_list, open(path+"/preprocessed_data/exp_list", "wb"))

saved_tf_dataset_path = path+"/preprocessed_data/"

pad_seq_list = pickle.load(open(saved_tf_dataset_path+"seq_list", "rb"))
exp_list = pickle.load(open(saved_tf_dataset_path+"exp_list", "rb"))

dataset = tf.data.Dataset.from_tensor_slices((pad_seq_list,exp_list))

vocab = ['A','C','G','T']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
dataset = dataset.map(lambda x, y: (lookup(x), y))
dataset = dataset.map(lambda x, y: (tf.cast(x, dtype=tf.float32), y))

data_size = len(dataset)
train_size = int(data_size*0.8)
val_size = int(data_size*0.2)

dataset = dataset.shuffle(100000, reshuffle_each_iteration=False)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

tf.data.experimental.save(train_dataset, os.path.join(saved_tf_dataset_path + "train_dataset"))
tf.data.experimental.save(val_dataset, os.path.join(saved_tf_dataset_path + "val_dataset"))
