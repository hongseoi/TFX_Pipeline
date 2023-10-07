import argparse
import math
import os
from typing import Tuple

import datasets
import numpy as np
import tensorflow as tf
import tqdm

RESOLUTION = 256

def load_nsmc_dataset(args):
    # huggingface의 dataset 불러오기
    hf_dataset_identifier = 'nsmc'
    ds = datasets.load_dataset(hf_dataset_identifier)

    ds = ds.shuffle(seed=1)
    ds = ds['train'].train_test_split(test_size=args.split, seed=ards.seed)
    train_ds = ds['train']
    val_ds = ds['test']

    return train_ds, val_ds

def create_tfrecord(text: str, id: int, label: int):
    text = _bytes_feature(text.encode())

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "id": _int64_feature(id),
                "text": text,
                "label": _int64_feature(label),
            }
        )
    ).SerializeToString()


def _bytes_feature(value: bytes):
    """Creates a bytes TFRecord feature.

    Args:
        value: A bytes value.

    Returns:
        A TFRecord feature.
    """

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

import pandas as pd
import tensorflow as tf

def write_tfrecords(root_dir, dataset, split, batch_size=None, resize=None):
    print(f"Preparing TFRecords for split: {split}.")

    for step in tqdm.tnrange(int(math.ceil(len(dataset) / batch_size))):
        temp_ds = dataset[step * batch_size : (step + 1) * batch_size]

        filename = os.path.join(
            root_dir, "{}-{:02d}-{}.tfrec".format(split, step, len(temp_ds))
        )

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(len(temp_ds)):
                text = temp_ds["text"][i]
                id = temp_ds["id"][i]
                label = temp_ds["label"][i]

                example = create_tfrecord(text, id, label)
                out_file.write(example)
            print("Wrote file {} containing {} records".format(filename, len(temp_ds)))


def _bytes_feature(value: bytes):
    """Creates a bytes TFRecord feature.

    Args:
        value: A bytes value.

    Returns:
        A TFRecord feature.
    """

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecords(root_dir, dataset, split, batch_size=None, resize=None):
    print(f"Preparing TFRecords for split: {split}.")

    for step in tqdm.tnrange(int(math.ceil(len(dataset) / batch_size))):
        temp_ds = dataset[step * batch_size : (step + 1) * batch_size]

        filename = os.path.join(
            root_dir, "{}-{:02d}-{}.tfrec".format(split, step, len(temp_ds))
        )

        with tf.io.TFRecordWriter(filename) as out_file:
            for i in range(len(temp_ds)):
                text = temp_ds["text"][i]
                id = temp_ds["id"][i]
                label = temp_ds["label"][i]

                example = create_tfrecord(text, id, label)
                out_file.write(example)
            print("Wrote file {} containing {} records".format(filename, len(temp_ds)))

def main(args):
    train_ds, val_ds = load_nsmc_dataset(args)
    print('dataset loaded from hf')

    if not os.path.exists(args.root_tfrecord_dir):
        os.makedirs(args.root_tfrecord_dir, exist_ok=True)

    write_tfrecords(
        args.root_tfrecord_dir, train_ds, "train", args.batch_size
    )

    write_tfrecords(
        args.root_tfrecord_dir, val_ds, "val", args.batch_size
    )    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", help="Train and test split.", default=0.2, type=float
    )
    parser.add_argument(
        "--seed",
        help="Seed to be used while performing train-test splits.",
        default=2022,
        type=int,
    )
    parser.add_argument(
        "--root_tfrecord_dir",
        help="Root directory where the TFRecord shards will be serialized.",
        default="sidewalks-tfrecords",
        type=str,
    )
    parser.add_argument(
        "--batch_size",
        help="Number of samples to process in a batch before serializing a single TFRecord shard.",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--resize",
        help="Width and height size the image will be resized to. No resizing will be applied when this isn't set.",
        type=int,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)