import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd


url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
unzipped_filename = "training.1600000.processed.noemoticon.csv"


def load_sentiment140(shuffle_size=10000, batch_size=64, validation_size=1000, epochs=5):
    tf.keras.utils.get_file(
        "trainingandtestdata.zip", url, extract=True, cache_dir=".", cache_subdir=""
    )
    dataset = pd.read_csv(unzipped_filename, encoding="latin-1", header=None)[[0, 5]]
    dataset[0] = dataset[0] / 4. # scale 0 to 1
    dataset = dataset.sample(frac=1) # Shuffle

    target = dataset.pop(0)
    dataset = tf.data.Dataset.from_tensor_slices((dataset.values.reshape(-1,), target.values))
    dataset = dataset.shuffle(shuffle_size).repeat(epochs).batch(batch_size)

    validation_batches = validation_size // batch_size

    validation_dataset= dataset.take(validation_batches)
    train_dataset = dataset.skip(validation_batches)
    return train_dataset, validation_dataset
