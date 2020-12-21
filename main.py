import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

from bert import build_bert_model
from bert_models import map_model_to_preprocess, map_name_to_handle
from data_utils import load_sentiment140

# TODO: Use args to collect
bert_model_name = "bert_en_uncased_L-12_H-768_A-12"

BATCH_SIZE = 64
EPOCHS = 5

def main():

    train_ds, val_ds = load_sentiment140(batch_size=BATCH_SIZE, epochs=EPOCHS)
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()

    init_lr = 3e-5
    # optimizer = optimization.create_optimizer(
    #     init_lr=init_lr,
    #     num_train_steps=num_train_steps,
    #     num_warmup_steps=num_warmup_steps,
    #     optimizer_type="adam",
    # )

    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr)

    # loss = tf.keras.losses.MSE()
    # metrics = tf.metrics.MSE()

    model = build_bert_model(bert_model_name)

    model.compile(optimizer=optimizer, loss='mse', metrics='mse')
    model.fit(train_ds, validation_data=val_ds, steps_per_epoch=steps_per_epoch)
    model.save('models/test_models.model')

if __name__ == "__main__":
    main()
