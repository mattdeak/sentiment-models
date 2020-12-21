import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

from data_utils import load_sentiment140
from bert_models import map_name_to_handle, map_model_to_preprocess


def build_bert_model(bert_model_name):
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name="inputs")
    preprocessing_layer = hub.KerasLayer(
        map_model_to_preprocess[bert_model_name], name="preprocessing"
    )

    encoder_inputs = preprocessing_layer(input_layer)
    bert_model = hub.KerasLayer(
        map_name_to_handle[bert_model_name], name="BERT_encoder"
    )
    outputs = bert_model(encoder_inputs)

    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name="classifier")(net)
    return tf.keras.Model(input_layer, net)
