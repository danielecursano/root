from exercise_4 import get_model_config
from exercise_5 import generate_sofie_model

import hls4ml
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(10,)),
    keras.layers.ELU(alpha=1.2),
    keras.layers.Dense(1, activation="relu")
])

config = hls4ml.utils.config_from_keras_model(model)
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

model_config = get_model_config(hls_model)

rmodel = generate_sofie_model(model_config)
rmodel.Generate()
rmodel.OutputGenerated()
rmodel.PrintGenerated()
