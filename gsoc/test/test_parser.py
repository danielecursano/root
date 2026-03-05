import hls4ml
import tensorflow as tf
from tensorflow import keras
import pytest

from exercise_4 import get_model_config

@pytest.mark.parametrize("shape", ([10, ], [2, 2], [1, 2, 3]))
def test_keras_sequential(shape):
    model = keras.Sequential([
        keras.layers.Input(shape=shape),
        keras.layers.Dense(32),
        keras.layers.Dense(16),
        keras.layers.Dense(1)
    ])

    config = hls4ml.utils.config_from_keras_model(model)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    ret_config = get_model_config(hls_model)

    assert ret_config["input_shapes"] == [shape]
    assert len(ret_config["layers"]) == len(model.layers), f'len(ret_config["layers"]) should be {len(model.layers)} != {len(ret_config["layers"])}'
    
