import hls4ml
import tensorflow as tf
from tensorflow import keras

from parser import get_model_config
    
def test_keras_sequential():
    model = keras.Sequential([
        keras.layers.Input(shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    config = hls4ml.utils.config_from_keras_model(model)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)

    ret_config = get_model_config(hls_model)
    
    assert len(ret_config) == 7
    
if __name__ == "__main__":
    test_keras_sequential()
