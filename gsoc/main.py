import hls4ml
from hls4ml.backends import register_backend
from hls4ml.writer import register_writer 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sofie_backend import SofieBackend, SofieWriter

register_writer('Sofie', SofieWriter)
register_backend('Sofie', SofieBackend)
    
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(10,)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Reshape((2, 2)),
    keras.layers.Activation("elu")
])

config = hls4ml.utils.config_from_keras_model(model, backend='Sofie')
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, backend='Sofie')

hls_model.compile()
x = np.random.rand(10).astype(np.float32)
y = hls_model.predict(x)
print(y)
print(model.predict(x.reshape((1, 10))))

# Alternative method to create the session once and reuse it
sofie_session = SofieBackend.get_sofie_session(hls_model)
print(sofie_session.infer(x))
