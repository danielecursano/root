import hls4ml
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(10,)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Reshape((2, 2)),
    keras.layers.Activation("elu")
])

config = hls4ml.utils.config_from_keras_model(model, backend='sofie')
hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, backend='sofie')

hls_model.write()
x = np.random.rand(10).astype(np.float32)
y = hls_model.predict(x)
print(y)
print(model.predict(x.reshape((1, 10))))

# Alternative method to create the session once and reuse it
sofie_session = hls4ml.backends.SofieBackend.get_sofie_session(hls_model)
print(sofie_session.infer(x))
