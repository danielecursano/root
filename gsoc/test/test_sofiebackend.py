from sofie_backend import SofieBackend, SofieWriter

import hls4ml
import tensorflow as tf
from tensorflow import keras
import ROOT
import numpy as np
import torch
import torch.nn as nn
import pytest
from hls4ml.backends import register_backend
from hls4ml.writer import register_writer 

register_writer('Sofie', SofieWriter)
register_backend('Sofie', SofieBackend)

def concat_model():
    input1 = keras.layers.Input(shape=(4,), name="input1")
    input2 = keras.layers.Input(shape=(3,), name="input2")

    x1 = keras.layers.Dense(8, activation='relu')(input1)
    x2 = keras.layers.Dense(8, activation='relu')(input2)

    merged = keras.layers.Concatenate(name="concat")([x1, x2])
    output = keras.layers.Dense(1)(merged)

    return keras.models.Model(inputs=[input1, input2], outputs=output)
    
class TorchModel(nn.Module):
    def __init__(self, input_shape):
        assert type(input_shape) == tuple
        super().__init__()
        self.input_shape = input_shape
    def forward(self, x):
        pass
    def predict(self, x):
        x = torch.from_numpy(x)
        return self.forward(x).detach().numpy()
   
class DenseModel(TorchModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)
        self.linear = nn.Linear(input_shape[0], 1)
        
    def forward(self, x):
        return self.linear(x)

class ReluModel(TorchModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)
    def forward(self, x):
        return nn.functional.relu(x)

class EluModel(TorchModel):
    def __init__(self, input_shape):
        super().__init__(input_shape)
    def forward(self, x):
        return nn.functional.elu(x)


TEST_MODELS = [
    ("dense_relu_1d", "keras", keras.Sequential([
    keras.layers.InputLayer(input_shape=(10,)),
    keras.layers.Dense(2, activation="relu"),
    keras.layers.Dense(1)
    ]), (10, )),
    ("dense_elu_1d", "keras", keras.Sequential([
    keras.layers.InputLayer(input_shape=(5,)),
    keras.layers.Dense(1, activation="elu")
    ]), (5, )),
    ("reshape", "keras", keras.Sequential([
    keras.layers.InputLayer(input_shape=(4,4)),
    keras.layers.Reshape((16,)),
    keras.layers.Dense(1),
    keras.layers.ELU()
    ]), (4, 4)),
    ("dense_1d", "torch", DenseModel(input_shape=(1,)), (1, )),
    ("relu_1d", "torch", ReluModel(input_shape=(1,)), (1, )),
    ("elu_1d", "torch", EluModel(input_shape=(1,)), (1, ))
]

def test_concat():
    python_model = concat_model()
    hls_config = hls4ml.utils.config_from_keras_model(python_model, backend="Sofie")
    hls_model = hls4ml.converters.convert_from_keras_model(python_model, hls_config=hls_config, backend="Sofie")
    hls_model.compile()
    
    x1 = np.random.rand(1, 4).astype(np.float32)
    x2 = np.random.rand(1, 3).astype(np.float32)
    
    sofie_pred = hls_model.predict([x1, x2])
    py_pred = python_model.predict([x1, x2])
    
    np.testing.assert_allclose(sofie_pred.flatten(), py_pred.flatten(), rtol=1e-6, atol=1e-7)

    
@pytest.mark.parametrize("name,framework,python_model,shape", TEST_MODELS)
def test_rmodel(name, framework, python_model, shape):
    output_dir = f"{name}_{framework}"
    if framework == "keras":
        hls_config = hls4ml.utils.config_from_keras_model(python_model, backend="Sofie")
        hls_model = hls4ml.converters.convert_from_keras_model(python_model, hls_config=hls_config, backend="Sofie", output_dir=output_dir)
    elif framework == "torch":
        hls_config = hls4ml.utils.config.config_from_pytorch_model(python_model, python_model.input_shape, backend="Sofie")
        hls_model = hls4ml.converters.convert_from_pytorch_model(python_model, hls_config=hls_config, backend="Sofie", output_dir=output_dir)
    
    # Necessary to prevent SOFIE from loading a .dat file from a project with the same name
    hls_model.config.config["ProjectName"] = name

    hls_model.compile()    

    x = np.random.rand(*shape).astype(np.float32)
    
    sofie_pred = hls_model.predict(x)

    py_pred = python_model.predict(x.reshape(1, *shape))
    
    np.testing.assert_allclose(np.array(sofie_pred).flatten(), py_pred.flatten(), rtol=1e-6, atol=1e-7)

    
