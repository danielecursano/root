from exercise_4 import get_model_config
from exercise_5 import generate_sofie_model

import hls4ml
import tensorflow as tf
from tensorflow import keras
import ROOT
import numpy as np
import torch
import torch.nn as nn

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
    ])),
    ("dense_elu_1d", "keras", keras.Sequential([
    keras.layers.InputLayer(input_shape=(5,)),
    keras.layers.Dense(1, activation="elu")
    ])),
    ("reshape", "keras", keras.Sequential([
    keras.layers.InputLayer(input_shape=(4,4)),
    keras.layers.Reshape((16,)),
    keras.layers.ELU()
    ])),
    ("dense_1d", "torch", DenseModel(input_shape=(1,))),
    ("relu_1d", "torch", ReluModel(input_shape=(1,))),
    ("elu_1d", "torch", EluModel(input_shape=(1,)))
]

def test_concat():
    python_model = concat_model()
    hls_config = hls4ml.utils.config_from_keras_model(python_model)
    hls_model = hls4ml.converters.convert_from_keras_model(python_model, hls_config=hls_config)
    
    model_config = get_model_config(hls_model)
    rmodel = generate_sofie_model(model_config)
    rmodel.Generate()
    rmodel.OutputGenerated()
    
    ROOT.gInterpreter.Declare(f'#include "myproject.hxx"')

    session = getattr(ROOT, f"TMVA_SOFIE_myproject").Session()
    
    x1 = np.random.rand(1, 4).astype(np.float32)
    x2 = np.random.rand(1, 3).astype(np.float32)
    
    sofie_pred = session.infer(x1.flatten(), x2.flatten())
    py_pred = python_model.predict([x1, x2])
    
    try:
        np.testing.assert_allclose(np.array(sofie_pred).flatten(), py_pred.flatten(), rtol=1e-6, atol=1e-7)
        print(f"Test concat passed")
    except AssertionError as e:
        print(f"Test concat failed")
        print(e)
    

def test_rmodel(name, framework, python_model):

    if framework == "keras":
        hls_config = hls4ml.utils.config_from_keras_model(python_model)
        hls_model = hls4ml.converters.convert_from_keras_model(python_model, hls_config=hls_config)
    elif framework == "torch":
        hls_config = hls4ml.utils.config.config_from_pytorch_model(python_model, python_model.input_shape)
        hls_model = hls4ml.converters.convert_from_pytorch_model(python_model, hls_config=hls_config)
    else:
        print(f"Test {name}_{framework} failed")
        print("{framework} not implemented")
        
    model_config = get_model_config(hls_model)
    
    model_config["model_name"] = name

    rmodel = generate_sofie_model(model_config)
    rmodel.Generate()
    rmodel.OutputGenerated()
    
    ROOT.gInterpreter.Declare(f'#include "{name}.hxx"')

    # Load the project namespace dynamically using a string.
    # This is required to get the correct session, because following
    # the tutorial methods directly may cause the session to read
    # an old .dat file from a previous project with the same name. 
    session = getattr(ROOT, f"TMVA_SOFIE_{name}").Session()
    
    input_shape = model_config["input_shapes"][0]

    x = np.random.rand(*input_shape).astype(np.float32)
    
    sofie_pred = session.infer(x)

    py_pred = python_model.predict(x.reshape(1, *input_shape))
    
    try:
        np.testing.assert_allclose(np.array(sofie_pred).flatten(), py_pred.flatten(), rtol=1e-6, atol=1e-7)
        print(f"Test {name}_{framework} passed")
    except AssertionError as e:
        print(f"Test {name}_{framework} failed")
        print(e)
    
if __name__ == "__main__":
    for test in TEST_MODELS:
        test_rmodel(*test)
    test_concat()
    
