from exercise_4 import get_model_config
from exercise_5 import generate_sofie_model

import hls4ml
import tensorflow as tf
from tensorflow import keras
import ROOT
import numpy as np

TEST_MODELS = [
    ("dense_1", keras.Sequential([
    keras.layers.InputLayer(input_shape=(10,)),
    keras.layers.Dense(1, activation="relu")
    ]))
]

def test_rmodel(name, python_model):
    hls_config = hls4ml.utils.config_from_keras_model(python_model)
    hls_model = hls4ml.converters.convert_from_keras_model(python_model, hls_config=hls_config)
    
    model_config = get_model_config(hls_model)
    
    rmodel = generate_sofie_model(model_config)
    rmodel.Generate()
    rmodel.OutputGenerated()
    
    # hls4ml default project name is myproject
    ROOT.gInterpreter.Declare(f'#include "{model_config["model_name"]}.hxx"')
    session = ROOT.TMVA_SOFIE_myproject.Session()
    
    input_shape = model_config["input_shape"][0]
    x = np.random.rand(*input_shape).astype(np.float32)
    
    sofie_pred = session.infer(x)
    py_pred = model.predict(x.reshape(1, *input_shape))
    
    try:
        np.testing.assert_allclose(np.array(sofie_pred).flatten(), py_pred.flatten(), rtol=1e-6, atol=1e-7)
        print(f"Test passed for model: {name}")
    except AssertionError as e:
        print(f"Test Failed for model: {name}")
        print(e)
    
if __name__ == "__main__":
    for name, model in TEST_MODELS:
        test_rmodel(name, model)
    
