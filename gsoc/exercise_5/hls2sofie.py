from ROOT.TMVA.Experimental import SOFIE

def MakeActivation(layer):
    return to_ROperator(layer, name=layer["attributes"]["activation"])
                                                                       
def MakeRelu(layer):
    return SOFIE.ROperator_Relu("float")(layer["inputs"][0], layer["outputs"][0])
    
def MakeElu(layer):
    return SOFIE.ROperator_Elu("float")(layer["attributes"].get("activ_param", 1.0), layer["inputs"][0], layer["outputs"][0])
    
str2method = {"Activation": MakeActivation, "relu": MakeRelu, "ParametrizedActivation": MakeActivation, "elu": MakeElu}

def to_ROperator(layer, name=None):
    if name is None:
        name = layer.get("type")
    return str2method.get(name, lambda _: None)(layer)

def generate_sofie_model(hls_config):
    rmodel = SOFIE.RModel.RModel()
    
    # config inputs
    rmodel.AddInputTensorInfo(hls_config["input_name"], SOFIE.ConvertStringToType("float"), hls_config["input_shape"])
    rmodel.AddInputTensorName(hls_config["input_name"])
    
    # config outputs
    rmodel.AddOutputTensorNameList(hls_config["layers"][-1]["outputs"])
    
    for layer in hls_config["layers"]:
        op = to_ROperator(layer)
        if op is None:
            continue
        rmodel.AddOperatorReference(op)
    return rmodel
