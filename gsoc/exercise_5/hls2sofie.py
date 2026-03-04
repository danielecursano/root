from ROOT.TMVA.Experimental import SOFIE
import numpy as np

class OperatorNotImplemented(Exception):
    def __init__(self, layer):
        super().__init__(f"Operator {layer.get('type')} not implemented")
        
def MakeActivation(layer):
    return to_ROperator(layer, name=layer["attributes"]["activation"])
                                                                       
def MakeRelu(layer):
    return SOFIE.ROperator_Relu("float")(layer["inputs"][0], layer["outputs"][0])
    
def MakeElu(layer):
    return SOFIE.ROperator_Elu("float")(layer["attributes"].get("activ_param", 1.0), layer["inputs"][0], layer["outputs"][0])
    
def MakeDense(layer):
    attr_alpha = 1.0
    attr_beta = 1.0
    attr_transA = 0
    attr_transB = 0
    fKernelName = f"{layer['name']}/kernel"
    fBiasName = f"{layer['name']}/bias"
    return SOFIE.ROperator_Gemm["float"](   
                attr_alpha, attr_beta, attr_transA, attr_transB, layer["inputs"][0], fKernelName, fBiasName, layer["outputs"][0]
    )
    
def MakeReshape(layer):
    fOpMode = SOFIE.ReshapeOpMode.Reshape
    fNameShape = layer["name"] + "_shape"
    op = SOFIE.ROperator_Reshape(fOpMode, 0, layer["inputs"][0], fNameShape, layer["outputs"][0])
    return op
    
str2method = {"Activation": MakeActivation, 
                "relu": MakeRelu, 
                "ParametrizedActivation": MakeActivation, 
                "elu": MakeElu, 
                "Dense": MakeDense,
                "Reshape": MakeReshape}

def to_ROperator(layer, name=None):
    if layer["type"] == "Input":
        return None
        
    if name is None:
        name = layer.get("type")
        
    method =  str2method.get(name)
    if method is None:
        raise OperatorNotImplemented(layer)
    return method(layer)
    
def generate_sofie_model(hls_config):
    rmodel = SOFIE.RModel.RModel(hls_config["model_name"])
    
    # config inputs
    rmodel.AddInputTensorInfo(hls_config["input_name"], SOFIE.ConvertStringToType("float"), hls_config["input_shape"])
    rmodel.AddInputTensorName(hls_config["input_name"])
    
    # config outputs
    rmodel.AddOutputTensorNameList(hls_config["layers"][-1]["outputs"])
    
    for layer in hls_config["layers"]:
        weight = layer["attributes"].get("weight_data")
        bias = layer["attributes"].get("bias_data")
        if weight is not None:
            rmodel.AddInitializedTensor["float"](f"{layer['name']}/kernel", weight.shape, weight.flatten())
        if bias is not None:
            rmodel.AddInitializedTensor["float"](f"{layer['name']}/bias", bias.shape, bias.flatten())
        if layer["type"] == "Reshape":
            shape = layer["attributes"].get("target_shape")
            if shape:
                rmodel.AddInitializedTensor["int64_t"](f"{layer['name']}_shape", [len(shape)], np.asarray(shape).data)
        op = to_ROperator(layer)
        if op is not None:
            rmodel.AddOperatorReference(op)
            
    return rmodel
