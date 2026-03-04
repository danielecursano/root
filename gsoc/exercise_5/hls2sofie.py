from ROOT.TMVA.Experimental import SOFIE
import numpy as np

class OperatorNotImplemented(Exception):
    def __init__(self, layer):
        super().__init__(f"Operator {layer.get('type')} not implemented")
        
def MakeActivation(layer, rmodel):
    return to_ROperator(layer, rmodel, name=layer["attributes"]["activation"])
                                                                       
def MakeRelu(layer, rmodel):
    return SOFIE.ROperator_Relu("float")(layer["inputs"][0], layer["outputs"][0])
    
def MakeElu(layer, rmodel):
    return SOFIE.ROperator_Elu("float")(layer["attributes"].get("activ_param", 1.0), layer["inputs"][0], layer["outputs"][0])
    
def MakeDense(layer, rmodel):
    attr_alpha = 1.0
    attr_beta = 1.0
    attr_transA = 0
    attr_transB = 0
    fKernelName = f"{layer['name']}/kernel"
    fBiasName = f"{layer['name']}/bias"
    return SOFIE.ROperator_Gemm["float"](   
                attr_alpha, attr_beta, attr_transA, attr_transB, layer["inputs"][0], fKernelName, fBiasName, layer["outputs"][0]
    )
    
def MakeReshape(layer, rmodel):
    fOpMode = SOFIE.ReshapeOpMode.Reshape
    fNameShape = layer["name"] + "_shape"
    shape = layer["attributes"].get("target_shape")
    rmodel.AddInitializedTensor["int64_t"](f"{layer['name']}_shape", [len(shape)], np.asarray(shape).data)
    return SOFIE.ROperator_Reshape(fOpMode, 0, layer["inputs"][0], fNameShape, layer["outputs"][0])

def MakeConcat(layer, rmodel):
    return SOFIE.ROperator_Concat(layer["inputs"], layer["attributes"]["axis"], 0, layer["outputs"][0])

str2method = {"Activation": MakeActivation, 
                "relu": MakeRelu, 
                "ParametrizedActivation": MakeActivation, 
                "elu": MakeElu, 
                "Dense": MakeDense,
                "Reshape": MakeReshape,
                "Concatenate": MakeConcat}

def to_ROperator(layer, rmodel, name=None):
    if layer["type"] == "Input":
        return None
        
    if name is None:
        name = layer.get("type")
        
    method =  str2method.get(name)
    if method is None:
        raise OperatorNotImplemented(layer)
    return method(layer, rmodel)
    
def generate_sofie_model(hls_config):
    if len(hls_config["layers"]) == 0:
        raise ValueError("Model must contain at least one layer")
        
    rmodel = SOFIE.RModel.RModel(hls_config["model_name"])
    
    # config inputs
    for inp_name, inp_shape in zip(hls_config["input_names"], hls_config["input_shapes"]):
        rmodel.AddInputTensorInfo(inp_name[0], SOFIE.ConvertStringToType("float"), inp_shape)
        rmodel.AddInputTensorName(inp_name[0])
    
    # config outputs
    rmodel.AddOutputTensorNameList(hls_config["layers"][-1]["outputs"])
    
    for layer in hls_config["layers"]:
        weight = layer["attributes"].get("weight_data")
        bias = layer["attributes"].get("bias_data")
        if weight is not None:
            rmodel.AddInitializedTensor["float"](f"{layer['name']}/kernel", weight.shape, weight.flatten())
        if bias is not None:
            rmodel.AddInitializedTensor["float"](f"{layer['name']}/bias", bias.shape, bias.flatten())
        #if layer["type"] == "Reshape":
         #   shape = layer["attributes"].get("target_shape")
          #  if shape:
           #     rmodel.AddInitializedTensor["int64_t"](f"{layer['name']}_shape", [len(shape)], np.asarray(shape).data)
        op = to_ROperator(layer, rmodel)
        if op is not None:
            rmodel.AddOperatorReference(op)
            
    return rmodel
