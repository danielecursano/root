from ROOT.TMVA.Experimental import SOFIE
import numpy as np
import os
import stat
from pathlib import Path

from hls4ml.writer.writers import Writer

class OperatorNotImplemented(Exception):
    def __init__(self, layer):
        super().__init__(f"Operator {layer.attributes.get('class_name')} not implemented")

def MakeActivation(layer, rmodel):
    return to_ROperator(layer, rmodel, name=layer.attributes.get('activation'))
                                                                       
def MakeRelu(layer, rmodel):
    return SOFIE.ROperator_Relu("float")(layer.inputs[0], layer.outputs[0])
    
def MakeElu(layer, rmodel):
    return SOFIE.ROperator_Elu("float")(layer.attributes.get("activ_param", 1.0), layer.inputs[0], layer.outputs[0])
    
def MakeDense(layer, rmodel):
    attr_alpha = 1.0
    attr_beta = 1.0
    attr_transA = 0
    attr_transB = 0
    fKernelName = f"{layer.name}/kernel"
    fBiasName = f"{layer.name}/bias"
    return SOFIE.ROperator_Gemm["float"](   
                attr_alpha, attr_beta, attr_transA, attr_transB, layer.inputs[0], fKernelName, fBiasName, layer.outputs[0]
    )
    
def MakeReshape(layer, rmodel):
    fOpMode = SOFIE.ReshapeOpMode.Reshape
    fNameShape = layer.name + "_shape"
    shape = layer.attributes.get("target_shape")
    rmodel.AddInitializedTensor["int64_t"](f"{layer.name}_shape", [len(shape)], np.asarray(shape).data)
    return SOFIE.ROperator_Reshape(fOpMode, 0, layer.inputs[0], fNameShape, layer.outputs[0])

def MakeConcat(layer, rmodel):
    return SOFIE.ROperator_Concat(layer.inputs, layer.attributes.get("axis"), 0, layer.outputs[0])

str2method = {"activation": MakeActivation, 
                "relu": MakeRelu, 
                "parametrizedactivation": MakeActivation, 
                "elu": MakeElu, 
                "dense": MakeDense,
                "reshape": MakeReshape,
                "concatenate": MakeConcat}

def to_ROperator(layer, rmodel, name=None):
    if layer.attributes.get("class_name") == "InputLayer":
        return None
        
    if name is None:
        name = layer.attributes.get("class_name")

    method =  str2method.get(name.lower())
    if method is None:
        raise OperatorNotImplemented(layer)
    return method(layer, rmodel)
        
class SofieWriter(Writer):

    def write_project_dir(self, model):
        if not os.path.isdir(f'{model.config.get_output_dir()}/firmware'):
            os.makedirs(f'{model.config.get_output_dir()}/firmware')

    def write_build_script(self, model):
        filedir = Path(__file__)
        # build_lib.sh
        build_lib_src = (filedir / '../templates/build_lib.sh').resolve()
        build_lib_dst = Path(f'{model.config.get_output_dir()}/build_lib.sh').resolve()
        with open(build_lib_src) as src, open(build_lib_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('mystamp', model.config.get_config_value('Stamp'))

                dst.write(line)
        build_lib_dst.chmod(build_lib_dst.stat().st_mode | stat.S_IEXEC)
        
    def write_bridge(self, model):
        filedir = Path(__file__)
        model_inputs = [k.name for k in model.get_input_variables()]

        bridge_src = (filedir / '../templates/myproject.cpp').resolve()
        bridge_dst = Path(f'{model.config.get_output_dir()}/{model.config.get_project_name()}.cpp').resolve()

        dat_path = "" 
        if any(f.endswith(".dat") for f in os.listdir(f'{model.config.get_output_dir()}/')):
            dat_path = f"{model.config.get_output_dir()}/{model.config.get_project_name()}.dat"

        with open(bridge_src) as src, open(bridge_dst, 'w') as dst:
            for line in src.readlines():
                line = line.replace('myproject', model.config.get_project_name())
                line = line.replace('//insert_inputs', ",".join([f"const float* {inp}" for inp in model_inputs]))
                line = line.replace('//insert_ref_inputs', ",".join([f"{inp}" for inp in model_inputs]))
                line = line.replace('//insert_dat_path', dat_path)
                dst.write(line)
    
    def write_rmodel(self, model):
        for input_layer in model.get_input_variables():
            self.rmodel.AddInputTensorInfo(input_layer.name, SOFIE.ConvertStringToType("float"), input_layer.shape)
            self.rmodel.AddInputTensorName(input_layer.name)

        self.rmodel.AddOutputTensorNameList(list(model.get_layers())[-1].outputs)

        for layer in model.get_layers():
            weight = layer.attributes.get("weight_data")
            bias = layer.attributes.get("bias_data")
            if weight is not None:
                self.rmodel.AddInitializedTensor["float"](f"{layer.name}/kernel", weight.shape, weight.flatten())
            if bias is not None:
                self.rmodel.AddInitializedTensor["float"](f"{layer.name}/bias", bias.shape, bias.flatten())
            op = to_ROperator(layer, self.rmodel)
            if op is not None:
                self.rmodel.AddOperatorReference(op)
        
    def write(self, model):
        self.rmodel = SOFIE.RModel.RModel(model.config.get_project_name())
        self.write_rmodel(model)
        self.write_project_dir(model)
        self.rmodel.Generate()
        self.rmodel.OutputGenerated(f"./{model.config.get_output_dir()}/{model.config.get_project_name()}.hxx")
        self.write_build_script(model)
        self.write_bridge(model)
