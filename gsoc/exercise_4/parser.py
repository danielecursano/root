from hls4ml.model import ModelGraph
from hls4ml.model.layers import Layer

    
def get_expected_attributes(layer):
    """
    Returns only attributes defined as expected bythe abstract layer class, 
    excluding attributes from the backends.
    """
    return {k.name: layer.attributes[k.name] for k in layer.expected_attributes if k.name in layer.attributes}

def parse(layer: Layer):
    """
    Returns general informations about the given layer.
    """
    return {
            "name": layer.name,
            "type": layer.class_name,
            "inputs": layer.inputs,
            "outputs": layer.outputs,
            "attributes": dict(layer.attributes)
        }
        
def get_model_config(model: ModelGraph):
    """
    Returns the model configuration.
    """
    model_config = {"model_name": model.config.get_project_name(), "layers": [], "input_shapes": [], "input_names": []}
    for layer in model.get_layers():
        layer_info = parse(layer)
        if layer_info["type"] == "Input":
            # Extract input tensor name and shape from the input layer
            model_config["input_shapes"].append(layer_info["attributes"]["input_shape"])
            model_config["input_names"].append(layer_info["outputs"])
            continue
        model_config["layers"].append(layer_info)
    return model_config 
