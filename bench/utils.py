def get_type(x):
    """
    Extract type from onnxruntime input.

    Parameters
    ----------
    x: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg
    """
    if x.type.startswith("tensor"):
        typ = x.type[7:-1]
    else:
        raise NotImplementedError("For type: {}".format(x.type))

    if typ == "float":
        typ = "float32"
    elif typ == "double":
        typ = "float64"
    return typ


def get_shape(x, unknown_dim_size=1):
    """
    Extract shape from onnxruntime input.
    Replace unknown dimension by default with 1.

    Parameters
    ----------
    x: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg
    unknown_dim_size: int
        Default: 1
    """
    shape = x.shape
    # replace unknown dimensions by default with 1
    shape = [i if isinstance(i, int) else unknown_dim_size for i in shape]
    return shape


def get_inputs_size(onnx_model):
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    input_sizes = [
        x.type.tensor_type.shape.dim
        for x in onnx_model.graph.input
        if x.name in net_feed_input
    ]
    return input_sizes
