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


def get_shape_onnx(x, unknown_dim_size=1):
    """
    Extract shape from onnx input.
    Replace unknown dimension by default with 1.

    Parameters
    ----------
    x: onnx.onnx_ONNX_REL_1_7_ml_pb2.ValueInfoProto
    unknown_dim_size: int
        Default: 1
    """
    shape = x.type.tensor_type.shape.dim
    # replace unknown dimensions by default with 1
    shape = [i.dim_value if i.dim_value else unknown_dim_size for i in shape]
    return shape


def get_type_onnx(x):
    """
    Extract type from onnx input.

    Parameters
    ----------
    x: onnx.onnx_ONNX_REL_1_7_ml_pb2.ValueInfoProto
    """
    typ = x.type.tensor_type.elem_type
    if typ in [1, 6]:  # from onnx.AttributeProto
        typ = "float32"
    elif typ in [2, 7]:  # from onnx.AttributeProto
        typ = "int32"
    else:
        raise NotImplementedError("For type: {}".format(x))
    return typ


def get_input_sample(onnx_model, backend="numpy"):
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    input_sizes = [
        get_shape_onnx(x) for x in onnx_model.graph.input if x.name in net_feed_input
    ]
    input_types = [
        get_type_onnx(x) for x in onnx_model.graph.input if x.name in net_feed_input
    ]

    if backend == "numpy":
        import numpy as np

        input_sample = [
            np.random.rand(*shape).astype(getattr(np, dtype))
            for shape, dtype in zip(input_sizes, input_types)
        ]
    elif backend == "pytorch":
        import torch

        input_sample = [
            torch.rand(shape).to(getattr(torch, dtype))
            for shape, dtype in zip(input_sizes, input_types)
        ]
    elif backend == "tf":
        import tensorflow as tf

        input_sample = [
            tf.constant(tf.cast(tf.random.uniform(shape), dtype=getattr(tf, dtype)))
            for shape, dtype in zip(input_sizes, input_types)
        ]
    else:
        raise KeyError("Backend {} does not exists.".format(backend))

    return input_sample
