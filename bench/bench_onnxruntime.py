import os

import numpy as np
from onnx import version_converter
import onnxruntime as ort

from utils import get_type, get_shape
from bench import benchmark_speed


def benchmark_onnxruntime(
    path_to_model, repeat=1000, number=1, warmup=100, quantize=False
):
    """
    Parameters
    ----------
    path_to_model: str or onnx.ModelProto
        Path to an onnx model.
    repeat: int
        Repetition of experiment. Default: 1000
    number: int
        Number of forward passes in each experiment. Default: 1
    warmup: int
        Number of disregarded experiments. Default: 100
    quantize: bool
        Dynamically quantize the model with default parameters.

    Returns
    -------
    info: dict
        Information about the size and min, max, mean, std of the time
        of the experiments.
    """
    assert repeat >= 2 * warmup

    if quantize:
        import onnx
        from onnxruntime.quantization import quantize_dynamic

        orig_model = onnx.load(path_to_model)
        if orig_model.opset_import[0].version < 11:
            converted_model = version_converter.convert_version(orig_model, 11)
            path_to_model = '/tmp/model_conv.onnx'
            with open(path_to_model, 'wb') as f:
                f.write(converted_model.SerializeToString())
            del orig_model, converted_model
        path_to_quant_model = "/tmp/model_quant.onnx"
        model = quantize_dynamic(path_to_model, path_to_quant_model)
        size = os.path.getsize(path_to_quant_model)
        sess = ort.InferenceSession(path_to_quant_model)
    else:
        size = os.path.getsize(path_to_model)
        sess = ort.InferenceSession(path_to_model)

    inputs = {
        x.name: np.random.randn(*get_shape(x)).astype(get_type(x))
        for x in sess.get_inputs()
    }

    def _benchmark():
        output = sess.run(None, inputs)

    res = dict(size=size, input_size=[tuple(x.shape) for x in inputs.values()])
    res.update(benchmark_speed(_benchmark, repeat, number, warmup))
    return res
