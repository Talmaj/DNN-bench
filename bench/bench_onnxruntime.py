import os

import numpy as np
import onnxruntime as ort

from utils import get_type, get_shape
from bench import benchmark_speed


def benchmark_onnxruntime(path_or_model, repeat=1000, number=1, warmup=100):
    """
    Parameters
    ----------
    path_or_model: str or onnx.ModelProto
        Path to an onnx model or a loaded onnx model.
    repeat: int
        Repetition of experiment. Default: 1000
    number: int
        Number of forward passes in each experiment. Default: 1
    warmup: int
        Number of disregarded experiments. Default: 100

    Returns
    -------
    info: dict
        Information about the size and min, max, mean, std of the time
        of the experiments.
    """
    assert repeat >= 2 * warmup

    if isinstance(path_or_model, str):
        size = os.path.getsize(path_or_model)
        sess = ort.InferenceSession(path_or_model)
    # elif isinstance(path_or_model, onnx.ModelProto):
    #     bitstream = path_or_model.SerializeToString()
    #     size = len(bitstream)
    #     sess = ort.InferenceSession(bitstream)
    #     del bitstream
    else:
        raise TypeError

    inputs = {
        x.name: np.random.randn(*get_shape(x)).astype(get_type(x))
        for x in sess.get_inputs()
    }

    def _benchmark():
        output = sess.run(None, inputs)

    res = dict(size=size, input_size=[tuple(x.shape) for x in inputs.values()])
    res.update(benchmark_speed(_benchmark, repeat, number, warmup))
    return res
