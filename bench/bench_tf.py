import numpy as np
import tensorflow as tf

from bench import benchmark_speed


def benchmark_tf(
    model,
    input_size,
    input_dtype=tf.float32,
    # device=torch.device("cpu"),
    repeat=1000,
    number=1,
    warmup=100,
):
    """
    Parameters
    ----------
    model: nn.Module
        Converted tensorflow model from onnx.
    input_size: tuple[int]
        Size of the input sample.
    input_dtype: torch.dtype
        Dtype of the input sample. Default: np.float32
    device: torch.device
        Device on which to run the experiment. Default: torch.device('cpu')
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

    inputs = tf.constant(tf.random.uniform(input_size))
    input_dict = dict([(model.inputs[0], inputs)])
    # size = len(model.tf_module)

    def _benchmark():
        output = model.tf_module(**input_dict)

    # res = dict(size=size)
    res = dict()
    res.update(benchmark_speed(_benchmark, repeat, number, warmup))
    return res
