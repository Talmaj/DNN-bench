import numpy as np
import tensorflow as tf

from bench import benchmark_speed


def benchmark_tf(
    model,
    input_sample,
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
    input_sample: list[tf.Constant]]
        Size of the input sample.
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

    inputs = input_sample
    input_sizes = [tuple(x.shape) for x in input_sample]
    # inputs = [tf.constant(tf.random.uniform(i)) for i in input_sizes]
    input_dict = dict(zip(model.inputs, inputs))
    # size = len(model.tf_module)

    def _benchmark():
        output = model.tf_module(**input_dict)

    # res = dict(size=size)
    res = dict(input_size=input_sizes)
    res.update(benchmark_speed(_benchmark, repeat, number, warmup))
    return res
