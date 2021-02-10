import torch

from bench import benchmark_speed


def benchmark_pytorch(
    model,
    input_sample,
    device=torch.device("cpu"),
    repeat=1000,
    number=1,
    warmup=100,
    quantize=False,
):
    """
    Parameters
    ----------
    model: nn.Module
        Pytorch model.
    input_sample: list[torch.Tensor]
        List of the input tensors.
    device: torch.device
        Device on which to run the experiment. Default: torch.device('cpu')
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
    if isinstance(device, str):
        device = torch.device(device)
    if quantize:
        model = torch.quantization.quantize_dynamic(model)
    model.eval()
    model.to(device)

    inputs = [x.to(device) for x in input_sample]
    input_sizes = [tuple(x.shape) for x in input_sample]
    model = torch.jit.trace(model, inputs, check_trace=False)
    size = len(model.save_to_buffer())

    def _benchmark():
        with torch.no_grad():
            output = model(*inputs)

    res = dict(size=size, input_size=input_sizes)
    res.update(benchmark_speed(_benchmark, repeat, number, warmup))
    return res
