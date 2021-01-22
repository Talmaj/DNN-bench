import torch

from bench import benchmark_speed


def benchmark_pytorch(
    model,
    input_size,
    input_dtype=torch.float32,
    device=torch.device("cpu"),
    repeat=1000,
    number=1,
    warmup=100,
):
    """
    Parameters
    ----------
    model: nn.Module
        Pytorch model.
    input_size: tuple[int]
        Size of the input sample.
    input_dtype: torch.dtype
        Dtype of the input sample. Default: torch.float32
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
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    model.to(device)

    inputs = torch.rand(input_size, device=device, dtype=input_dtype)
    model = torch.jit.trace(model, inputs)
    size = len(model.save_to_buffer())

    def _benchmark():
        with torch.no_grad():
            output = model(inputs)

    res = dict(size=size)
    res.update(benchmark_speed(_benchmark, repeat, number, warmup))
    return res
