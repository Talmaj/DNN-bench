import timeit


def benchmark_speed(benchmark_func, repeat=1000, number=1, warmup=100):
    out = timeit.repeat(benchmark_func, repeat=repeat, number=number)

    # remove warmup
    out = out[warmup:]

    # calculate statistics
    mean = sum(out) / len(out)
    std = (sum((x - mean) ** 2 for x in out) / len(out)) ** 0.5
    return dict(min=min(out), max=max(out), mean=mean, std=std)
