import argparse
import json
from utils import get_inputs_size


def main(model_path, backend, repeat=1000, number=1, warmup=100, device="cpu"):
    if backend == "onnxruntime":
        from bench_onnxruntime import benchmark_onnxruntime

        res = benchmark_onnxruntime(model_path, repeat, number, warmup)
    elif backend == "pytorch":
        import onnx
        from onnx2pytorch import ConvertModel
        from bench_pytorch import benchmark_pytorch

        onnx_model = onnx.load(model_path)
        model = ConvertModel(onnx_model)
        input_sizes = get_inputs_size(onnx_model)
        input_size = tuple([x.dim_value for x in input_sizes[0]])
        res = benchmark_pytorch(
            model,
            input_size,
            device=device,
            repeat=repeat,
            number=number,
            warmup=warmup,
        )
    elif backend == "tf":
        import onnx
        from onnx_tf.backend import prepare
        from bench_tf import benchmark_tf

        onnx_model = onnx.load(model_path)
        input_sizes = get_inputs_size(onnx_model)
        input_size = tuple([x.dim_value for x in input_sizes[0]])
        model = prepare(onnx_model)
        res = benchmark_tf(
            model, input_size, repeat=repeat, number=number, warmup=warmup
        )
    else:
        raise ValueError("backend {} does not exists".format(backend))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark model and backend.")
    parser.add_argument("model_path", metavar="M", type=str, help="Path to model")
    parser.add_argument("--backend", metavar="B", type=str, help="DNN backend")
    parser.add_argument("--device", type=str, default="cpu", help="Device backend")
    parser.add_argument("--repeat", type=int, default=1000, help="Benchmark repeats")
    parser.add_argument("--number", type=int, default=1, help="Benchmark number")
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Benchmark warmup repeats that are discarded",
    )
    parser.add_argument(
        "--backend-meta", type=str, default="", help="Additional backend info"
    )

    args = parser.parse_args()

    res = main(
        args.model_path,
        args.backend,
        args.repeat,
        args.number,
        args.warmup,
        args.device,
    )

    out = dict(args._get_args())
    out.update(dict(args._get_kwargs()))
    out.update(res)
    print(json.dumps(out))
