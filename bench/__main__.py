import os
import argparse
import json
from utils import get_input_sample


def main(
    model_path, backend, repeat=1000, number=1, warmup=100, device="cpu", quantize=False
):
    if backend == "onnxruntime":
        from bench_onnxruntime import benchmark_onnxruntime

        res = benchmark_onnxruntime(model_path, repeat, number, warmup, quantize)
    elif backend == "pytorch":
        import onnx
        from onnx2pytorch import ConvertModel
        from bench_pytorch import benchmark_pytorch

        onnx_model = onnx.load(model_path)
        model = ConvertModel(onnx_model)
        input_sample = get_input_sample(onnx_model, backend=backend)
        res = benchmark_pytorch(
            model,
            input_sample,
            device=device,
            repeat=repeat,
            number=number,
            warmup=warmup,
            quantize=quantize,
        )
    elif backend == "tf":
        assert not quantize, "Use tf-lite to quantize tensorflow models."
        import onnx
        from onnx_tf.backend import prepare
        from bench_tf import benchmark_tf

        onnx_model = onnx.load(model_path)
        input_sample = get_input_sample(onnx_model, backend=backend)
        model = prepare(onnx_model)
        res = benchmark_tf(
            model, input_sample, repeat=repeat, number=number, warmup=warmup
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
    parser.add_argument("--quantize", type=bool, default=False, help="Quantize or not")
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Benchmark warmup repeats that are discarded",
    )
    parser.add_argument(
        "--backend-meta", type=str, default="", help="Additional backend info"
    )
    parser.add_argument(
        "--output-path", type=str, default="", help="Where to store results"
    )

    args = parser.parse_args()

    res = main(
        args.model_path,
        args.backend,
        args.repeat,
        args.number,
        args.warmup,
        args.device,
        args.quantize,
    )

    out = dict(args._get_args())
    out.update(dict(args._get_kwargs()))
    out.update(res)

    output_path = args.output_path
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            print(json.dumps(out), file=f)
    else:
        print(json.dumps(out))
