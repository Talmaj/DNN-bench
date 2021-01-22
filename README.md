# Tori Bench

A library to benchmark your deep learning model against various frameworks and 
backends.  
The model is benchmarked within the docker containers

## Installation
- Install [docker](https://docs.docker.com/get-docker/).
- Prepare docker images  
```
./prepare_images.sh
```

## Usage
Benchmark an onnx model against different backends:
```
./bench_model.sh path_to_model --repeat=100 --number=1 --warmup=10 --device=cpu \
--tf --onnxruntime --openvino --pytorch --nuphar
```
Parameters:
```
  --repeat   REPEAT       Benchmark repeats
  --number   NUMBER       Benchmark number
  --warmup   WARMUP       Benchmark warmup repeats that are discarded
  --device   DEVICE       Device backend, CPU or CUDA
  --quantize QUANTIZE     Dynamic quantization in a corresponding backend
```

# Limitations
`--device=GPU` and `--quantize` is not supported yet.