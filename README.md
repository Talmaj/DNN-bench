# DNN Bench
![GitHub](https://img.shields.io/github/license/ToriML/dnn-bench)
[![ToriML](https://circleci.com/gh/ToriML/DNN-bench.svg?style=shield)](https://app.circleci.com/pipelines/github/ToriML/DNN-bench)

DNN Bench is a library that lets you benchmark your deep learning models against various frameworks and 
backends, with a single command.

With DNN Bench you can answer questions like:
- to which hardware should I deploy my model?
- which backend should I use?
- should I apply an optimisation technique, e.g. quantisation, before I deploy it?

The goal is to make it easy for developers to choose the most optimal deployment configuration (optimization on/off, backend, hardware) for their particular use-cases.

Side note: Models are benchmarked within docker containers.

## Example
Performance of 
[BERT-Squad](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad)
and
[ResNet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
on 
[c5a.4xlarge](https://aws.amazon.com/about-aws/whats-new/2020/06/now-available-amazon-ec2-c5a-instances-featuring-2nd-generation-amd-epyc-processors/),
an AWS EC2 CPU compute instance. 
It shows number of processed samples per second, where more is better.

![Bert-CPU](docs/figs/bertsquad-10.onnx-c5a.4xlarge.png)
![Resnet-CPU](docs/figs/resnet50v1.onnx-c5a.4xlarge.png)

See further [analysis](docs/analysis.md) for more models benchmarked on different hardware.

## Supported devices and backends

|        | PyTorch      | TensorFlow | ONNX-Runtime  | OpenVINO* | Nuphar*  | CUDA*  | TensorRT* |
|--------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
|  CPU   | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |                    |                    |
|  GPU   | :white_check_mark: | :white_check_mark: |                    |                    |                    | :white_check_mark: | :white_check_mark: |
|  ARM   |                    |                    | :white_check_mark: |                    |                    |                    |                    |

*Marked backends are executed within ONNX-Runtime framework.

## Installation
### Dependencies
#### Ubuntu
```
./install_dependencies.sh cpu
```
Replace `cpu` argument with `gpu` for nvidia-docker.

#### Other
- Install [docker](https://docs.docker.com/get-docker/).
- Install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Add yourself to [docker group](https://docs.docker.com/engine/install/linux-postinstall/)
  `sudo usermod -aG docker $USER` to run docker commands without sudo.

### Deep learning backends 
You can use pre-compiled images from dockerhub. 
They will be downloaded automatically when running `./bench_model.sh`  

Optional.  
Prepare docker images for various deep learning backends locally.
```
./prepare_images.sh cpu
```
Replace `cpu` argument with `gpu` for gpu backends or `arm` for arm backends.

## Usage
Benchmark an onnx model against different backends:
```
./bench_model.sh path_to_model --repeat=100 --number=1 --warmup=10 --device=cpu \
--tf --onnxruntime --openvino --pytorch --nuphar
```
Possible backends:
```
  --tf              (with --device=cpu or gpu)
  --onnxruntime     (with --device=cpu or arm)
  --openvino        (with --device=cpu)
  --pytorch         (with --device=cpu or gpu)
  --nuphar          (with --device=cpu)
  --ort-cuda        (with --device=gpu)
  --ort-tensorrt    (with --device=gpu)
```

Additional Parameters:
```
  --output   OUTPUT       Directory of benchmarking results. Default: ./results
  --repeat   REPEAT       Benchmark repeats. Default: 1000
  --number   NUMBER       Benchmark number. Default: 1
  --warmup   WARMUP       Benchmark warmup repeats that are discarded. Default: 100
  --device   DEVICE       Device backend: CPU or GPU or ARM. Default: CPU
  --quantize              Dynamic quantization in a corresponding backend.
```

### Results
Results are stored by default to `./results` directory. Each benchmarking result
is stored in a json format.

```
{
   'model_path': '/models/efficientnet-lite4.onnx',
   'output_path': '/results/efficientnet-lite4-onnxruntime-openvino.json',
   'backend': 'onnxruntime',
   'backend_meta': 'openvino',
   'device': 'cpu',
   'number': 1,
   'repeat': 100,
   'warmup': 10,
   'size': 51946641,
   'input_size': [[1, 224, 224, 3]],
   'min': 0.038544699986232445,
   'max': 0.05930669998633675,
   'mean': 0.04293907555596282,
   'std': 0.0039751552053260125,
   'data': [0.04748649999964982,
            0.05760759999975562, ... ]
}
```
- __model_path__: path to the input model
- __output_path__: path to the results file
- __backend__: deep learning backend used to produce the results
- __backend_meta__: special parameters used with the backend. 
  Example: onnxruntime used with openvino.
- __device__: gpu, cpu, arm, etc. where the model was benchmarked.
- __number__: Number of inferences in a _single_ experiment.
- __repeat__: Number of repeated experiments.
- __warmup__: Number of discarded experiments. Reasoning:
  inference might not reach its optimal performance in the first few runs.
- __size__: Size of the model in bytes.
- __min__: Minimum time of an experiment run.
- __max__: Maximum time of an experiment run.
- __mean__: Mean time of an experiment run.
- __std__: Standard deviation of an experiment run.
- __data__: All measurements of the experiment runs.

### Plotting
A simple plotting utility to generate quick plots is available in 
[plot_results.py](./vis/plot_results.py).
- Dependencies:  
```pip install seaborn matplotlib pandas```
- Usage:  
```python vis/plot_results.py results_dir plots_dir```


## Limitations and known issues
- `--quantize` flag not supported for `--ort-cuda`, `--ort-tensorrt` and `--tf`
- Current version supports onnx models only. To convert models from other frameworks  
  follow [these](https://github.com/onnx/tutorials#converting-to-onnx-format) examples.
- The following docker images for CPU execution utilize only half of the CPUs on Linux
  ec2 instances: 
  - onnxruntime with openvino,
  - pytorch
- onnxruntime with nuphar utilizes total count of CPUs - 1 on Linux
  ec2 instances.
  
## Troubleshoot
- If running tensorflow image fails due to onnx-tf conversion, 
  re-build the image locally:
  ```docker build -f dockerfiles/Dockerfile.tf -t toriml/tensorflow:latest .```
- If you have permission errors to run docker, add yourself to docker group
  `sudo usermod -aG docker $USER` and re-login `su - $USER`.
