#!/bin/bash

for ARGUMENT in "$@"
do
    # keyword arguments
    if [[ $ARGUMENT = --* ]]; then
      KEY=$(echo $ARGUMENT | cut -f1 -d=)
      VALUE=$(echo $ARGUMENT | cut -f2 -d=)
      case "$KEY" in
              --tf)              tf=1 ;;
              --pytorch)         pytorch=1 ;;
              --onnxruntime)     onnxruntime=1 ;;
              --openvino)        openvino=1 ;;
              --nuphar)          nuphar=1 ;;
              --tvm)             tvm=1 ;;
              --repeat)          repeat=${VALUE} ;;
              --number)          number=${VALUE}  ;;
              --warmup)          warmup=${VALUE}  ;;
              --device)          device=${VALUE}  ;;
              *)
      esac
    else
      # positional arguments
      model=$ARGUMENT
    fi
done

# default parameters
repeat=${repeat:-1000}
number=${number:-1}
warmup=${warmup:-100}
device=${device:-cpu}

cwd=$(pwd)

if [[ -n $pytorch ]]; then
  echo PyTorch
  docker run --rm -v $cwd/bench:/bench -v $model:$model pytorch:latest \
  python3 /bench $model --backend pytorch --repeat $repeat --number $number --warmup $warmup \
  --backend-meta onnx2pytorch --device $device
fi

if [[ -n $onnxruntime ]]; then
  echo OpenMP
  docker run --rm -v $cwd/bench:/bench -v $model:$model mcr.microsoft.com/azureml/onnxruntime:latest \
  python3 /bench $model --backend onnxruntime --repeat $repeat --number $number --warmup $warmup \
  --backend-meta openmp --device $device
fi

if [[ -n $openvino ]]; then
  echo OpenVino
  docker run --rm -v $cwd/bench:/bench -v $model:$model mcr.microsoft.com/azureml/onnxruntime:latest-openvino-cpu \
  python3 /bench $model --backend onnxruntime --repeat $repeat --number $number --warmup $warmup \
  --backend-meta openvino --device $device
fi

if [[ -n $tvm || -n $nuphar ]]; then
  echo Nuphar
  docker run --rm -v $cwd/bench:/bench -v $model:$model mcr.microsoft.com/azureml/onnxruntime:latest-nuphar \
  python3 /bench $model --backend onnxruntime --repeat $repeat --number $number --warmup $warmup \
  --backend-meta nuphar --device $device
fi

if [[ -n $tf ]]; then
  echo TensorFlow
  docker run --rm -v $cwd/bench:/bench -v $model:$model tensorflow:latest \
  python3 /bench $model --backend tf --repeat $repeat --number $number --warmup $warmup \
  --backend-meta onnx_tf --device $device
fi
