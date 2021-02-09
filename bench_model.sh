#!/usr/bin/env bash

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
              --output)          output=${VALUE}  ;;
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
output=${output:-$(pwd)/results}
name=$(echo $(basename $model) | cut -f1 -d .)

cwd=$(pwd)

if [[ -n $pytorch ]]; then
  echo PyTorch
  docker run --rm -v $cwd/bench:/bench -v $model:$model -v $output:$output pytorch:latest \
  python3 /bench $model --backend pytorch --repeat $repeat --number $number --warmup $warmup \
  --backend-meta onnx2pytorch --device $device \
  --output-path $output/"$name"-pytorch-onnx2pytorch.json
fi

if [[ -n $onnxruntime ]]; then
  echo OpenMP
  docker run --rm -v $cwd/bench:/bench -v $model:$model -v $output:$output mcr.microsoft.com/azureml/onnxruntime:latest \
  python3 /bench $model --backend onnxruntime --repeat $repeat --number $number --warmup $warmup \
  --backend-meta openmp --device $device \
  --output-path $output/"$name"-onnxruntime-openmp.json
fi

if [[ -n $openvino ]]; then
  echo OpenVino
  docker run --rm -v $cwd/bench:/bench -v $model:$model -v $output:$output mcr.microsoft.com/azureml/onnxruntime:latest-openvino-cpu \
  python3 /bench $model --backend onnxruntime --repeat $repeat --number $number --warmup $warmup \
  --backend-meta openvino --device $device \
  --output-path $output/"$name"-onnxruntime-openvino.json
fi

if [[ -n $tvm || -n $nuphar ]]; then
  echo Nuphar
  docker run --rm -v $cwd/bench:/bench -v $model:$model -v $output:$output mcr.microsoft.com/azureml/onnxruntime:latest-nuphar \
  python3 /bench $model --backend onnxruntime --repeat $repeat --number $number --warmup $warmup \
  --backend-meta nuphar --device $device \
  --output-path $output/"$name"-onnxruntime-nuphar.json
fi

if [[ -n $tf ]]; then
  echo TensorFlow
  docker run --rm -v $cwd/bench:/bench -v $model:$model -v $output:$output tensorflow:latest \
  python3 /bench $model --backend tf --repeat $repeat --number $number --warmup $warmup \
  --backend-meta onnx_tf --device $device \
  --output-path $output/"$name"-tf-onnx_tf.json
fi
