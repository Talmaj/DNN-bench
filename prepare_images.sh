#!/usr/bin/env bash

arch=${1:-cpu}

if [[ $arch == cpu ]]; then
  # CPU images
  docker build -f dockerfiles/Dockerfile.tf -t tensorflow:latest .
  docker build -f dockerfiles/Dockerfile.pytorch -t pytorch:latest .
  #docker pull mcr.microsoft.com/azureml/onnxruntime:latest-cuda
  docker build -f dockerfiles/Dockerfile.onnxruntime -t onnxruntime:latest .
  docker pull mcr.microsoft.com/azureml/onnxruntime:latest-openvino-cpu
elif [[ $arch == gpu ]]; then
  # GPU images
  docker pull mcr.microsoft.com/azureml/onnxruntime:latest-cuda
  docker pull mcr.microsoft.com/azureml/onnxruntime:latest-tensorrt
  docker build -f dockerfiles/Dockerfile.tf-gpu -t tensorflow:latest-gpu .
elif [[ $arch == arm ]]; then
  # ARM images
  docker pull toriml/onnxruntime:arm64v8
else
  echo "Select one of the following architectures: cpu, gpu or arm"
fi