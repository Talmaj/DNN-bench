#!/usr/bin/env bash

docker build -f dockerfiles/Dockerfile.tf -t tensorflow:latest .
docker build -f dockerfiles/Dockerfile.pytorch -t pytorch:latest .
#docker pull mcr.microsoft.com/azureml/onnxruntime:latest-cuda
docker build -f dockerfiles/Dockerfile.onnxruntime -t onnxruntime:latest .
docker pull mcr.microsoft.com/azureml/onnxruntime:latest-openvino-cpu