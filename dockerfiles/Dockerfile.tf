FROM tensorflow/tensorflow:latest

RUN apt-get install -y git
RUN pip install --no-cache git+https://github.com/onnx/onnx-tensorflow.git