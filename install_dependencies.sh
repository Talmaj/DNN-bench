#!/usr/bin/env bash
# Ubuntu installation guide
device=${1:-cpu}

# Install docker if not installed already
if ! command -v docker &> /dev/null; then
  echo "Install docker"
  curl https://get.docker.com | sh \
    && sudo systemctl --now enable docker
fi

if [[ $device == gpu ]]; then
  # Install nvidia-docker
  echo "Install nvidia-docker"
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
     && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update
  sudo apt-get install -y nvidia-docker2
  sudo systemctl restart docker

  echo "Docker with gpu support installed."
else
  echo "Docker installed"
fi
