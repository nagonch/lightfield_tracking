FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git build-essential \
    libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
RUN pip3 install \
    open3d==0.19.0 \
    opencv-python==4.12.0.88 \
    viser==1.0.4 \
    scipy \
    einops \
    kornia \
    yacs \
    pyyaml \
    tqdm

WORKDIR /workspace
CMD ["bash"]
