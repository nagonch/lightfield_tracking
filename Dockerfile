FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*


ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV CUDAARCHS="89"

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation

RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libglib2.0-0

RUN pip3 install plenpy==0.9.2 packaging==25.0 viser==1.0.4 opencv-python==4.12.0.88 open3d==0.19.0
COPY sam2/ sam2/
WORKDIR /sam2
RUN pip3 install -e .
WORKDIR /
RUN pip3 install transformers==4.57.3 nerfview==0.1.3 splines==0.3.3 e3nn==0.5.9 einops==0.8.1 easydict==1.13 kornia yacs
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
COPY Video-Depth-Anything/ Video-Depth-Anything/

CMD ["bash"]
