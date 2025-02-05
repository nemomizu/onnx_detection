

# FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# =====================
# Initialize
# =====================

SHELL [ "/bin/bash", "-c"]
ENV LC_CTYPE="ja_JP.UTF-8" TZ="Asia/Tokyo"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

ARG main_repo_name="onnx_detection"
ARG main_repo_path="/home/nemoto/${main_repo_name}"
WORKDIR ${main_repo_path}

ENV http_proxy=http://proxy.mei.co.jp:8080/
ENV HTTP_PROXY=http://proxy.mei.co.jp:8080/
ENV ftp_proxy=http://proxy.mei.co.jp:8080/
ENV FTP_PROXY=http://proxy.mei.co.jp:8080/
ENV https_proxy=http://proxy.mei.co.jp:8080/
ENV HTTPS_PROXY=http://proxy.mei.co.jp:8080/

# Install packages
RUN set -x \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        gnupg2 \
        software-properties-common \
        language-pack-ja \
        tzdata \
        curl \
        lsb-release \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*
RUN set -x \
    # Python3.12をインストールするために、deadsnakes PPAを追加
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        # common tools
        python3.12-dev \
        bash-completion \
        build-essential \
        git \
        iputils-ping \
        jq \
        less \
        net-tools \
        openssh-client \
        sudo \
        tar \
        time \
        unzip \
        vim \
        wget \
        xz-utils \
        zip \
        connect-proxy \
        libsm6 \
        libxrender1 \
        libxext-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-dev \
        libxdamage1 \
        default-jre \
        graphviz \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
# Python / pip
RUN ln -s $(which python3.12) /usr/bin/python
RUN curl -sSL https://bootstrap.pypa.io/get-pip.py | python - --break-system-packages
# RUN echo `python --version`

COPY ${main_repo_name}/requirements.txt ${main_repo_path}/requirements.txt
# # Install requirement packages of `pasai-tfkeras`
RUN python -m pip install -r ${main_repo_path}/requirements.txt --break-system-packages