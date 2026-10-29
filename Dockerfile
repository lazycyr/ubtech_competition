ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG APP_DIR=/workspace/GlobalHumanoidRobotChallenge_2026_Baseline
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG CN_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    XDG_RUNTIME_DIR=/tmp \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=${APP_DIR}

USER root

RUN mkdir -p /var/lib/apt/lists/partial \
    && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . ${APP_DIR}

# Install common dependencies into Isaac Sim's Python environment.
# Keep numpy and packaging pinned/limited to avoid breaking Isaac Sim.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -i ${PIP_INDEX_URL} \
    "cmake>=3.29.0.1" \
    "datasets>=2.19.0" \
    "deepdiff>=7.0.1" \
    "diffusers>=0.27.2" \
    "draccus>=0.10.0" \
    "einops>=0.8.0" \
    "flask>=3.0.3" \
    "gdown>=5.1.0" \
    "gymnasium==0.29.1" \
    "h5py>=3.10.0" \
    "huggingface-hub>=0.27.1" \
    "imageio[ffmpeg]>=2.34.0" \
    "jsonlines>=4.0.0" \
    "numba>=0.59.0" \
    "omegaconf>=2.3.0" \
    "opencv-python>=4.9.0" \
    "packaging==23.0" \
    "av>=12.0.5,<13.0.0" \
    "pymunk>=6.6.0" \
    "pyzmq>=26.2.1" \
    "termcolor>=2.4.0" \
    "pyserial>=3.5,<4.0" \
    "rerun-sdk>=0.24.0,<0.27.0" \
    "peft>=0.18.0,<1.0.0" \
    "wandb>=0.16.3" \
    "zarr>=2.17.0" \
    "safetensors" \
    "regex" \
    "python-xlib" \
    "pin" \
    "numpy<1.27,>=1.22" \
    "transformers>=4.48.0" \
    "accelerate>=1.10.0,<2.0.0" \
    "num2words" \
    "pytest>=8.1.0" \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -i ${PIP_INDEX_URL} --no-deps "pynput>=1.7.7"

WORKDIR ${APP_DIR}

# Avoid LeRobot dependency constraints from upgrading/downgrading Isaac Sim core deps.
RUN if [[ -f pyproject.toml ]]; then \
        sed -i 's/numpy[^"]*/numpy/g' pyproject.toml; \
        sed -i 's/packaging[^"]*/packaging/g' pyproject.toml; \
    else \
        echo "ERROR: pyproject.toml not found in ${APP_DIR}" >&2; \
        exit 1; \
    fi

# Reinstall low-level keyboard dependency before editable install.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -i ${CN_PIP_INDEX_URL} evdev-binary

# Safe editable install after removing risky numpy / packaging version constraints.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -i ${CN_PIP_INDEX_URL} -e ${APP_DIR}

RUN /isaac-sim/python.sh -m pip uninstall -y \
    numpy \
    numba \
    lxml \
    usd-core \
    usd-core-parser \
    docstring-parser \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -i ${PIP_INDEX_URL} \
    numpy==1.26.4 \
    lxml==4.9.3 \
    "usd-core>=25.2.post1,<26.0" \
    docstring-parser==0.16 \
    pyjwt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN /isaac-sim/python.sh -m pip uninstall -y torchcodec \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -i ${PIP_INDEX_URL} torchcodec==0.5.0

RUN apt-get update && apt-get install -y evtest
WORKDIR ${APP_DIR}

CMD ["/bin/bash"]