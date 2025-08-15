FROM mambaorg/micromamba:1.5.6-bullseye
LABEL maintainer="SEMCOG"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    build-essential \
    wget \
    curl \
    libsndfile1-dev \
    tesseract-ocr \
    espeak-ng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create micromamba env with Python 3.9
RUN micromamba create -y -n forecast python=3.9 -c conda-forge

# Set ENV and activate shell
ENV PATH=/opt/conda/envs/forecast/bin:$PATH
ENV CONDA_DEFAULT_ENV=forecast
ENV CONDA_PREFIX=/opt/conda/envs/forecast
SHELL ["/bin/bash", "-c"]

# Optional for interactive shell convenience
RUN echo "source activate forecast" >> ~/.bashrc

# Copy Python requirements
COPY requirements.txt /tmp/requirements.txt

# Install dependencies in the forecast env
RUN micromamba run -n forecast pip install --no-cache-dir --upgrade pip && \
    micromamba run -n forecast pip install --no-cache-dir -r /tmp/requirements.txt

CMD [ "bash" ]
