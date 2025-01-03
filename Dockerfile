# Use the NVIDIA CUDA base image
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
# WORKDIR /app

# Install Miniconda and bash
# RUN apt-get update && apt-get install -y wget bzip2 bash && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
#     rm Miniconda3-latest-Linux-x86_64.sh && \
#     /opt/conda/bin/conda clean -afy

# Ensure Conda is initialized and available in all shells
# ENV PATH=/opt/conda/bin:$PATH

# Install conda-lock
# RUN pip install conda-lock

# Copy the conda-lock file
# COPY conda-lock.yml /app/conda-lock.yml

# Install dependencies using conda-lock
# RUN conda-lock install --name swLorebot && conda clean -afy

# Ensure the environment is in PATH
# ENV PATH=/opt/conda/envs/swLorebot/bin:$PATH

# Copy the application files
# COPY . /app

# Set the entry point to run the script
# ENTRYPOINT ["sh", "-c", "/opt/conda/bin/activate swLorebot && python swBotAPI_Runpod.py"]
# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set the working directory
WORKDIR /app

# Install Miniconda, bash, and conda-lock in a single RUN command
RUN apt-get update && apt-get install -y wget bzip2 bash && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/pip install --no-cache-dir conda-lock

# Ensure Conda is initialized and available in all shells
ENV PATH=/opt/conda/bin:$PATH

# Copy the conda-lock file and install dependencies
COPY conda-lock.yml /app/conda-lock.yml
RUN conda-lock install --name swLorebot && conda clean -afy

# Ensure the environment is in PATH
ENV PATH=/opt/conda/envs/swLorebot/bin:$PATH

# Copy the application files
COPY . /app

# Set the entry point to run the script
ENTRYPOINT ["sh", "-c", "/opt/conda/bin/activate swLorebot && python swBotAPI_Runpod.py"]