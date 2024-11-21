# # Use the NVIDIA CUDA base image
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# # Set the working directory
# WORKDIR /app

# # Install Anaconda
# RUN apt-get update && apt-get install -y wget bzip2 && \
#     wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh && \
#     bash Anaconda3-2023.03-Linux-x86_64.sh -b -p /opt/conda && \
#     rm Anaconda3-2023.03-Linux-x86_64.sh && \
#     /opt/conda/bin/conda clean -afy

# # Update PATH environment variable
# ENV PATH=/opt/conda/bin:$PATH

# # Copy the environment.yml file into the Docker image
# COPY swLorebotLinux.yml /app/environment.yml

# # Create the conda environment from the .yml file and clean up
# RUN conda env create -f /app/environment.yml && conda clean -afy

# # Activate the environment and ensure the environment is activated in the shell
# RUN echo "conda activate swLorebot" >> ~/.bashrc
# ENV PATH=/opt/conda/envs/swLorebot/bin:$PATH

# # Install NVIDIA Container Toolkit
# RUN apt-get update && apt-get install -y nvidia-container-toolkit

# # Copy the necessary files into the Docker image
# COPY . /app

# # Set the entry point to run the script
# ENTRYPOINT ["python", "swBotAPI_Runpod.py"]


# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
RUN ls -l /bin
RUN ls -l /usr/bin
RUN echo "PATH: $PATH"
# Set the working directory
WORKDIR /app

# Install Miniconda and bash
RUN apt-get update && apt-get install -y wget bzip2 bash && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean -afy

# Ensure Conda is initialized and available in all shells
ENV PATH=/opt/conda/bin:$PATH

# Install conda-lock
RUN pip install conda-lock

# Copy the conda-lock file
COPY conda-lock.yml /app/conda-lock.yml

# Install dependencies using conda-lock
RUN conda-lock install --name swLorebot && conda clean -afy

# Ensure the environment is in PATH
ENV PATH=/opt/conda/envs/swLorebot/bin:$PATH

# Copy the application files
COPY . /app

# Debug PATH during runtime
RUN echo "Final PATH: $PATH"
RUN bash --version || echo "Bash is not installed"

# Set the entry point to run the script
# ENTRYPOINT ["bash", "-c", "source /opt/conda/bin/activate swLorebot && python swBotAPI_Runpod.py"]
ENTRYPOINT ["sh", "-c", "/opt/conda/bin/activate swLorebot && python swBotAPI_Runpod.py"]
