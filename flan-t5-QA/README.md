# Flan-T5 Fine-Tuning for Question Answering

This repository contains a professional setup for fine-tuning the `google/flan-t5-small` model for question-answering (QA) tasks using the SQuAD v1.1 dataset. The project is containerized with Docker, leveraging CUDA for GPU-accelerated training, making it portable across local machines and HPC clusters.

## Project Overview
- **Task**: Fine-tune `google/flan-t5-small`, a lightweight, instruction-tuned T5 model, for extractive question answering on the SQuAD v1.1 dataset.
- **Dataset**: SQuAD v1.1 (Stanford Question Answering Dataset), containing 87,599 training and 10,570 validation question-answer pairs based on Wikipedia articles.
- **Environment**: Docker with PyTorch 2.1.0, CUDA 12.1, and Hugging Face libraries (`transformers`, `datasets`).
- **Output**: Fine-tuned model weights and training logs saved to the `output` directory.

## Directory Structure
```
flan-t5-QA/
├── data/
│   ├── train.jsonl       # SQuAD v1.1 training data (downloaded when run train.py)
│   ├── validation.jsonl  # SQuAD v1.1 validation data (downloaded when run train.py)
├── output/
│   ├── model/            # Fine-tuned model weights
│   ├── logs/             # Training logs
├── train.py              # Fine-tuning script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Dockerfile for CUDA-enabled environment
├── README.md             # This file
```

## Prerequisites
- **Docker**: Installed with NVIDIA Container Toolkit for GPU support.
- **Hardware**: A CUDA-compatible GPU (e.g., NVIDIA) for training.
- **Storage**: ~10GB for the Docker image and dataset/model outputs.

## Setup and Usage

### 1. Build the Docker Image
```bash
docker build -t flan-t5-qa .
```
- Builds the image `flan-t5-qa` with PyTorch, CUDA, and dependencies.

### 2. Run the Container
```bash
docker run --rm -v $(pwd)/output:/app/output --gpus all flan-t5-qa
```
- Creates a container, runs `train.py`, and saves outputs to `./output`.
- `--rm`: Deletes the container after execution.
- `-v $(pwd)/output:/app/output`: Mounts the local `output` folder to save model weights/logs.
- `--gpus all`: Enables GPU acceleration.

### 3. View Results
- Check `./output/model` for fine-tuned model weights.
- Check `./output/logs` for training logs (e.g., loss curves).

### 4. Deploy to HPC
- Push to Docker Hub:
  ```bash
  docker tag flan-t5-qa yourusername/flan-t5-qa:latest
  docker push yourusername/flan-t5-qa:latest
  ```
- Pull and run on HPC:
  ```bash
  docker pull yourusername/flan-t5-qa:latest
  docker run --rm -v /hpc/output:/app/output --gpus all yourusername/flan-t5-qa
  ```
- For Singularity (common on HPC):
  ```bash
  singularity run --nv docker://yourusername/flan-t5-qa
  ```

## Dockerfile Explanation
The `Dockerfile` sets up a CUDA-enabled environment for training:
```Dockerfile
# Use PyTorch with CUDA and cuDNN for GPU support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py /app/train.py

# Command to run training
CMD ["python", "train.py"]
```
- **FROM**: Uses `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`, which includes Python 3.9, PyTorch 2.1.0, CUDA 12.1, and cuDNN 8.
- **WORKDIR**: Sets `/app` as the working directory.
- **COPY requirements.txt**: Copies the dependency list.
- **RUN pip install**: Installs `transformers`, `datasets`, and other libraries.
- **COPY train.py**: Copies the training script.
- **CMD**: Runs `python train.py` when the container starts.

## Notes
- The `train.py` script downloads SQuAD v1.1 automatically via the `datasets` library. The dataset is downloaded to a temporary cache in the container’s filesystem (managed by the `datasets` library, typically in `/root/.cache/huggingface/datasets`).
- Training hyperparameters (e.g., batch size, epochs) are set in `train.py` and can be adjusted.
- For larger models (e.g., `flan-t5-base`), ensure sufficient GPU memory or modify the batch size.
- To extend the project, add datasets to `./data` and mount them with `-v $(pwd)/data:/app/data`.