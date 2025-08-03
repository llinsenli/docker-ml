# Docker for Machine Learning: Transformer-Based Projects

This repository provides a template and guide for using Docker to run transformer-based machine learning projects, leveraging a PyTorch environment. It includes explanations of Docker concepts, motivations for using Docker in ML, a project structure with a Dockerfile with common Docker commands, and a real LLM project for Question Answer.

## Part 1: Docker Basics

### What is Docker?
Docker is a platform for containerization, allowing you to package applications and their dependencies into portable, isolated environments called **containers**. Containers ensure consistent execution across different systems (e.g., local machine, HPC, cloud).

### Key Concepts
- **Image**: A read-only template, like a class in programming, containing the application, dependencies, and configuration. For example, a PyTorch image includes Python, PyTorch, and libraries like `transformers`.
- **Container**: A runnable instance of an image, like an object instantiated from a class. Containers are lightweight, isolated environments that run your code. One image can spawn multiple containers.
- **Registry**: A storage and distribution system for Docker images, such as Docker Hub or private registries. You can pull pre-built images (e.g., `pytorch/pytorch`) or push your own images to share them.

### Docker Lifecycle
1. **Build an Image**: Use a `Dockerfile` to define the environment (e.g., base image, dependencies, code). Run `docker build` to create the image. The `docker build` command executes all instructions in the `Dockerfile` except CMD, building the image. These instructions set up the image’s filesystem and environment.
2. **Store the Image**: Images are stored locally in Docker’s storage (e.g., `/var/lib/docker` on Linux). Optionally, push to a registry like Docker Hub.
3. **Run a Container**: Use `docker run` to create and start a container from an pre-built image by executes the `CMD` in `Dockerfile`. The container executes the code and may create files (e.g., model weights) or read in dataset. Without Volume Mounts `-v`, outputs are stored in the container’s `/app/output` and are lost with `--rm` or require manual extraction from a stopped container. With Volume Mounts `-v`, outputs are saved directly to the host’s `./output` directory and persist after the container is deleted.
4. **Manage Containers**: Containers stop when their main process ends. You can delete them (`docker rm`), restart them (`docker start`), or auto-delete after execution (`--rm`). 
5. **Share Images**: Push images to a registry for use on other machines (e.g., HPC) or export/import as `.tar` files.

## Part 2: Why Use Docker for Machine Learning?

### Motivation
Docker is particularly valuable for machine learning, especially for transformer-based tasks (e.g., fine-tuning large language models like BERT). Here’s why:

- **Dependency Management**: ML frameworks like PyTorch and libraries like `transformers` have complex dependencies (e.g., CUDA, Python versions). Docker packages everything into an image, avoiding installation conflicts on your local machine or HPC.
- **Portability**: Build and test an image locally, then run it on a virtual device, cloud, or HPC cluster without worrying about environment differences.
- **Consistency**: Ensure the same environment (e.g., PyTorch 2.1.0, `transformers`) across development, testing, and production.
- **Isolation**: Run multiple experiments with different library versions in separate containers without conflicts.
- **GPU Support**: Docker supports NVIDIA GPUs via the NVIDIA Container Toolkit, ideal for training transformer models on virtual devices or HPC.
- **Reproducibility**: Share images via registries (e.g., Docker Hub) to reproduce experiments exactly, crucial for ML research.

### Use Case: Training Transformers
For transformer-based tasks (e.g., fine-tuning a model), Docker enables:
- **Local Development**: Test your code in a container without installing Python or `transformers` locally.
- **HPC Deployment**: Run the same image on an HPC cluster with GPU access, mounting large datasets.
- **Model Training**: Train models in containers, saving outputs (e.g., model weights) to local or mounted directories.
- **Scalability**: Use orchestration tools (e.g., Kubernetes) for distributed training.

## Part 3: Project Directory Template and Dockerfile

### Project Directory Structure
This template is for a PyTorch-based machine learning project, such as fine-tuning a transformer model.

```
ml-transformer-project/
├── data/
│   ├── train.csv       # Training dataset
│   ├── test.csv        # Test dataset
├── output/
│   ├── model/          # Directory for model weights
│   ├── logs/           # Directory for training logs
├── train.py            # Script for training the model
├── requirements.txt     # Python dependencies
├── Dockerfile          # Dockerfile to build the image
└── README.md           # This file
```

- **data/**: Contains datasets (e.g., CSV files for training/testing).
- **output/**: Stores model weights and logs, populated when training.
- **train.py**: Main script for training (e.g., fine-tuning a transformer model).
- **requirements.txt**: Lists Python packages (e.g., `transformers`, `datasets`).
- **Dockerfile**: Defines the image for the project.

### Dockerfile Template
```Dockerfile
# Use PyTorch with GPU support as the base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and data
COPY train.py /app/train.py
COPY data /app/data

# Command to run training
CMD ["python", "train.py"]
```

### Dockerfile Explanation
- **FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime**:
  - Specifies the base image, which includes:
    - Python (e.g., 3.9).
    - PyTorch 2.1.0 with CUDA 12.1 and cuDNN 8 for GPU support.
    - A minimal Linux environment (Ubuntu-based).
  - Provides the foundation for ML tasks without needing local Python installation.
- **WORKDIR /app**:
  - Sets `/app` as the working directory for subsequent commands and container runtime.
  - All files copied to the image (e.g., `train.py`) are placed in `/app`.
- **COPY requirements.txt /app/requirements.txt**:
  - Copies the `requirements.txt` file to `/app/requirements.txt` in the image.
  - Lists dependencies (e.g., `transformers==4.35.2`, `datasets==2.14.5`).
- **RUN pip install --no-cache-dir -r requirements.txt**:
  - Executes during image build, installing packages listed in `requirements.txt`.
  - `--no-cache-dir` reduces image size by not storing pip cache.
  - Adds libraries like `transformers` to the image’s Python environment.
- **COPY train.py /app/train.py**:
  - Copies the training script to `/app/train.py` in the image.
- **COPY data /app/data**:
  - Copies the `data` directory (e.g., `train.csv`, `test.csv`) to `/app/data` in the image.
  - This is only suggest when the dataset is small. For large datasets, avoid `COPY` and use volume mounts to access datasets stored on the host.
- **CMD ["python", "train.py"]**:
  - Specifies the default command to run when a container starts.
  - Executes `python train.py` in the container, starting the training process.
  - Doesn’t run during image build; it’s metadata for containers.

### Example `requirements.txt`
```
transformers==4.35.2
datasets==2.14.5
huggingface_hub==0.17.3
```

## Part 4: Common Docker Commands

Below are common Docker commands with examples tied to the project structure above.

### 1. Build an Image
- **Command**: `docker build -t <image-name> .`
- **Purpose**: Builds an image from the `Dockerfile` in the current directory.
- **Example**:
  ```bash
  docker build -t ml-transformer .
  ```
  - Builds an image named `ml-transformer` from the `Dockerfile`.
  - Installs `transformers`, copies `train.py` and `data`, and sets up the environment.

### 2. Run a Container
- **Command**: `docker run [--rm] [-v <local-path>:<container-path>] [--gpus all] <image-name>`
- **Purpose**: Creates and starts a container from an image, optionally mounting directories or enabling GPUs.
- **Example**:
  ```bash
  docker run --rm -v $(pwd)/output:/app/output --gpus all ml-transformer
  docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output --gpus all flan-t5-qa
  ```
  - Creates a container from `ml-transformer`.
  - `--rm`: Deletes the container after execution.
  - `-v $(pwd)/output:/app/output`: Mounts the local `output` folder to `/app/output` in the container to save model weights/logs. The training output will store locally. 
  - `-v $(pwd)/data:/app/data`: Mounts local `./data` (containing train.jsonl, validation.jsonl) to `/app/data` in the container. This accesses dataset from local `./data` keeps the image lightweight when the dataset is large. 
  - `--gpus all`: Enables GPU access for training.
  - Runs `python train.py` (from `CMD`).

### 3. List Images
- **Command**: `docker image ls`
- **Purpose**: Lists locally stored images.
- **Example**:
  ```bash
  docker image ls
  ```
  - Output:
    ```
    REPOSITORY       TAG       IMAGE ID       CREATED        SIZE
    ml-transformer   latest    abc123def456   5 minutes ago  6.5GB
    ```

### 4. List Containers
- **Command**: `docker container ls [-a]`
- **Purpose**: Lists running containers (`docker container ls`) or all containers, including stopped ones (`-a`).
- **Example**:
  ```bash
  docker container ls -a
  ```
  - Shows stopped containers (if `--rm` wasn’t used):
    ```
    CONTAINER ID   IMAGE           COMMAND          STATUS
    xyz789         ml-transformer  "python train.py" Exited (0)
    ```

### 5. Delete a Container
- **Command**: `docker rm <container-id>`
- **Purpose**: Removes a stopped container to free space.
- **Example**:
  ```bash
  docker rm xyz789
  ```

### 6. Push an Image to a Registry
- **Command**: `docker push <username>/<image-name>:<tag>`
- **Purpose**: Uploads an image to a registry (e.g., Docker Hub) for sharing or use on other machines (e.g., HPC).
- **Example**:
  ```bash
  docker tag ml-transformer yourusername/ml-transformer:latest
  docker push yourusername/ml-transformer:latest
  ```

### 7. Pull an Image from a Registry
- **Command**: `docker pull <username>/<image-name>:<tag>`
- **Purpose**: Downloads an image from a registry.
- **Example**:
  ```bash
  docker pull yourusername/ml-transformer:latest
  ```

### 8. Run with Dynamic Code (No New Image)
- **Command**: `docker run --rm -v <local-project>:<container-path> -w <container-path> <base-image> <command>`
- **Purpose**: Runs a container from a base image, mounting project code without building a new image.
- **Example**:
  ```bash
  docker run --rm -v $(pwd)/ml-transformer-project:/app -w /app -v $(pwd)/output:/app/output pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime bash -c "pip install -r requirements.txt && python train.py"
  ```
  - Uses the base PyTorch image, mounts the project folder, installs dependencies, and runs `train.py`.

## Getting Started
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd ml-transformer-project
   ```
2. Build the image:
   ```bash
   docker build -t ml-transformer .
   ```
3. Run a container with GPU support and output mounting:
   ```bash
   docker run --rm -v $(pwd)/output:/app/output --gpus all ml-transformer
   ```
4. Check the `output` folder for model weights and logs.

## Notes
- Use `--rm` for disposable containers to save space.
- Mount local directories (`-v`) to persist outputs.
- For HPC, push the image to Docker Hub or use Singularity (`singularity run --nv docker://yourusername/ml-transformer`).
- Extend the `Dockerfile` for project-specific dependencies or code.