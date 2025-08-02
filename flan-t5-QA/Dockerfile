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