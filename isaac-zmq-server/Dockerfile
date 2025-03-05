# Base on PyTorch Docker image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Create a non-root user and group for security
RUN addgroup --system appgroup
RUN adduser --system appuser --ingroup appgroup

# Install dependencies for DearPyGui and ZMQ
RUN apt-get update
RUN apt-get --no-install-recommends install \
    libgl1-mesa-glx \
    libglib2.0-0 -y

# Switch to the non-root user for security
USER appuser

# Install Python dependencies
# - dearpygui: GUI framework for Python
# - zmq: ZeroMQ messaging library
# - opencv-python: Computer vision library
# - protobuf: Protocol Buffers library
RUN pip install \
    dearpygui \
    zmq \
    opencv-python \
    protobuf==3.19.4

# Set the working directory for the application
WORKDIR /isaac-zmq-server/src

# Add the source code to the image (We will also mount it for dynamic updates)
ADD ./src /isaac-zmq-server/src