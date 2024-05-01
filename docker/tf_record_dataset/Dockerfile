FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TF_ENABLE_ONEDNN_OPTS=0

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip

WORKDIR /home/code
COPY ./docker/tf_record_dataset/requirements.txt /home/code/requirements.txt
RUN pip3 install -r requirements.txt

# Copy the source code
COPY ./src /home/code

CMD ["bash"]
