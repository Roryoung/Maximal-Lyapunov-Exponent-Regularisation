# Build pip dependencies
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS pip_build
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc git

WORKDIR /usr/app
RUN python -m venv /usr/app/venv
ENV PATH="/usr/app/venv/bin:$PATH"

COPY requirements.txt .
RUN /usr/app/venv/bin/pip install -r requirements.txt

# Build linux dependencies
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as linux_build
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgl1-mesa-glx libosmesa6
RUN apt-get update
RUN apt-get install --fix-missing -y parallel

# Build code
FROM linux_build
WORKDIR /usr/app
COPY --from=pip_build /usr/app/venv ./venv
COPY . .

# Set environment variables
ENV PATH="/usr/app/venv/bin:$PATH"
ENV MUJOCO_GL="osmesa"
ENV PYTHONPATH=$PYTHONPATH':/home/user/'
