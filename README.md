# CLEAN: Enzyme Function Prediction using Contrastive Learning

This is an *unofficial* repo for the paper `Enzyme Function Prediction using Contrastive Learning`. *Official* repo can be found [here](https://github.com/tttianhao/CLEAN)

This repo provides an option to use GPU for training and inference CLEAN in docker. The docker in the official repo only supports CPU.

## Requirements and Installation

A machine running Linux with NVIDIA GPU is required. 

Please follow the following steps:

1. Install [docker](https://docs.docker.com/engine/install/) and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for GPU support. And [setup docker to run without root](https://docs.docker.com/engine/install/linux-postinstall/).

2. Clone this repo and `cd` into it:

    ```
    git clone https://github.com/yuxin212/CLEAN_GPU.git
    cd CLEAN_GPU
    ```

3. Make sure docker can run with NVIDIA GPUs:

    ```
    docker run --rm --gpus all nvidia/cuda:11.3.0--cudnn8-runtime-ubuntu18.04 nvidia-smi
    ```

    This command will output a list of GPUs on your machine. If it doesn't work, please check the installation of docker and nvidia-container-toolkit.

4. Build the docker image:

    We provide a pre-built docker image for training using `triplet margin` loss hosted on [docker hub](https://hub.docker.com/repository/docker/yuxin60/clean_train/). Run the following command to pull the image:

    ```
    docker pull yuxin60/clean_train:latest
    ```
    
    If you want to build the image by yourself, please follow the following steps:

    1. Docker image for training

        The image built with default dockerfile will use `triplet margin` loss, if you want to train the model with `SupCon-Hard` loss, modify `docker/Dockerfile_train` accordingly. 

        Run the following command to build the image:

        ```
        docker build -f docker/Dockerfile_train -t clean_train .
        ```
    
    2. Docker image for inference

        The image built with default dockerfile will use the pretrained weights from the official repo. If you want to use the weights trained by yourself, modify `app/CLEAN_infer_fasta.py` and `docker/Dockerfile_infer` accordingly.

        Run the following command to build the image:

        ```
        docker build -f docker/Dockerfile_infer -t clean_infer .
        ```

5. Create a virual environment using [venv](https://docs.python.org/3/tutorial/venv.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) and install dependencies for `run_docker.py`:

    ```
    python -m pip install -r docker/requirements.txt
    ```

## Runnning CLEAN

### Training

Activate the virtual environment and run `run_docker.py` to train CLEAN. Change `--output_dir` to the absolute path of the directory you want to save the model and output from [ESMFold](https://github.com/facebookresearch/esm). Change `--training_data` to `split30`, `split50`, `split70`, or `split100` if you want to train the model with different splits. Change `--epoch` to the number of epochs you want to train the model. 

```
python docker/run_docker.py \
    --output_dir=/home/user/absolute_path \
    --training_data=split10 \
    --epoch=2500
```

### Inference

Run the following command to infer CLEAN. Change `/home/user/absolute_path` to the absolute path of the directory you want to save the output. Change `--fasta_data` to the name of the fasta file you want to infer. 

```
docker run --rm --gpus all --name clean_infer \
    -v /home/user/absolute_path:/app/CLEAN/results \
    clean_infer --fasta_data query
```

