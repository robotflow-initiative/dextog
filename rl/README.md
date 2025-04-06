# Reinforcement Learning Part Guide

This section provides a thorough guide on using the reinforcement learning component of DexTOG.

## Installation

### Prerequisites
Change to the rl branch:
```shell
git checkout rl
```
Before you start the installation process, make sure you have entered the correct environment.
This step is crucial to ensure that all the dependencies are installed correctly and work as expected.

### Install Dependencies
Execute the following commands to install all the necessary dependencies. Each command plays a vital role in setting up the environment for the reinforcement learning part of DexTOG.

```shell
pip install setuptools==61.2.0
pip install hydra-core numpy tqdm icecream wandb loguru 
pip install pyrfuniverse==0.30.0.3
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

After installing the general dependencies, you need to install a specific version of `stable-baselines3`. Navigate to the relevant directory and install it in development mode:

```shell
cd third_party/stable_baselines3
python setup.py develop
```

### Download the Essential Components
You should download checkpoint files, and essential components on [Hugging Face](https://huggingface.co/datasets/robotflow/DexTOG).

After downloading, extract the dataset into the `assets` directory using the following commands:
```shell
mv /path/to/download/rl/assets.zip .
unzip assets.zip
rm assets.zip
```

## Running the Code

### Training the RL Policy
To initiate the training of the RL policy, run the following command:

```shell
python train.py
```

We utilize a parallel environment to speed up the training process. If you wish to adjust the number of parallel environments, you can modify the `proc_num` (which indicates the number of processes) and `in_proc_env_num` (which indicates the number of environments within one process) parameters. For example:

```shell
python train.py train.proc_num=20 train.in_proc_env_num=10
```

### Testing the RL Policy
To test the trained RL policy, run the following command:

```shell
python test.py
```

### Changing the Initial Pose
If you want to change the initial pregrasp pose dataset, you can modify the `dataset_path` parameter in the configuration file. This allows you to customize the starting pose for your reinforcement learning experiments. 