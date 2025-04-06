# DexTOG: Learning Task-Oriented Dexterous Grasp With Language Condition

[**Paper**](https://ieeexplore.ieee.org/document/10803020) | [**Project Page**](https://dextog.robotflow.ai/) <br>

This repository offers the implementation of the research paper:

**DexTOG: Learning Task-Oriented Dexterous Grasp With Language Condition**  
Jieyi Zhang, Wenqiang Xu, Zhenjun Yu, Pengfei Xie, Tutian Tang, Cewu Lu  
**IEEE Robotics and Automation Letters**

## Installation

### System Compatibility
The code is designed to run on Linux systems.

### Environment Setup with Conda and Pip
First, install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). For setting up the Conda environment, we'll use the name `dextog`:

```shell
conda create -n dextog python=3.10
conda activate dextog
```

### Installing Dependencies for Each Part
Follow the installation guides in the [DexDiffu part](dexdiffu/README.md) and [RL part](rl/README.md) to complete the installation of their respective dependencies.

### Downloading Essential Components
You need to download the essential components from [Hugging Face](https://huggingface.co/datasets/robotflow/DexTOG).

After downloading, extract the dataset into the `assets` directory using the following commands:

```shell
mkdir assets
zip -FF /path/to/download/assets/assets.zip --out assets/tmp.zip
cd assets
unzip tmp.zip && rm tmp.zip
cd ..
```

### Installing Dependencies for the RFUniverse Simulator
Use the following command to install the necessary libraries for the RFUniverse simulator:

```shell
sudo apt install minizip libc6-dev
```

If you're using Ubuntu 22.04, you also need to run this additional command:

```shell
sudo ln -s /lib/x86_64-linux-gnu/libdl.so.2 /lib/x86_64-linux-gnu/libdl.so
```

## Running the Code
Refer to the [DexDiffu part](dexdiffu/README.md) and [RL part](rl/README.md) to run each part separately.

Before running, ensure you are on the `main` branch:

```shell
git checkout main
```

In this repository, we only provide the code to integrate the two parts. To start the data engine described in the paper, run the following command:

```shell
python refine_loop.py
```

## Citation
If you find our code or paper useful, please consider citing it using the following BibTeX entry:

```bibtex
@article{dextog,
    author = {Zhang, Jieyi and Xu, Wenqiang and Yu, Zhenjun and Xie, Pengfei and Tang, Tutian and Lu, Cewu},
    journal = {IEEE Robotics and Automation Letters},
    title = {DexTOG: Learning Task-Oriented Dexterous Grasp With Language Condition},
    year = {2025},
    volume = {10},
    number = {2},
    pages = {995--1002},
    keywords = {Grasping;Robots;Planning;Grippers;Vectors;Three-dimensional displays;Noise reduction;Engines;Noise;Diffusion processes;Deep learning in grasping and manipulation;dexterous manipulation},
    doi = {10.1109/LRA.2024.3518116}
}
```