# DexTOG: Learning Task-Oriented Dexterous Grasp With Language Condition

[**Paper**] | [**Project Page**](https://dextog.robotflow.ai/) <br>

This repository contains the implementation of the paper:

**DexTOG: Learning Task-Oriented Dexterous Grasp With Language Condition**  
Jieyi Zhang, Wenqiang Xu, Zhenjun Yu, Pengfei Xie, Tutian Tang, Cewu Lu
**IEEE Robotics and Automation Letters**

## Installation

The code should run on Linux System.

### Set up the environments with conda and pip

Install [anaconda](https://www.anaconda.com/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Supposing that the name `dextog` is used for conda environment:

```shell
conda create -n dextog python=3.10
conda activate dextog
```
### Install the dependency of each part
Please completion the installation by following the guide of [DexDiffu part](dexdiffu/README.md) and [rl part](rl/README.md).

### Download the Essential Components
You should download the essential components on [Hugging Face](https://huggingface.co/datasets/robotflow/DexTOG).

After downloading, extract the dataset into the `assets` directory using the following commands:
```shell
mkdir assets
zip -FF /path/to/download/assets/assets.zip --out assets/tmp.zip
cd assets
unzip tmp.zip && rm tmp.zip
cd ..
```
### Install the essential dependencies for RFUniverse simulator

Please use the following command to install the dependency libraries for RFUniverse:

```shell
sudo apt install minizip libc6-dev
```

If you are using Ubuntu 22.04, you’ll also need to run this additional command:

```shell
sudo ln -s /lib/x86_64-linux-gnu/libdl.so.2 /lib/x86_64-linux-gnu/libdl.so
```

## Running the code
Please refer to the [dexdiffu part](dexdiffu/README.md) and [rl part](rl/README.md)
to run the two parts respectively.

Make sure you are in `main` branch:
```shell
git checkout main
```

In this part, we only provide the code to integrate the two parts.
By running `refine_loop.py`, you could start up the data engine described in the paper.
```shell
python refine_loop.py
```
## Citation
If you find our code or paper useful, please consider citing
```bibtex
@article{dextog,
      author={Zhang, Jieyi and Xu, Wenqiang and Yu, Zhenjun and Xie, Pengfei and Tang, Tutian and Lu, Cewu},
      journal={IEEE Robotics and Automation Letters},
      title={DexTOG: Learning Task-Oriented Dexterous Grasp With Language Condition},
      year={2025},
      volume={10},
      number={2},
      pages={995-1002},
      keywords={Grasping;Robots;Planning;Grippers;Vectors;Three-dimensional displays;Noise reduction;Engines;Noise;Diffusion processes;Deep learning in grasping and manipulation;dexterous manipulation},
      doi={10.1109/LRA.2024.3518116}}
}

```
