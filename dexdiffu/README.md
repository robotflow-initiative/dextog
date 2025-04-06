# DexDiffu Guide

This section serves as a comprehensive guide to using DexDiffu.

## Installation

### Prerequisites
Change to the dexdiffu branch:
```shell
git checkout dexdiffu
```
Ensure you have entered the correct environment before proceeding with the installation.

### Install Dependencies
Execute the following commands to install the necessary dependencies:
```shell
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install hydra-core lightning diffusers scipy open3d fire tensorboard
pip install git+https://github.com/SJTUzjy/urdfpy.git@afc0e0b821256ca952e59aefa0f6fe35a2caffcf
```

### Install Kaolin Library
Follow the [Installation Guide](https://kaolin.readthedocs.io/en/latest/notes/installation.html) to install the kaolin library. Then, run the command below:
```shell
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html
```

### Download Dataset and Essential Components
You can access the dataset, checkpoint files, and essential components on [Hugging Face](https://huggingface.co/datasets/robotflow/DexTOG).

After downloading, extract the dataset into the `assets` directory using the following commands:
```shell
mkdir assets
zip -FF /path/to/download/DexTOG-80K/DexTOG-80k.zip --out assets/tmp.zip
cd assets
unzip tmp.zip && rm tmp.zip
cd ..
```

## Running the Code

### Generate Grasp Poses
Before performing inference, confirm that you have the checkpoint files and that the checkpoint path in the configuration file is accurate.

- To generate and visualize grasp poses, run the following command:
```shell
python sample.py
```

- To specify an object category, use the following parameterized command:
```shell
python sample.py sample.category=drink
```

- To generate a large number of poses, use the following command:
```shell
python generate.py generate.save_path=diffusion_output
```

### Training the Model
To train the model, execute the following command:
```shell
python train.py
``` 