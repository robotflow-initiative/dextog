# DiPGrasp: Parallel Local Searching for Efficient Differentiable Grasp Planning

[**Paper**] | [**Project Page**](https://dextog.robotflow.ai/) <br>

<div style="text-align: center">
<img src="media/pipeline.png" width="1000"/>
</div>

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

## Installation
using the commands to install the dependencies 
```
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
pip install hydra-core lightning diffusers scipy open3d fire tensorboard
pip install git+https://github.com/SJTUzjy/urdfpy.git@afc0e0b821256ca952e59aefa0f6fe35a2caffcf
```
Following the [Installation Guidance](https://kaolin.readthedocs.io/en/latest/notes/installation.html) to install the kaolin library
```aiignore
 pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.0_cu124.html
```


### Download the essential components

We upload the objects mentioned in paper and some essential components on https://huggingface.co/datasets/robotflow/DipGrasp  
You can follow the instructions to download the assets.
After downloading, you should move the `assets` directory to the project main directory:

```shell
mv path/to/assets .
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

> This version is structured to guide users step-by-step, making it easy to follow.

## Container Based Deployment

To quick explore the code, you can use docker to setup a demo environment. You can use the following command to build the docker image:

```shell
docker build -t dipgrasp_env .
```

> You need to have a decent Internet connection to build the docker image, because the docker image will download the essential components from the Internet.

> To enable GPU acceleration, you need to enable nvidia-container-toolkit by following the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

Then, you can start the container with the following command:

```shell
docker run --rm -it --gpus all -p 3000:80 -v path/to/assets:/opt/dipgrasp/assets dipgrasp_env
```

You may want to change the command according to your needs. For example, if you want to use a different port, you should change the `-p 3000:80` section. If you don't have nvidia-container-toolkit installed, you can remove the `--gpus all` section.

After the container is started, you can access the Development by visiting:

- [http://localhost:3000/coder/](http://localhost:3000/coder/), this is a Web-based vscode-like IDE, you can use it to edit the code.
- [http://localhost:3000/novnc/](http://localhost:3000/), this is a vnc viewer, you can use it to visualize the simulation.

For example, in the vnc viewr, you can click the terminal button on the bottom, and then type the following command to activate the conda environment and change to the project directory:

```shell
conda activate dipgrasp
cd /opt/dipgrasp
```

![Docker Demo](media/docker_demo.png)

## Generate the grasp pose for one object

You can run the following command to generate the grasp pose:

```shell
python main.py
```

We use Schunk SVH hand as our default gripper. You could see the object and the gripper with generated pose after it generate each batch of pose.

The output pose will be saved at `output` directory, consisting of two numpy files `pose.npy` and `joint_state.npy`. The `pose.npy` contains a numpy array with shape `(N, 4, 4)`,
while the `joint_state.npy` contains a numpy array with shape `(N, D)`, where `N` is the number of generated poses and the `D` is the DoF of the gripper.

To generate the pose with other grippers, you could use the following command:

```shell
python main.py gripper=shadow
```

where the `shadow` could be repleced by `svh` or `barrett`.

By setting the the `algo_params.sample_time`, you can adjust the number of parallel gripper trial to fit the memory size, e.g.:

```shell
python main.py algo_params.sample_time=64
```

The default value of `algo_params.sample_time` is 256, which typically occupies about 16GB memory.

To get the subtle and stable pose where the gripper contacts with the object surface, we use the [RFUniverse](https://github.com/robotflow-initiative/rfuniverse) as the simulator to refine the generated poses.
You can use the following command to get the refined poses:

```shell
python main.py datafile=assets/object/class/drink/drink9/drink9.obj simulator=True visualize_setting.visualize=False visualize_setting.visualize_after_simulator=True
```

You may notice that we change the target object file, because the simulator needs a mesh file rather than pointcloud.

## Generate the grasp pose for a batch of object

To generate a large number of poses on different objects, we provide a [script file](scripts/generate_for_batch_obj.py), you can run the following command to generate the poses:

```shell
python scripts/generate_for_batch_obj.py 
```

The poses saved have passed through the simulator by default.
You could also alter the gripper type by adding `gripper` parameter.
The number of generated poses will be logged into `results.log`.

You could generate grasp poses on your own objects by modifying the script.
The simulator only supports object with .obj format.

## Adapting to Custom Hand Models

If you're interested in applying our algorithm to your custom hand model, you can easily do so by following these steps:

1. **Generate the Point Cloud XML File:**
   - First, obtain the palmar side point cloud XML file. We've provided a point cloud generator as discussed in Section IV.D of our paper. You can find this tool in the `assets/xml_generator` directory. For detailed instructions, check out [this document](media/XMLGeneratorTutorial.md).

2. **Organize Your Files:**
   - Place the generated XML file alongside your hand URDF file and any related description files within the `assets/gripper` directory. Ensure they follow the same format as the hand models we currently support.

3. **Update Configuration Files:**
   - Add a new gripper configuration file to the `conf/gripper` directory, similar to the existing gripper configuration files.

4. **Modify the Main Script:**
   - Include the name of your new hand model in the list within [main.py](./main.py) at line 113.

5. **Adjust Initial Position Settings:**
   - Ensure that the hand is oriented correctly towards the object in its initial position. You can tweak the `init_sample_config` function in [config.py](core/config.py) and look at the `correct_coordinate` function in [samplePosition.py](core/geometry/samplePosition.py) for guidance.

By following these steps, you'll be able to integrate your custom hand model into our system.
## Citation
If you find our code or paper useful, please consider citing
```bibtex
@article{dipgrasp,
      title = {DiPGrasp: Parallel Local Searching for Efficient Differentiable Grasp Planning},
      author = {Xu, Wenqiang and Zhang, Jieyi and Tang, Tutian and Yu, Zhenjun and Li, Yutong and Lu, Cewu},
      journal = {IEEE Robotics and Automation Letters},
      year = {2024},
      publisher = {IEEE},
    }
}

```
