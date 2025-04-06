import math
import os
import json
import time
import multiprocessing

import numpy as np
import os.path as osp

from typing import Dict, Tuple
from loguru import logger
from omegaconf import DictConfig
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv

from rl.train.dataset import get_data_dict
from rl.train.trainer import Trainer as RLTrainer
from dexdiffu.train import train as diffu_train
from dexdiffu.generate import generate as diffu_generate

def filter_grasp_pose(cfg: DictConfig,
                      dataset_path: str,
                      tmp_root_path: str) -> Tuple[str, float]:
    train_cfg = cfg.train_cfg
    train_cfg.train.dataset_path = dataset_path
    train_trainer = RLTrainer(train_cfg)
    ckpt = train_trainer.train()
    logger.info('RL Train process ended')
    time.sleep(5)

    test_cfg = cfg.test_cfg
    test_cfg.train.dataset_path = dataset_path
    test_cfg.train.ckpt = ckpt
    test_trainer = RLTrainer(test_cfg)
    success_dict, success_rate = test_trainer.play_mp()
    data_dict = get_data_dict(test_cfg.task, dataset_path)

    generate_dataset(data_dict, success_dict, tmp_root_path, test_cfg.task)
    logger.info('RL Test process ended')

    return ckpt, success_rate


def generate_dataset(data_dict: Dict[str, Dict[str, np.ndarray]],
                     success_dict: Dict,
                     root_path: str,
                     task: str):
    dataset_path = osp.join(root_path, task)
    os.makedirs(dataset_path, exist_ok=True)
    for obj_id in data_dict.keys():
        obj_path = osp.join(dataset_path, obj_id)
        os.makedirs(obj_path, exist_ok=True)
        success_label = success_dict[obj_id]
        pose = data_dict[obj_id]["pose"]
        joint_state = data_dict[obj_id]["joint_state"]
        pose = pose[success_label]
        joint = joint_state[success_label]
        success_label = ["True"] * joint.shape[0]
        assert pose.shape[0] == joint.shape[0]
        logger.info(f"success rate of {obj_id} is: {1.0 * joint.shape[0] / joint_state.shape[0]}")
        logger.info(f"Success Number: {joint.shape[0]}, Total Number: {joint_state.shape[0]}")
        pose_file = osp.join(obj_path, 'success_pose.json')
        joint_state_file = osp.join(obj_path, 'success_grab_joint_state.json')
        success_label_file = osp.join(obj_path, 'success_pressed.json')
        with open(pose_file, 'w') as f:
            json.dump(pose.tolist(), f)
        with open(joint_state_file, 'w') as f:
            json.dump(joint.tolist(), f)
        with open(success_label_file, 'w') as f:
            json.dump(success_label, f)


def augment_dataset(cfg: DictConfig,
                    dataset_path: str,
                    save_path: str,
                    iter_time: int):
    train_cfg = cfg.train_cfg
    train_cfg.train.dataset_path = dataset_path
    ckpt_save_dir = diffu_train(train_cfg)

    ckpt_path = get_ckpt_path_from_dir(osp.join(ckpt_save_dir, f"version_{iter_time}", "checkpoints"))
    scheduler_path = osp.join(ckpt_save_dir, f"version_{iter_time}")

    generate_cfg = cfg.generate_cfg
    generate_cfg.generate.scheduler_path = scheduler_path
    generate_cfg.generate.ckpt_path = ckpt_path
    generate_cfg.generate.save_path = save_path

    diffu_generate(generate_cfg)


def get_ckpt_path_from_dir(ckptdir: str):
    files = os.listdir(osp.join(os.getcwd(), ckptdir))

    max_iter = 0
    max_idx = -1
    for idx, file in enumerate(files):
        file_name = file.split('.')[0]
        file_iter = file_name.split('=')[-1]
        if int(file_iter) > max_iter:
            max_iter = int(file_iter)
            max_idx = idx
    best_file = files[max_idx]
    return osp.join(ckptdir, best_file)


def filter_collision(datapath: str, task: str):
    processPool = []

    main_path = osp.join(datapath, task)
    obj_list = os.listdir(main_path)
    process_num = len(obj_list)

    for idx in range(process_num):
        processPool.append(
            multiprocessing.Process(target=filter_collision_worker, args=(datapath, task, idx, process_num)))
    for process in processPool:
        time.sleep(3)
        process.start()
    [process.join() for process in processPool]
    print('All Done!')

def filter_collision_worker(datapath: str, task: str, id: int, p: int):
    main_path = osp.join(datapath, task)
    obj_list = os.listdir(main_path)

    success = None
    pressed = None
    joint_state = None
    done = None

    def ReceiveData(obj: list):
        nonlocal success
        nonlocal pressed
        nonlocal joint_state
        nonlocal done

        success = obj[0]
        joint_state = obj[1]
        pressed = obj[2]
        done = True

    graphics = True if os.environ.get("DISPLAY", None) is not None and id == 0 else False
    env = RFUniverseBaseEnv(executable_file='assets/filter/grasp_filter/filter.x86_64',
                            assets=['shadowhand'],
                            graphics=graphics,
                            log_level=0,
                            communication_backend="grpc"
                            )
    env.AddListenerObject('Result', ReceiveData)
    env.log_level = 0
    env.SetTimeScale(2)
    env.SetTimeStep(0.02)
    for idx, one in enumerate(obj_list):
        if idx % p != id: continue
        logger.debug(f"Process {id} start to process object {one}")
        current_path = os.path.join(main_path, one)
        joint = os.path.join(current_path, 'joint_state.npy')
        pose = os.path.join(current_path, 'pose.npy')

        pose = np.load(pose)
        joint = np.load(joint)

        result_success = np.empty((pose.shape[0]), dtype=bool)

        chunk_count = 256
        chunk = pose.shape[0] / chunk_count
        chunk = math.ceil(chunk)

        for i in range(chunk):
            start = i * chunk_count
            end = (i + 1) * chunk_count
            if end > pose.shape[0]:
                end = pose.shape[0]
            pose_chunk = pose[start:end]
            joint_chunk = joint[start:end]

            env.SendObject(
                'Test',
                one.split('_')[0],
                'shadowhand',
                23,
                [0., 0., -90.],
                pose_chunk.reshape(-1).tolist(),
                joint_chunk.reshape(-1).tolist(),
                50,
                task,
                False,  # grasp
                False,  # gravity
                False,  # press
            )
            done = False
            while not done:
                env.step()

            result_success[start:end] = np.array(success)

        save_path = os.path.join(current_path, 'success_label.json')
        with open(save_path, 'w') as f:
            json.dump(result_success.tolist(), f)
        logger.debug(f"Process {id} finish {idx}")

    env.close()

def filter_not_catch(datapath: str, task: str, process_num: int = 5):
    processPool = []

    main_path = osp.join(datapath, task)
    obj_list = os.listdir(main_path)
    process_num = len(obj_list)

    for idx in range(process_num):
        processPool.append(multiprocessing.Process(target=filter_not_catch_worker, args=(datapath, task, idx, process_num)))
    for process in processPool:
        time.sleep(3)
        process.start()
    [process.join() for process in processPool]
    print('All Done!')

def filter_not_catch_worker(datapath: str, task: str, id: int, p: int):
    main_path = osp.join(datapath, task)
    obj_list = os.listdir(main_path)
    gravity_switch = False if task == 'drink' else True

    success = None
    pressed = None
    joint_state = None
    done = None

    def ReceiveData(obj: list):
        nonlocal success
        nonlocal pressed
        nonlocal joint_state
        nonlocal done

        success = obj[0]
        joint_state = obj[1]
        pressed = obj[2]
        done = True

    graphics = True if os.environ.get("DISPLAY", None) is not None and  id == 0 else False
    env = RFUniverseBaseEnv(executable_file='assets/filter/grasp_filter/filter.x86_64',
        assets=['shadowhand'],
        graphics=graphics,
        log_level=0,
        communication_backend="grpc"
        )
    env.AddListenerObject('Result', ReceiveData)
    env.log_level = 0
    env.SetTimeScale(2)
    env.SetTimeStep(0.02)
    for idx, one in enumerate(obj_list):
        if idx % p != id: continue
        logger.debug(f"Process {id} start to process object {one}")
        current_path = os.path.join(main_path, one)
        joint = os.path.join(current_path, 'joint_state.npy')
        pose = os.path.join(current_path, 'pose.npy')

        pose = np.load(pose)
        joint = np.load(joint)

        result_success = np.empty((pose.shape[0]), dtype=bool)
        result_joint_state = np.empty((joint.shape))
        result_pressed = np.empty((pose.shape[0]), dtype=bool)

        joint_ids = [3, 4, 7, 8, 11, 12, 16, 17, 21, 22]
        for joint_idx in joint_ids:
            joint[:, joint_idx] = 0.0

        chunk_count = 500
        chunk = pose.shape[0] / chunk_count
        chunk = math.ceil(chunk)

        for i in range(chunk):
            start = i * chunk_count
            end = (i + 1) * chunk_count
            if end > pose.shape[0]:
                end = pose.shape[0]
            pose_chunk = pose[start:end]
            joint_chunk = joint[start:end]

            env.SendObject(
                'Test',
                one.split('_')[0],
                'shadowhand',
                23,
                [0., 0., -90.],
                pose_chunk.reshape(-1).tolist(),
                joint_chunk.reshape(-1).tolist(),
                50,
                task,
                True, # grasp
                gravity_switch, # gravity
                True, # press
            )
            done = False
            while not done:
                env.step()

            print("end")
            result_success[start:end] = np.array(success)
            result_joint_state[start:end] = np.array(joint_state)
            result_pressed[start:end] = np.array(pressed)

        # success filter
        result_pressed = result_pressed[result_success]
        result_joint_state = result_joint_state[result_success]
        pose = pose[result_success]
        result_success = result_success[result_success]

        # pressed filter
        result_joint_state = result_joint_state[result_pressed]
        pose = pose[result_pressed]
        result_success = result_success[result_pressed]

        logger.debug(f"The pressed label of {id} number is {np.sum(result_pressed)} after success filter")

        np.save(osp.join(current_path, 'joint_state.npy'), result_joint_state * np.pi / 180.0)
        np.save(osp.join(current_path, 'pose.npy'), pose)
        assert pose.shape[0] == result_joint_state.shape[0]

        save_path = os.path.join(current_path, 'success_label.json')
        with open(save_path, 'w') as f:
            json.dump(result_success.tolist(), f)
        logger.debug(f"Process {id} finish {idx}")


    env.close()