import os
import time
import torch
import queue
import wandb
import numpy as np
import os.path as osp
import multiprocessing as mp

from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig
from collections import OrderedDict
from stable_baselines3.ppo import PPO
from pyrfuniverse.utils.proc_wrapper import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from .dataset import get_data_dict, DataIndexLoader
from ..envs import dextog_env_map, dextog_callback_map


class env_object(object):
    def __init__(self,
                 task: str,
                 in_proc_envs: int,
                 root_dir: str = osp.join("assets", "grasp", "TOGSample")):

        self.task = task
        self.in_proc_envs = in_proc_envs
        self._data = get_data_dict(task, root_dir)

    def __call__(self, **kwargs):
        VecEnv = dextog_env_map[self.task]
        return VecEnv(self.in_proc_envs, shared_data=np.array(self._data), **kwargs)

class Trainer(object):

    def __init__(self, cfg: DictConfig):
        task = cfg.task

        if cfg.mode == 'train':
            n_proc = cfg.train.proc_num
            in_proc_envs = cfg.train.in_proc_env_num
            wandb.init()
            env_obj = env_object(task, in_proc_envs, cfg.train.dataset_path)
            env_fns = [env_obj] * n_proc
            self.env = SubprocVecEnv(env_fns, n_agents=in_proc_envs)
        else:
            pass

        self.cfg = cfg


    def run(self):
        if self.cfg.mode == 'train':
            self.train()
        elif self.cfg.mode in ['test', 'generate']:
            return self.play_mp()
        else:
            raise NotImplementedError

    def play(self):
        cfg = self.cfg

        lr = cfg.train.lr
        epoch = cfg.train.epoch
        horizon = cfg.train.horizon
        ent_coef = cfg.train.ent_coef
        ckpt_path = cfg.train.ckpt

        net_arch = dict(pi=list(cfg.model.pi), vf=list(cfg.model.vf))
        activation = nn.__getattribute__(cfg.model.activation)

        model = PPO(
            "MultiInputPolicy",
            self.env,
            verbose=1,
            n_epochs=epoch,
            n_steps=horizon,
            learning_rate=lr,
            batch_size=horizon * self.env.num_envs,
            ent_coef=ent_coef,
            policy_kwargs=dict(
                activation_fn=activation,
                net_arch=[net_arch],
            ),
            tensorboard_log="./tb_logs",
        )

        custom_objects = {
            # "observation_space": env.observation_space,
            "n_envs": self.env.num_envs,
            "lr_schedule": model.lr_schedule,
            "clip_range": model.clip_range
        }
        device = "cpu"
        model = model.load(ckpt_path, self.env, device, custom_objects=custom_objects)

        data_idx_loader = DataIndexLoader(get_data_dict(cfg.task))

        data_len = len(data_idx_loader)
        t = 0
        batch_size = cfg.train.in_proc_env_num
        while t < data_len:
            cate_ids, ids = data_idx_loader.get_batch(batch_size)

            obs = self.env.reset_by_ids(cate_ids, ids)
            for _ in range(100):
                actions = []
                for i in range(self.env.num_envs):
                    obs_i = {key: value[i].reshape(1, -1) for key, value in obs.items()}
                    obs_i = OrderedDict(obs_i)
                    action, _states = model.predict(obs_i, deterministic=True)
                    actions.append(action)
                action = np.stack(actions)
                obs, rewards, dones, info = self.env.step(action)

            print(info)
            t += batch_size

    @staticmethod
    def worker_fn(cfg, local_rank: int, world_size: int, output_queue: mp.Queue):
        print(f"local_rank={local_rank}")
        env = env_object(cfg.task, cfg.train.in_proc_env_num, cfg.train.dataset_path)(proc_id=local_rank)

        lr = cfg.train.lr
        epoch = cfg.train.epoch
        horizon = cfg.train.horizon
        ent_coef = cfg.train.ent_coef
        ckpt_path = cfg.train.ckpt

        net_arch = dict(pi=list(cfg.model.pi), vf=list(cfg.model.vf))
        activation = nn.__getattribute__(cfg.model.activation)
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_epochs=epoch,
            n_steps=horizon,
            learning_rate=lr,
            ortho_init=False,
            device = "cpu",
            batch_size=horizon * env.num_envs,
            ent_coef=ent_coef,
            policy_kwargs=dict(
                activation_fn=activation,
                net_arch=[net_arch],
            ),
            tensorboard_log="./tb_logs",
        )
        custom_objects = {
            "n_envs": env.num_envs,
            "lr_schedule": model.lr_schedule,
            "clip_range": model.clip_range
        }
        del model
        device = "cpu"
        model = PPO.load(ckpt_path, env, device, custom_objects=custom_objects)

        data_idx_loader = DataIndexLoader(get_data_dict(cfg.task, cfg.train.dataset_path))

        data_len = len(data_idx_loader)
        t = 0
        batch_size = cfg.train.in_proc_env_num
        rank = -1
        while t < data_len:
            # load input
            cate_ids, ids = data_idx_loader.get_batch(batch_size)
            t += batch_size
            rank += 1
            if rank % world_size != local_rank:
                continue

            obs = env.reset_by_ids(cate_ids, ids)
            obs = {key: torch.as_tensor(_obs).to(device) for (key, _obs) in obs.items()}
            for _ in range(100):
                actions = []
                for i in range(env.num_envs):
                    obs_i = {key: value[i].reshape(1, -1) for key, value in obs.items()}
                    obs_i = OrderedDict(obs_i)
                    action, _states = model.predict(obs_i, deterministic=True)
                    actions.append(action)
                action = np.stack(actions)
                obs, rewards, dones, info = env.step(action)
            output_queue.put(info)
        env.close()

    def play_mp(self):
        print("starting play_mp")
        data_idx_loader = DataIndexLoader(get_data_dict(self.cfg.task, self.cfg.train.dataset_path))

        data_len = len(data_idx_loader)
        if data_len == 0:
            print(f"success rate of {self.cfg.task} is: 0")
            return
        success_cnt = 0.0
        success_dict = data_idx_loader.get_success_dict()

        output_queue = mp.Queue(maxsize=1024)
        world_size = self.cfg.train.proc_num
        processes = [
            mp.Process(
                target=self.worker_fn,
                args=(self.cfg, i, world_size, output_queue)
            )
            for i in range(world_size)
        ]
        [p.start() for p in processes]
        print("jobs submitted")
        with tqdm(total=data_len) as pbar:
            pbar.set_description("Processing:")
            while pbar.n < data_len:
                try:
                    info = output_queue.get(timeout=300)
                    pbar.update(self.cfg.train.in_proc_env_num)
                    for info_i in info:
                        if info_i['success']:
                            obj_id = info_i['obj_id']
                            pose_id = info_i['pose_id']
                            success_dict[obj_id][pose_id] = True
                            success_cnt += 1
                    if pbar.n % 50 == 0:
                        print(f"Success rate in {pbar.n} tries is: {success_cnt / pbar.n}")
                except queue.Empty:
                    print("No data")
                    break
        for key, value in success_dict.items():
            success_rate = np.sum(value) * 1.0 / len(value)
            print(f"success rate of {key} is: {success_rate}")
        print(f"success rate of {self.cfg.task} is: {success_cnt / data_len}")
        return success_dict, success_cnt / pbar.n

    def train(self) -> str:
        cfg = self.cfg

        lr = cfg.train.lr
        epoch = cfg.train.epoch
        horizon = cfg.train.horizon
        ent_coef = cfg.train.ent_coef
        if not hasattr(cfg.train, 'max_timestep') or cfg.train.max_timestep == 0:
            max_timestep = 100000000000
        else:
            max_timestep = cfg.train.max_timestep

        net_arch = dict(pi=list(cfg.model.pi), vf=list(cfg.model.vf))
        activation = nn.__getattribute__(cfg.model.activation)

        model = PPO(
            "MultiInputPolicy",
            self.env,
            verbose=1,
            n_epochs=epoch,
            n_steps=horizon,
            learning_rate=lr,
            batch_size=horizon * self.env.num_envs,
            ent_coef=ent_coef,
            policy_kwargs=dict(
                activation_fn=activation,
                net_arch=[net_arch],
            ),
            tensorboard_log="./tb_logs",
        )

        net_structure = ""
        for key, value in net_arch.items():
            net_structure += f"{key}-"
            for v in value:
                net_structure += f"{v}-"

        log_name = "ppo-rand-" + self.cfg.task + "-" + net_structure + "ep-" + str(epoch) + "-bs-" + str(
                horizon * self.env.num_envs)
        if not hasattr(cfg.train, 'save_path') or cfg.train.save_path == None:
            save_path = os.path.join(
                "./tb_logs", "checkpoints", log_name, time.strftime("%Y-%m-%d-%H-%M-%S")
            )
        else:
            save_path = cfg.train.save_path

        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=save_path,
        )

        tensorboard_callback = dextog_callback_map[self.cfg.task]()

        model.learn(
            total_timesteps = max_timestep,
            tb_log_name=log_name,
            log_interval=1,
            callback=[checkpoint_callback, tensorboard_callback],
            progress_bar=True,
        )
        self.env.close()
        del model
        return osp.join(save_path, f'rl_model_{max_timestep}_steps.zip')


