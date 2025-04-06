import gym
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from typing import List
from stable_baselines3.common.vec_env import DummyVecEnv
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.util import obs_space_info
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class BallpointCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_rollout_end(self) -> bool:
        # Log scalar value (here a random variable)
        success = self.training_env.get_attr("success")
        height = np.asarray(self.training_env.get_attr("object_height")).reshape(-1)
        dof_delta = np.asarray(self.training_env.get_attr("dof_delta")).reshape(-1)
        success = np.asarray(success).reshape(-1)
        success_rate = np.sum(np.asarray(success)) / success.shape[0]
        ave_height = np.mean(height)
        ave_delta_dof = np.mean(dof_delta)

        self.logger.record("train/success_rate", success_rate)
        self.logger.record("train/ave_height", ave_height)
        self.logger.record("train/ave_delta_dof", ave_delta_dof)
        return True

    def _on_step(self) -> bool:
        return True


class BallpointChildEnv:
    def __init__(self, rfu, env_idx, shared_data, **kwargs):
        # super().__init__(**kwargs)
        self.rfu = rfu
        self.id = env_idx
        self.observation_space = gym.spaces.Dict(
            {
                "hand_position": gym.spaces.Box(
                    low=-1, high=1, shape=(3,), dtype=np.float32
                ),
                "button_position": gym.spaces.Box(
                    low=-1, high=1, shape=(3,), dtype=np.float32
                ),
                "hand_object_offset": gym.spaces.Box(
                    low=-1, high=1, shape=(29, 3), dtype=np.float32
                ),
                "joint_positions": gym.spaces.Box(
                    low=0, high=1, shape=(24,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(29,)
        )  # base * 5, shadow * 24

        self.base = self.rfu.GetAttr(123 * 1000 + env_idx)
        self.object = self.rfu.GetAttr(111 * 1000 + env_idx)
        self.shadow = self.rfu.GetAttr(222 * 1000 + env_idx)
        self.shadow_upper_action = np.array(self.shadow.data["joint_upper_limit"])
        self.shadow_lower_action = np.array(self.shadow.data["joint_lower_limit"])
        self.base_upper_action = np.array(self.base.data["joint_upper_limit"])
        self.base_lower_action = np.array(self.base.data["joint_lower_limit"])
        self.upper_action = np.concatenate(
            [self.base_upper_action, self.shadow_upper_action]
        )
        self.lower_action = np.concatenate(
            [self.base_lower_action, self.shadow_lower_action]
        )
        self.shared_data = shared_data.item()
        self.obj_ids = list(self.shared_data.keys())

        self.obj_initial_position = None

    def reset_async(self):
        rand_obj_id = str(np.random.choice(self.obj_ids))
        self.rfu.SendObject("SetObj", rand_obj_id)
        trajectory_data = self.shared_data[rand_obj_id]

        rand_tog_idx = np.random.randint(len(trajectory_data["pose"]))
        self.obj_id = rand_obj_id
        self.pose_id = rand_tog_idx
        self.rfu.SendObject(
            "NewReset",
            self.id,
            trajectory_data["pose"][rand_tog_idx].reshape(-1).tolist(),
            trajectory_data["joint_state"][rand_tog_idx].reshape(-1).tolist(),
        )

    def reset_by_idx(self, obj_id: str, tog_idx: int):
        assert obj_id in self.obj_ids
        self.rfu.SendObject("SetObj", obj_id)
        trajectory_data = self.shared_data[obj_id]

        assert tog_idx < len(trajectory_data["pose"])
        self.obj_id = obj_id
        self.pose_id = tog_idx
        self.rfu.SendObject(
            "NewReset",
            self.id,
            trajectory_data["pose"][tog_idx].reshape(-1).tolist(),
            trajectory_data["joint_state"][tog_idx].reshape(-1).tolist(),
        )

    def reset_wait(self):
        self.obj_initial_position = np.array(self.object.data["positions"])
        self.obj_initial_joint_normalized = (
                                                    self.object.data["joint_positions"][0] - self.object.data["joint_lower_limit"][0]
                                                    ) / (
                                                    self.object.data["joint_upper_limit"][0]
                                                    - self.object.data["joint_lower_limit"][0]
                                            )
        self.obj_initial_height = self.object.data["positions"][0][1]
        self.shadow_initial_position = np.asarray(self.shadow.data['positions'][29])
        self.success = False

        return self._get_obs()

    def _get_obs(self):
        shadow_positions = np.array(self.shadow.data["positions"])[:-1]
        object_positions = np.array(self.object.data["positions"])[[0]]
        button_positions = np.array(self.object.data["positions"])[[2]]
        shadow_joint_positions = (
                                         self.shadow.data["joint_positions"] - self.shadow_lower_action
                                 ) / (self.shadow_upper_action - self.shadow_lower_action)
        return {
            "button_position": button_positions.mean(axis=0) - object_positions.mean(axis=0),
            "hand_position": np.asarray(self.shadow.data['positions'][29]) - self.shadow_initial_position,
            "hand_object_offset": shadow_positions - object_positions.mean(axis=0),
            "joint_positions": shadow_joint_positions,
        }

    def _get_reward(self):
        object_dof = self.object.data["joint_positions"][0]
        object_dof_normalized = (
                                    object_dof - self.object.data["joint_lower_limit"][0]
                                    ) / (
                                    self.object.data["joint_upper_limit"][0]
                                    - self.object.data["joint_lower_limit"][0]
                                 )

        object_height = self.object.data['positions'][0][1] - self.obj_initial_height
        lift_reward = 10 * np.clip(object_height, 0, 0.15)

        obj_joint_delta_normalized = object_dof_normalized - self.obj_initial_joint_normalized
        goal_reward = (0 if object_height < 0.05 else 1) * (80 * obj_joint_delta_normalized)

        hand_object_dist = (np.linalg.norm(
            np.asarray(self.object.data['positions'][0]) - np.asarray(self.shadow.data['positions'][29])))
        drop_penalty = 10 * hand_object_dist

        success = object_height > 0.1 # and obj_joint_delta_normalized > 0.6
        success_reward = 50.0 if success else 0

        reward = lift_reward + goal_reward + success_reward - drop_penalty

        self.success = self.success or success
        self.object_height = object_height
        self.dof_delta = obj_joint_delta_normalized
        return reward

    def step_async(self, action):
        action = np.clip(action, -1, 1)
        assert self.action_space.contains(action)
        recon_action = (action + 1) / 2 * (
                self.upper_action - self.lower_action
        ) + self.lower_action
        base_action = recon_action[:5]
        shadow_action = recon_action[5:]
        self.base.SetJointPosition(base_action.tolist())
        self.rfu.SendObject("WorldMove", self.id, base_action[0], base_action[1], base_action[2])
        # self.rfu.SendObject("WorldMove", self.id, 0., 0.1, 0.)
        self.rfu.SendObject("RootRotate", self.id, base_action[3:])
        self.shadow.SetJointPosition(shadow_action.tolist())

    def step_wait(self):
        obs = self._get_obs()
        reward = self._get_reward()
        done = False
        return obs, reward, done, {"obj_id": self.obj_id, "pose_id": self.pose_id, "success": self.success}


class RFUVecEnv(DummyVecEnv):
    def __init__(
            self,
            make_env,
            num_envs,
            graphics=False,
            horizon=100,
            shared_data=None,
            **kwargs,
    ):
        self.num_envs = num_envs
        import os.path as osp
        self.rfu = RFUniverseBaseEnv(
            executable_file=osp.join(osp.split(__file__)[0], "../assets/envs/ballpoint/ballpoint_env.x86_64"),
            # executable_file="../rfu/win/GraspTest.exe",
            # executable_file=None,
            graphics=graphics,
            communication_backend = "grpc",
            log_level=0,
            **kwargs,
        )
        self.rfu.SendObject("Parallel", self.num_envs)
        self.reset_done = np.zeros((self.num_envs,), dtype=bool)
        self.rfu.AddListenerObject("ResetDone", self._handle_reset_done)
        self.rfu.step()
        self.envs = [
            make_env(self.rfu, i, shared_data, **kwargs) for i in range(num_envs)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.step_ctr = 0
        self.horizon = horizon
        self.reset_step_limit = 10000

        self.keys, self.shapes, dtypes = obs_space_info(self.observation_space)
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs,) + tuple(self.shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.proc_id = kwargs["proc_id"]

    def _handle_reset_done(self, obj):
        id = obj[0]
        collision = obj[1]
        # if collision == False:
        self.reset_done[id] = True
        # else:
        #     self.envs[id].reset_async()

    def reset(self):
        for i in range(self.num_envs):
            self.envs[i].reset_async()
        self.reset_done = np.zeros((self.num_envs,), dtype=bool)

        while not np.all(self.reset_done):
            self.rfu.step()

        self.rfu.step()

        self.step_ctr = 0  # reset episode steps counter

        for i in range(self.num_envs):
            obs = self.envs[i].reset_wait()
            self._save_obs(i, obs)

        return self._obs_from_buf()

    def reset_by_ids(self, obj_ids: List[str], tog_ids: List[int]):
        assert len(obj_ids) == self.num_envs
        assert len(tog_ids) == self.num_envs

        for i in range(self.num_envs):
            self.envs[i].reset_by_idx(obj_ids[i], tog_ids[i])
        self.reset_done = np.zeros((self.num_envs,), dtype=bool)

        while not np.all(self.reset_done):
            self.rfu.step()

        self.rfu.step()

        self.step_ctr = 0  # reset episode steps counter

        for i in range(self.num_envs):
            obs = self.envs[i].reset_wait()
            self._save_obs(i, obs)

        return self._obs_from_buf()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        # print(f'RFU ENV STEP BEGIN: {self.proc_id}')
        if self.step_ctr >= self.horizon:
            obs = self.reset()
            return (
                obs,
                np.copy(self.buf_rews),
                np.copy(self.buf_dones),
                deepcopy(self.buf_infos),
            )
        self.step_async(actions)
        # print(f'RFU ENV STEP processing: {self.proc_id}')
        ret = self.step_wait()
        # print(f'RFU ENV STEP END: {self.proc_id}')
        return ret

    def step_async(self, actions):
        actions = actions.reshape(self.num_envs, -1)
        for i in range(self.num_envs):
            self.envs[i].step_async(actions[i])  # send actions
        obj_pos = self.rfu.GetAttr(111000).data["positions"][0]
        self.rfu.ViewLookAt(obj_pos)
        self.rfu.step(simulate = False)  # simulator step

    def step_wait(self):
        for env_idx in range(self.num_envs):
            (
                obs,
                self.buf_rews[env_idx],
                self.buf_dones[env_idx],
                self.buf_infos[env_idx],
            ) = self.envs[env_idx].step_wait()
            # TODO: reset on dones
            self._save_obs(env_idx, obs)

        self.step_ctr += 1
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self):
        self.rfu.close()

    def __getattr__(self, attr_name: str):
        return [getattr(env, attr_name) for env in self.envs]



def BallpointVecEnv(num_envs, proc_id=0, shared_data=None, **kwargs):
    import os
    graphics = True if os.environ.get("DISPLAY", None) is not None and proc_id == 0 else False
    # graphics = False
    return RFUVecEnv(
        BallpointChildEnv,
        num_envs,
        graphics=graphics,
        proc_id=proc_id,
        shared_data=shared_data,
        **kwargs,
    )
