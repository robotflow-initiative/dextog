import asyncio
from pyrfuniverse.envs.gym_wrapper_env import RFUniverseGymWrapper
import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from pyrfuniverse.envs.base_env import RFUniverseBaseEnv
from envs.comm import RFUniverseCommManager


class _BottleCapEnv:
    def __init__(self, idx, comm_manager: RFUniverseCommManager):
        self.idx = idx
        self.comm_manager = comm_manager
        self.observation_space = gym.spaces.Dict(
            {
                "shadow_cap_relative_positions": gym.spaces.Box(
                    low=-1, high=1, shape=(29,)
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(29,), dtype=np.float32
        )
        self.env_attributes = self.comm_manager.get_env_attributes(self.idx)

        self.shadow_upper_action = np.array(self.env_attributes.shadow.data["joint_upper_limit"])
        self.shadow_lower_action = np.array(self.env_attributes.shadow.data["joint_lower_limit"])
        self.base_upper_action = np.array(self.env_attributes.base.data["joint_upper_limit"])
        self.base_lower_action = np.array(self.env_attributes.base.data["joint_lower_limit"])
        self.upper_action = np.concatenate(
            [self.base_upper_action, self.shadow_upper_action]
        )
        self.lower_action = np.concatenate(
            [self.base_lower_action, self.shadow_lower_action]
        )
        self.default_bottle_joint_positions = [
            0,
            0,
            0,
            30,
            30,
            30,
            30,
            0,
            0,
            30,
            30,
            30,
            0,
            30,
            30,
            30,
            0,
            30,
            30,
            30,
            0,
            30,
            30,
            30,
        ]

    def reset(self):
        # self.comm_manager.reset_env(self.idx, self.default_bottle_joint_positions)
        return self._get_obs()

    def step(self, action):
        recon_action = (action + 1) / 2 * (
            self.upper_action - self.lower_action
        ) + self.lower_action
        base_action = recon_action[:5]
        shadow_action = recon_action[5:]
        self.env_attributes.base.SetJointPosition(base_action.tolist())
        self.env_attributes.shadow.SetJointPosition(shadow_action.tolist())


        asyncio.create_task(self.comm_manager.sim_step(self.idx))
        
        return self._get_obs(), self._get_reward(), False, {}

    def _get_obs(self):
        shadow_positions = np.array(self.env_attributes.shadow.data["positions"])
        cap_positions = np.array(self.env_attributes.stapler.data["positions"])[1]
        return {"shadow_cap_relative_positions": shadow_positions - cap_positions}

    def _get_reward(self):
        # Calculate and return reward. This method needs to be implemented based on the specific reward logic.
        return 0


class BottleCapEnv(RFUniverseBaseEnv, DummyVecEnv):
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        assets: list = [],
        num_envs: int = 10,
        **kwargs
    ):
        RFUniverseBaseEnv.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            assets=assets,
            **kwargs,
        )
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(29,))
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(29,)
        )  # base * 5, shadow * 24

        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(29,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(29,), dtype=np.float32
        )

        # Initialize buffers
        self.buf_obs = np.zeros(
            (self.num_envs,) + self.observation_space.shape, dtype=np.float32
        )
        self.buf_rews = np.zeros(self.num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(self.num_envs, dtype=bool)
        self.buf_infos = [{} for _ in range(self.num_envs)]

        self.SendObject("Parallel", self.num_envs)
        self.Pend()
        self._step()
        # self._get_obs()
        self.base = self.GetAttr(123 * 1000)
        self.bottle = self.GetAttr(111 * 1000)
        self.shadow = self.GetAttr(222 * 1000)

        self.default_bottle_joint_positions = [
            0,
            0,
            0,
            30,
            30,
            30,
            30,
            0,
            0,
            30,
            30,
            30,
            0,
            30,
            30,
            30,
            0,
            30,
            30,
            30,
            0,
            30,
            30,
            30,
        ]

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

    def reset(self, seed=None, options=None):
        observations = []
        for env_idx in range(self.num_envs):
            obs, _ = self.reset_single_env(env_idx)
            observations.append(obs)
        self.buf_obs = np.array(observations)
        self.buf_dones[:] = False
        return self.buf_obs.copy()

    def reset_single_env(self, env_idx):
        self.bottle.SetJointPosition(self.default_bottle_joint_positions)
        obs = self._get_obs(env_idx)
        info = {}
        return obs, info

    def _get_obs(self, env_idx=0):
        self.shadow_positions = np.array(
            self.shadow.data["positions"]
        )  # [8, 14, 19, 24, 29]
        self.bottle_cap_position = np.array(self.bottle.data["positions"])[1]
        return self.shadow_positions - self.bottle_cap_position

    def _get_reward(self):
        cap_dof = self.bottle.data["joint_positions"][0]
        fingertip_positions = np.array(self.shadow.data["positions"])[
            np.array([8, 14, 19, 24, 29]) - 1
        ]
        bottle_cap_position = np.array(self.bottle.data["positions"])[1]
        avg_fingertip_distance = np.mean(
            np.linalg.norm(fingertip_positions - bottle_cap_position, axis=1)
        )

        return -avg_fingertip_distance + 0.1 * cap_dof

    def step(self, action):

        recon_action = (action + 1) / 2 * (
            self.upper_action - self.lower_action
        ) + self.lower_action
        base_action = recon_action[:5]
        shadow_action = recon_action[5:]
        # self.shadow.SetJointPosition(shadow_action.tolist())
        # self.base.SetJointPosition(base_action.tolist())
        self.shadow.SetJointPosition(
            (np.array(self.shadow.data["joint_positions"])).tolist()
        )
        self.base.SetJointPosition(
            (np.array(self.base.data["joint_positions"])).tolist()
        )
        self._step()

        obs = self._get_obs()
        reward = self._get_reward()
        done = False
        info = {}
        return obs, reward, done, info
