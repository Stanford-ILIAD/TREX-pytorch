import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils


DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': 2,
    'distance': 3.0,
    'lookat': np.array((0.0, 0.0, 1.15)),
    'elevation': -20.0,
}

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 reward_mode='state_only',
                 xml_file='hopper.xml',
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_state_range=(-100.0, 100.0),
                 healthy_z_range=(0.7, float('inf')),
                 healthy_angle_range=(-0.2, 0.2),
                 reset_noise_scale=5e-3,
                 exclude_current_positions_from_observation=True,
                 reward_net=None):
        utils.EzPickle.__init__(**locals())

        self.reward_net = reward_net
        self.reward_mode = reward_mode

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        mujoco_env.MujocoEnv.__init__(self, xml_file, 4)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.sim.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(
            np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = np.clip(
            self.sim.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def set_reward_net(self, reward_net):
        self.reward_net = reward_net

    def step(self, action):
        observation_before = self._get_obs()
        self.do_simulation(action, self.frame_skip)

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        if self.reward_net is not None:
            if self.reward_mode == 'state_only':
                net_reward = self.reward_net.compute_reward(observation)
            elif self.reward_mode == 'state_pair':
                net_reward = self.reward_net.compute_reward(np.concatenate([observation_before, obseravtion], axis=0))
            elif self.reward_mode == 'state_action':
                net_reward = self.reward_net.compute_reward(np.concatenate([observation_before, np.clip(action, -1., 1.)], axis=0))
            elif self.reward_mode == 'state_action_state':
                net_reward = self.reward_net.compute_reward(np.concatenate([observation_before, np.clip(action, -1., 1.), observation], axis=0))
            reward = net_reward# - ctrl_cost
        else:
            reward = 0
        done = self.done
        info = {
        }
        
        return observation, reward, done, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
