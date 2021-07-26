from gym.envs.registration import register

register(
    id='CustomWalker2d-v0',
    max_episode_steps=1000,
    entry_point='walker2d.walker2d_v3:Walker2dEnv',
)
