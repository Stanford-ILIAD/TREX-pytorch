from gym.envs.registration import register

register(
    id='CustomHopper-v0',
    entry_point='hopper.hopper_v3:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
