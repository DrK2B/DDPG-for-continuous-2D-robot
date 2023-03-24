from gymnasium.envs.registration import register

register(
     id="2DRobot-v0",
     entry_point="gym_examples.2D_robot_env:Continuous_2D_RobotEnv",
     max_episode_steps=999,
)