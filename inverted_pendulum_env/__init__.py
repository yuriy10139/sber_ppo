import gymnasium as gym

gym.envs.registration.register(
    id="InvertedPendulumEnv-v0",
    entry_point="inverted_pendulum_env.envs:InvertedPendulumEnv",
)
