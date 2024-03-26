import os
import time
from dataclasses import dataclass

import gymnasium as gym
import inverted_pendulum_env

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.distributions.normal import Normal


@dataclass
class Args:
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    env_id: str = "InvertedPendulumEnv-v0"
    """the id of the environment"""
    num_runs: int = 3
    """number of eval runs"""
    num_steps: int = 1024
    """the number of steps to run in a single eval rollout"""
    set_target: bool = False
    """a hard scenario with a target where to balance the pendulum"""
    set_target_step: int = 0
    """Number of step in a single env to generate a target"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    record_video_from: int = 0
    """record video starting from this step"""
    model_path: str = (
        "runs/InvertedPendulumEnv-v0__ppo_cont_action__3__1710928856/ppo_cont_action.cleanrl_model"
    )
    """path to saved model weights"""
    set_x_coordinate: float = None
    """X-coordinate of the target to stabilize a pendulum"""
    randomize_mass: bool = False
    """whether to randomize pole mass"""
    set_mass: float = 0.0
    """the value of pendulum's mass"""


def make_env(
    env_id,
    idx,
    capture_video,
    run_name,
    num_steps,
    record_video_from,
    set_target,
    set_target_step,
    set_x_coordinate,
    randomize_mass,
    set_mass,
):
    def thunk():
        env = gym.make(
            env_id,
            max_steps=num_steps,
            record_video=capture_video,
            run_name=run_name,
            record_video_from=record_video_from,
            set_target=set_target,
            set_target_step=set_target_step,
            x_coordinate=set_x_coordinate,
            randomize_mass=randomize_mass,
            set_mass=set_mass
        )
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.ClipAction(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.set_x_coordinate is not None:
        assert -1.0 <= args.set_x_coordinate <= 1.0, "coordinate must be in [-1.0, 1.0]"
    run_name = f"eval_run_{args.env_id}__{int(time.time())}"
    
    if args.randomize_mass:
        assert args.set_mass == 0.0, "either randomize_mass or set_mass should be used"

    if not os.path.exists(f"runs/{run_name}"):
        os.makedirs(f"runs/{run_name}")

    #device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device = torch.device("cpu") #works faster on such a small net

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.capture_video,
                run_name,
                args.num_steps,
                args.record_video_from,
                args.set_target,
                args.set_target_step,
                args.set_x_coordinate,
                args.randomize_mass,
                args.set_mass,
            )
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    for eval_run in range(args.num_runs):
        for step in range(0, args.num_steps):
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
            obs = next_obs
        episodic_returns.append(envs.envs[0].unwrapped.rewards_raw.sum())
        print(f"eval run {eval_run} return: {episodic_returns[-1]}")
    envs.close()
