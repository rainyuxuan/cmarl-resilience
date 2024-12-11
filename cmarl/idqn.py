import argparse
import collections
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pettingzoo
import torch
import torch.nn as nn
import torch.nn.functional as F

from cmarl.runner import Hyperparameters, evaluate_model, run_model_train_test, run_experiment
from cmarl.utils import compute_output_dim, reseed, TeamManager, is_model_found, today, load_model, save_model, \
    save_data, save_dict
from cmarl.utils.buffer import ReplayBuffer
import gymnasium as gym

from cmarl.utils.env import envs_config

seed = 42

@dataclass
class IdqnHyperparameters(Hyperparameters):
    chunk_size = 1


class IqdnQNet(nn.Module):
    model_name = 'IDQN'

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super(IqdnQNet, self).__init__()
        self.hx_size = 32  # latent repr size
        self.observation_space = observation_space  # observation space of agents
        self.action_space = action_space  # action space of agents
        self.n_obs = observation_space.shape  # observation space size of agents
        self.n_act = action_space.n  # action space size of agents

        stride1, stride2 = 1, 1
        padding1, padding2 = 0, 0
        kernel_size1, kernel_size2 = 3, 3
        pool_kernel_size, pool_stride = 2, 2

        height = self.n_obs[0]  # n_obs is a tuple (height, width, channels)
        out_dim1 = compute_output_dim(height, kernel_size1, stride1, padding1) // pool_stride
        out_dim2 = compute_output_dim(out_dim1, kernel_size2, stride2, padding2) // pool_stride

        # Compute the final flattened size
        flattened_size = out_dim2 * out_dim2 * 64
        self.feature_cnn = nn.Sequential(
            nn.Conv2d(self.n_obs[2], 32, kernel_size=kernel_size1, stride=stride1, padding=padding1),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.Conv2d(32, 64, kernel_size=kernel_size2, stride=stride2, padding=padding2),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride),
            nn.Flatten(),
            nn.Linear(flattened_size, self.hx_size),
            nn.ReLU()
        )
        self.q_val = nn.Linear(self.hx_size, self.n_act)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the Q-values of one agent's observation.
        :param obs:
        :return: shape (batch_size, n_act)
        """
        obs = obs.permute(0, 3, 1, 2)
        hx = self.feature_cnn(obs)
        return self.q_val(hx)

    def sample_action(self, obs: torch.Tensor, epsilon: float):
        """
        Sample an action from the Q-values.
        :param obs:
        :param epsilon:
        :return:
        """
        if random.random() < epsilon:
            return self.action_space.sample()
        else:
            return self.forward(obs).argmax(dim=1)


def run_episode(env: pettingzoo.ParallelEnv, q: IqdnQNet, memory: Optional[ReplayBuffer]=None, epsilon: float=0, random_rate: float=0.0):
    observations: dict[str, np.ndarray] = env.reset()
    team_manager = TeamManager(env.agents)
    teams = team_manager.get_teams()
    my_team = team_manager.get_my_team()
    score = 0.0

    while not team_manager.has_terminated_teams():
        my_team_observations = team_manager.get_info_of_team(my_team, observations)
        # Get actions for each agent based on the team
        agent_actions: dict[str, Optional[int]] = {}  # {agent: action}
        for team in teams:
            if team == my_team:
                team_observations = torch.tensor(np.array([
                    my_team_observations[agent]
                    for agent in team_manager.get_team_agents(team)
                ])).unsqueeze(0)  # [batch_size=1, num_agents, n_obs]

                team_actions = {
                    agent: q.sample_action(team_observations[:, i], epsilon).item()
                    for i, agent in enumerate(team_manager.get_team_agents(team))
                }

                if random_rate > 0:
                    random_agents = team_manager.get_random_agents(random_rate)
                    for agent in random_agents:
                        team_actions[agent] = env.action_space(agent).sample()
            else:
                # Opponents choose random actions
                team_actions = {agent: env.action_space(agent).sample() for agent in team_manager.get_team_agents(team)}
            agent_actions.update(team_actions)

        # Terminated agents use None action
        for agent in team_manager.terminated_agents:
            agent_actions[agent] = None

        # Step the environment
        observations, agent_rewards, agent_terminations, agent_truncations, agent_infos = env.step(agent_actions)
        score += sum(team_manager.get_info_of_team(my_team, agent_rewards, 0).values())

        if memory is not None:
            memory.put((
                list(my_team_observations.values()),
                list(team_manager.get_info_of_team(my_team, agent_actions).values()),
                list(team_manager.get_info_of_team(my_team, agent_rewards, 0).values()),
                list(team_manager.get_info_of_team(my_team, observations).values()),
                [int(team_manager.has_terminated_teams())]
            ))

        # Check for termination
        for agent, done in agent_terminations.items():
            if done:
                team_manager.terminate_agent(agent)
        for agent, done in agent_truncations.items():
            if done:
                team_manager.terminate_agent(agent)
    print('Score:', score)
    return score


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=1):
    q.train()
    q_target.eval()
    chunk_size = 1
    losses = []
    for i in range(update_iter):
        # Get data from buffer
        states, actions, rewards, next_states, dones = memory.sample_chunk(batch_size, chunk_size)

        num_agents = states.shape[2]
        losses.append([])

        for agent_i in range(num_agents):
            s, a, r, s_prime, done_mask = states[:, 0, agent_i], actions[:, 0, agent_i], rewards[:, 0, agent_i], next_states[:, 0, agent_i], dones
            q_out = q(s)    # (batch_size, n_act)
            q_a = q_out.gather(1, a.long().unsqueeze(1)).squeeze(1) # (batch_size, 1) -> (batch_size)
            max_q_prime = q_target(s_prime).max(dim=1)[0]
            target = r + gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target.detach())

            losses[-1].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('Loss:', np.mean(losses[0]), np.mean(losses[-1]))
    return np.mean(losses, axis=1)


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Independent Deep Q-Network (IDQN)')
    parser.add_argument('--max-episodes', type=int, default=160, required=False)
    parser.add_argument('--batch', type=int, default=2048, required=False)
    parser.add_argument('--env', type=str, default='adversarial_pursuit', required=False)
    parser.add_argument('--load_model', type=str, default=None, required=False)
    parser.add_argument('--task', type=str, default='train', required=False, choices=['train', 'experiment'])

    # Process arguments
    args = parser.parse_args()
    env_cfg = envs_config[args.env]
    max_episodes = args.max_episodes
    batch_size = args.batch
    task = args.task
    save_name = f'idqn-{args.env}-{today}'
    loaded_model = args.load_model

    # Hyperparameters
    hp = IdqnHyperparameters(
        lr=0.001,
        gamma=0.99,
        batch_size=batch_size,
        buffer_limit=9000,
        log_interval=20,
        max_episodes=max_episodes,
        max_epsilon=0.9,
        min_epsilon=0.1,
        test_episodes=5,
        warm_up_steps=3000,
        update_iter=20,
        update_target_interval=20
    )

    # Create env
    env = env_cfg["module"].parallel_env(**env_cfg["args"])
    test_env = env_cfg["module"].parallel_env(**env_cfg["args"])
    env.reset(seed=seed)
    test_env.reset(seed=seed)
    team_manager = TeamManager(env.agents)

    # Create model
    any_agent = team_manager.get_my_agents()[0]
    q = IqdnQNet(env.observation_spaces[any_agent], env.action_spaces[any_agent])
    q_target = IqdnQNet(env.observation_spaces[any_agent], env.action_spaces[any_agent])

    # Do task
    if task == 'train':
        # Load model if exists
        if loaded_model is not None and is_model_found(loaded_model):
            q = load_model(q, loaded_model)
            test_score = evaluate_model(test_env, hp.test_episodes, q, run_episode)
            print("Pretrained Model loaded. Test score: ", test_score)

        # Train and test
        train_scores, test_scores, losses = run_model_train_test(
            env, test_env, IqdnQNet, q, q_target,
            save_name, team_manager, hp, train, run_episode
        )

        # Save data
        save_data(np.array(train_scores), f'{save_name}-train_scores')
        save_data(np.array(test_scores), f'{save_name}-test_scores')
        save_data(np.array(losses), f'{save_name}-losses')
    elif task == 'experiment':
        if loaded_model is None or not is_model_found(loaded_model):
            raise ValueError("Please provide a model to load for experiment.")

        q = load_model(q, loaded_model)

        # Run experiment
        rate_avg_scores, rate_scores = run_experiment(
            env, q, hp.test_episodes * 2, run_episode, num_tests=10
        )

        # Save data
        save_dict(rate_avg_scores, f'{save_name}-rate_avg_scores')
        save_dict(rate_scores, f'{save_name}-rate_scores')