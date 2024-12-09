import collections
import random
from typing import Optional

import numpy as np
import pettingzoo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from cmarl.utils import compute_output_dim, reseed, TeamManager, has_today_model, today, load_model, save_model
from cmarl.utils.buffer import ReplayBuffer
import gymnasium as gym

from cmarl.utils.env import envs_config

seed = 42

class QNet(nn.Module):
    model_name = 'IDQN'

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super(QNet, self).__init__()
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


def run_episode(env: pettingzoo.ParallelEnv, q: QNet, memory: Optional[ReplayBuffer]=None, epsilon: float=0):
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


def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10):
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


def test(env: pettingzoo.ParallelEnv, num_episodes: int, q: QNet):
    """
    :param env: Environment
    :param num_episodes: How many episodes to test
    :param q: Trained QNet
    :return: average score over num_episodes
    """
    q.eval()
    score = 0
    for episode_i in range(num_episodes):
        score += run_episode(env, q, epsilon=0)
    return score / num_episodes


def main(
    env: pettingzoo.ParallelEnv, test_env: pettingzoo.ParallelEnv,
    lr, gamma, batch_size, buffer_limit, log_interval, max_episodes,
    max_epsilon, min_epsilon, test_episodes, warm_up_steps, update_iter
):
    reseed(seed)
    # create env.
    memory = ReplayBuffer(buffer_limit)

    # Setup env
    test_env.reset(seed=seed)
    env.reset(seed=seed)
    team_manager = TeamManager(env.agents)
    my_team = team_manager.get_my_team()
    print(my_team)

    # create networks
    any_agent = team_manager.get_team_agents(my_team)[0]
    q = QNet(env.observation_spaces[any_agent], env.action_spaces[any_agent])
    q_target = QNet(env.observation_spaces[any_agent], env.action_spaces[any_agent])
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    # Load model if exists
    test_score = 0
    if has_today_model(f'idqn-{today}'):
        q_test = load_model(
            QNet(env.observation_spaces[any_agent], env.action_spaces[any_agent]),
            f'idqn-{today}')
        test_score = test(test_env, test_episodes, q_test)
        print("Model loaded. Test score: ", test_score)

    score = 0
    for episode_i in tqdm(range(max_episodes)):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * max_episodes)))
        q.eval()
        score += run_episode(env, q, memory, epsilon)

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter)

        if (episode_i + 1) % log_interval == 0:
            q_target.load_state_dict(q.state_dict())
            print("Test phase:")
            prev_test_score = test_score
            test_score = test(test_env, test_episodes, q)
            if test_score > prev_test_score:
                save_model(q, f'idqn-{today}')
                print("Model saved at episode: ", episode_i)
            print("Test score: ", test_score)

            train_score = score / log_interval
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, train_score, test_score, memory.size(), epsilon))
            score = 0
            print('#' * 30)

    env.close()
    test_env.close()


if __name__ == '__main__':
    env = envs_config['adversarial_pursuit']
    kwargs = {
        "env": env["module"].parallel_env(**env["args"]),
        "test_env": env["module"].parallel_env(**env["args"]),
        "lr": 0.001,
        "batch_size": 32,
        "gamma": 0.99,
        "buffer_limit": 9000,
        "log_interval": 10,
        "max_episodes": 150,
        'max_epsilon': 0.9,
        'min_epsilon': 0.1,
        'test_episodes': 5,
        "warm_up_steps": 3000,
        "update_iter": 20,  # epochs
    }
    main(**kwargs)