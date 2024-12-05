import argparse
import collections
from collections import defaultdict
from typing import Optional, Tuple

from magent2.environments import battle_v4
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce
import gymnasium as gym
import pettingzoo
from tqdm import tqdm

from cmarl.utils.team import TeamManager

USE_WANDB = False  # if enabled, logs data on wandb server

@dataclass
class VdnHyperparameters:
    lr: float = 0.001


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition: list[dict[str, any]]):
        """Update buffer with a new transition step
        :param transition: tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)

    def sample_chunk(self, batch_size: int, chunk_size: int, agents: list[str]) -> dict[str, any]:
        """Sample a batch of chunk_size transitions from the buffer
        :param batch_size: max number of batches to sample
        :param chunk_size: max length of horizon of each batch
        :return: dict of (states, actions, rewards, next_states, dones),
        their shapes are respectively:


        start_idx = [0, 11, 20]
        chunk_size = 5
        each agent should have = [[0, 1, 2, 3,4], [11, 12, 13, 14, 15], [20, 21, 22, 23, 24]]
        actual result ={
            red_0: [[0, 1, 2, 3, 4], [11, 12, 13, 14, 15], [20, 21, 22, 23, 24]],
            red_1: [[0, 1, 2, 3, 4], [11, 12, 13, 14, x], [x, x, 31, 32, 33]],
        }
        """
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size) # [batch_size]
        states, actions, rewards, next_states, dones = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

        for idx in start_idx:   # [batch_size]
            # We need a placeholder for each agent
            # { agent: [batch_size * [chunk_size * Tensor(n)]] }
            for agent in agents:
                states[agent].append([])
                actions[agent].append([])
                rewards[agent].append([])
                next_states[agent].append([])
                dones[agent].append([])

            for chunk_step in range(idx, idx + chunk_size): # [chunk_size]
                # The transition of agents at each step
                # state, action, reward, next_state, done
                agent_states, agent_actions, agent_rewards, agent_next_states, agent_dones = self.buffer[chunk_step]
                for agent in agent_states.keys():
                    states[agent][-1].append(agent_states[agent])
                    actions[agent][-1].append(agent_actions[agent])
                    rewards[agent][-1].append(agent_rewards[agent])
                    next_states[agent][-1].append(agent_next_states[agent])
                    dones[agent][-1].append(agent_dones[agent])

        # Convert to {agent:
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, agents: list[str], observation_spaces: dict[str, gym.spaces.Space], action_spaces: dict[str, gym.spaces.Space], recurrent=False):
        super(QNet, self).__init__()
        self.agents = agents
        self.num_agents = len(agents)
        self.recurrent = recurrent
        self.hx_size = 32   # latent repr size
        self.n_obs_map = {agent: reduce((lambda x, y: x * y), observation_spaces[agent].shape) for agent in agents}  # observation space flatten size of agents
        self.n_act_map = {agent: action_spaces[agent].n for agent in agents}  # action space size of agents
        self.idx_map = {agent: i for i, agent in enumerate(agents)}

        for agent_i in agents:
            # flatten magent env observation space
            n_obs = self.n_obs_map[agent_i]
            idx = self.idx_map[agent_i]
            setattr(
                self, 'agent_feature_{}'.format(agent_i),   # shape: n_obs, hx_size
                nn.Sequential(
                    nn.Linear(n_obs, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.hx_size),
                    nn.ReLU()
                )
            )
            if recurrent:
                setattr(
                    self, 'agent_gru_{}'.format(agent_i),   # shape: hx_size, hx_size
                    nn.GRUCell(self.hx_size, self.hx_size)
                )
            setattr(
                self, 'agent_q_{}'.format(agent_i),     # shape: hx_size, n_actions
                nn.Linear(self.hx_size, self.n_act_map[agent_i])
            )

    def forward(self, obs: dict[str, torch.Tensor], hidden: dict[str, torch.Tensor]) -> (dict[str, torch.Tensor], dict[str, torch.Tensor]):
        """Predict q values for each agent's actions in the batch
        :param obs: {agent: [batch_size, n_obs]}
        :param hidden: {agent: [batch_size, hx_size]}
        :return: q_values: {agent: [batch_size, n_actions]}, hidden: {agent: [batch_size, hx_size]}
        """
        # TODO: we may have changing num_agents
        q_values = {}  # {agent: (batch_size, n_actions)}
        next_hidden = {}  # {agent: (batch_size, hx_size)}

        """
        red_01 (3 batch: 0, 1, 2) -> qval of (0, 1, 2) -> act of (0, 1, 2)
        red_02 (2 batch: 0, 2   ) -> qval of (0, 2)    -> act of (0, 1, 2)
        """

        for i, agent in enumerate(obs.keys()):
            x = getattr(self, 'agent_feature_{}'.format(agent))(obs[agent]) # [batch_size, n_obs] -> [batch_size, hx_size]
            if self.recurrent:
                x = getattr(self, 'agent_gru_{}'.format(agent))(x, hidden[:, i, :])   # [batch_size, hx_size]
                next_hidden[agent] = x  # [batch_size, hx_size]
            q_values[agent] = getattr(self, 'agent_q_{}'.format(agent))(x)   # [batch_size, n_actions]
        return q_values, next_hidden

    def sample_action(self, obs: dict[str, torch.Tensor], hidden: dict[str, torch.Tensor], epsilon=1e3) -> (dict[str, torch.Tensor], dict[str, torch.Tensor]):
        """Choose action with epsilon-greedy policy, for each agent in the batch
        :param obs: a batch of observations for each agent, {agent: [batch_size, n_obs]}
        :param hidden: a batch of hidden states for each agent, {agent: [batch_size, hx_size]}
        :param epsilon: exploration rate
        :return: actions: {agent: (batch, action)}, hidden: {agent: (batch, hx_size)}
        """
        q_values, hidden = self.forward(obs, hidden)    # {agent: [batch_size, n_actions]}, {agent: [batch_size, hx_size]}
        # epsilon-greedy action selection
        max_batch_size = max([q_values[agent].shape[0] for agent in q_values.keys()])
        masked_batch = np.random.rand(max_batch_size) < epsilon  # Create a mask for the entire batch
        actions = {}
        for agent in q_values.keys():
            agent_batch_size = q_values[agent].shape[0]
            actions[agent] = torch.empty((agent_batch_size,), dtype=torch.long)
            # Truncate the mask to the agent's batch size
            agent_masks = masked_batch[:agent_batch_size]
            # Random actions for masked batches
            actions[agent][agent_masks] = torch.randint(0, self.n_act_map[agent], (agent_masks.sum(),))
            # Best actions for non-masked batches
            actions[agent][~agent_masks] = q_values[agent][~agent_masks].argmax(dim=1)
        return actions, hidden   # [batch_size, num_agents], [batch_size, num_agents, hx_size]

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))


def train(q: QNet, q_target: QNet, memory: ReplayBuffer, optimizer: optim.Optimizer, gamma: float, batch_size: int, update_iter=10, chunk_size=10, grad_clip_norm=5):
    q.train()
    q_target.eval()
    chunk_size = chunk_size if q.recurrent else 1
    for _ in range(update_iter):
        # Get data from buffer
        states, actions, rewards, next_states, dones = memory.sample_chunk(batch_size, chunk_size)

        hidden = q.init_hidden(batch_size)
        target_hidden = q_target.init_hidden(batch_size)
        loss = 0
        for step_i in range(chunk_size):
            out: tuple[torch.Tensor, torch.Tensor] = q(states[:, step_i, :, :], hidden)  # [batch_size, num_agents, n_actions]
            q_out, hidden = out
            q_a = q_out.gather(2, actions[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)   # [batch_size, num_agents]: q values of actions taken
            # TODO: Using mean instead of sum, because num_agents may change
            sum_q = q_a.mean(dim=1, keepdims=True)   # [batch_size, 1]

            max_q_prime, target_hidden = q_target(next_states[:, step_i, :, :], target_hidden.detach())
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
            target_q = rewards[:, step_i, :].sum(dim=1, keepdims=True)  # [batch_size, 1]
            target_q += gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - dones[:, step_i])

            loss += F.smooth_l1_loss(sum_q, target_q.detach())

            # FIXME: may have error with dones
            done_mask = dones[:, step_i].squeeze(-1).bool()
            hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))
            target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()


def run_episode(env: pettingzoo.ParallelEnv, q: QNet, memory: Optional[ReplayBuffer] = None, epsilon=0.1) -> float:
    """Run an episode in the environment
    :return: total score of the episode
    """
    def flatten_observation(observations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {agent: obs.flatten() for agent, obs in observations.items()}

    observations: dict[str, np.ndarray] = env.reset()
    team_manager = TeamManager(env.agents)
    teams = team_manager.get_teams()
    my_team = teams[0]
    hidden = q.init_hidden()
    score = 0.0
    flatten_observations = flatten_observation(observations)

    while not team_manager.has_terminated_teams():
        my_team_observations = team_manager.get_info_of_team(my_team, flatten_observations)
        for agent in my_team_observations.keys():
            if my_team_observations[agent] is None:
                my_team_observations[agent] = torch.zeros(q.n_obs_map[agent])
        # Get actions for each agent based on the team
        agent_actions: dict[str, Optional[int]] = {}  # {agent: action}
        for team in teams:
            if team == my_team:
                # TODO: Fill rows with zeros for terminated agents
                team_observations = torch.tensor([my_team_observations[agent] for agent in team_manager.get_team_agents(team)]).unsqueeze(0)
                # team_hidden = team_manager.get_info_of_team(team, hidden) TODO: we need a mapping from agent to index
                actions, team_hiddens = q.sample_action(team_observations, hidden, epsilon)
                team_actions = {
                    agent: action
                    for agent, action in zip(
                        my_team_observations.keys(), actions.squeeze(0).data.numpy().tolist()
                    )
                }
            else:
                # TODO: selfplay
                # Opponents choose random actions
                team_actions = {agent: env.action_space(agent).sample() for agent in team_manager.get_team_agents(team)}
            agent_actions.update(team_actions)

        # Terminated agents use None action
        for agent in team_manager.terminated_agents:
            agent_actions[agent] = None

        # Step the environment
        observations, agent_rewards, agent_terminations, agent_truncations, agent_infos = env.step(agent_actions)
        score += sum(team_manager.get_info_of_team(my_team, agent_rewards, 0).values())
        flatten_observations = flatten_observation(observations)

        if memory is not None:
            memory.put((
                list(my_team_observations.values()),
                list(team_manager.get_info_of_team(my_team, agent_actions).values()),
                list(team_manager.get_info_of_team(my_team, agent_rewards, 0).values()),
                list(team_manager.get_info_of_team(my_team, flatten_observations).values()),
                [int(team_manager.has_terminated_teams())]
            ))

        # Check for termination
        for agent, done in agent_terminations.items():
            if done:
                team_manager.terminate_agent(agent)
        for agent, done in agent_truncations.items():
            if done:
                team_manager.terminate_agent(agent)
    return score

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
        score += run_episode(env, q)
    return score / num_episodes


def main(
        env: pettingzoo.ParallelEnv, test_env: pettingzoo.ParallelEnv,
        lr, gamma, batch_size, buffer_limit, log_interval, max_episodes, max_epsilon, min_epsilon,
        test_episodes, warm_up_steps, update_iter, chunk_size, update_target_interval, recurrent: bool = False
):
    # create env.
    memory = ReplayBuffer(buffer_limit)

    # Setup env
    env.reset()
    team_manager = TeamManager(env.agents)
    my_team = team_manager.get_teams()[0]

    # create networks
    q = QNet(team_manager.get_team_agents(my_team), env.observation_spaces, env.action_spaces, recurrent)
    q_target = QNet(team_manager.get_team_agents(my_team), env.observation_spaces, env.action_spaces, recurrent)
    q_target.load_state_dict(q.state_dict())
    optimizer = optim.Adam(q.parameters(), lr=lr)

    score = 0
    for episode_i in tqdm(range(max_episodes)):
        epsilon = max(min_epsilon, max_epsilon - (max_epsilon - min_epsilon) * (episode_i / (0.6 * max_episodes)))
        with torch.no_grad():
            score += run_episode(env, q, memory, epsilon)

        if memory.size() > warm_up_steps:
            train(q, q_target, memory, optimizer, gamma, batch_size, update_iter, chunk_size)

        if episode_i % update_target_interval:
            q_target.load_state_dict(q.state_dict())

        if (episode_i + 1) % log_interval == 0:
            test_score = test(test_env, test_episodes, q)
            train_score = score / log_interval
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, eps : {:.1f}"
                  .format(episode_i, max_episodes, train_score, test_score, memory.size(), epsilon))
            score = 0

    env.close()
    test_env.close()


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Value Decomposition Network (VDN)')
    parser.add_argument('--env-name', required=False, default='ma_gym:Checkers-v0')
    parser.add_argument('--seed', type=int, default=42, required=False)
    parser.add_argument('--no-recurrent', action='store_true')
    parser.add_argument('--max-episodes', type=int, default=150, required=False)

    # Process arguments
    args = parser.parse_args()

    kwargs = {
        "env": battle_v4.parallel_env(
            map_size=20, render_mode="human", max_cycles=5000, step_reward=0, attack_penalty=0.01
        ),
        "test_env": battle_v4.parallel_env(
            map_size=20, render_mode="human", max_cycles=5000, step_reward=0, attack_penalty=0.01
        ),
        "lr": 0.001,
        "batch_size": 32,
        "gamma": 0.99,
        "buffer_limit": 5000,
        "update_target_interval": 20,
        "log_interval": 100,
        "max_episodes": args.max_episodes,
        "max_epsilon": 0.9,
        "min_epsilon": 0.1,
        "test_episodes": 5,
        "warm_up_steps": 20,
        "update_iter": 10,
        "chunk_size": 10,  # if not recurrent, internally, we use chunk_size of 1 and no gru cell is used.
        "recurrent": False,
    }

    main(**kwargs)
