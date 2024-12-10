import argparse
from typing import Optional

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import pettingzoo

from cmarl.runner import Hyperparameters, run_model_train_test, evaluate_model, seed, run_experiment
from cmarl.utils import compute_output_dim, load_model, is_model_found, today, save_data, save_dict
from cmarl.utils.env import envs_config
from cmarl.utils.team import TeamManager
from cmarl.utils.buffer import ReplayBuffer

@dataclass
class VdnHyperparameters(Hyperparameters):
    recurrent: bool = False


class VdnQNet(nn.Module):
    model_name  = "VDN"

    def __init__(self, agents: list[str], observation_spaces: dict[str, gym.spaces.Space], action_spaces: dict[str, gym.spaces.Space], recurrent=False):
        super(VdnQNet, self).__init__()
        self.agents = agents
        self.num_agents = len(agents)
        self.recurrent = recurrent
        self.hx_size = 32   # latent repr size
        self.n_obs = observation_spaces[agents[0]].shape    # observation space size of agents
        self.n_act = action_spaces[agents[0]].n  # action space size of agents

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
        if recurrent:
            self.gru =  nn.GRUCell(self.hx_size, self.hx_size)  # shape: hx_size, hx_size
        self.q_val = nn.Linear(self.hx_size, self.n_act)    # shape: hx_size, n_actions


    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Predict q values for each agent's actions in the batch
        :param obs: [batch_size, num_agents, ...n_obs]
        :param hidden: [batch_size, num_agents, hx_size]
        :return: q_values: [batch_size, num_agents, n_actions], hidden: [batch_size, num_agents, hx_size]
        """
        # TODO: need to have a done_mask param
        batch_size = obs.shape[0]
        q_values = torch.empty((batch_size, self.num_agents, self.n_act))
        next_hidden = torch.empty((batch_size, self.num_agents, self.hx_size))

        for i, agent_i in enumerate(self.agents):
            agent_obs = obs[:, i].permute(0, 3, 1, 2)   # [batch_size, *n_obs] -> [batch_size, channels, height, width]
            x = self.feature_cnn(agent_obs) # [batch_size, *n_obs] -> [batch_size, hx_size]
            if self.recurrent:
                x = self.gru(x, hidden[:, i, :])   # [batch_size, hx_size]
                next_hidden[:, i, :] = x  # [batch_size, hx_size]
            q_values[:, i, :] = self.q_val(x)   # [batch_size, n_actions]
        # q_values: [num_agents, (batch_size, 1, n_actions)]
        return q_values, next_hidden

    def sample_action(self, obs: torch.Tensor, hidden: torch.Tensor, epsilon=1e3) -> (torch.Tensor, torch.Tensor):
        """Choose action with epsilon-greedy policy, for each agent in the batch
        :param obs: a batch of observations, [batch_size, num_agents, n_obs]
        :param hidden: a batch of hidden states, [batch_size, num_agents, hx_size]
        :param epsilon: exploration rate
        :return: actions: [batch_size, num_agents], hidden: [batch_size, num_agents, hx_size]
        """
        # TODO: need to have a done_mask param
        q_values, hidden = self.forward(obs, hidden)    # [batch_size, num_agents, n_actions], [batch_size, num_agents, hx_size]
        # epsilon-greedy action selection: choose random action with epsilon probability
        mask = (torch.rand((q_values.shape[0],)) <= epsilon)     # [batch_size]
        actions = torch.empty((q_values.shape[0], q_values.shape[1],))     # [batch_size, num_agents]
        actions[mask] = torch.randint(0, q_values.shape[2], actions[mask].shape).float()
        actions[~mask] = q_values[~mask].argmax(dim=2).float()  # choose action with max q value
        return actions, hidden   # [batch_size, num_agents], [batch_size, num_agents, hx_size]

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self.num_agents, self.hx_size))


def train(q: VdnQNet, q_target: VdnQNet, memory: ReplayBuffer, optimizer: optim.Optimizer, gamma: float, batch_size: int, update_iter=10, chunk_size=10, grad_clip_norm=5):
    q.train()
    q_target.eval()
    chunk_size = chunk_size if q.recurrent else 1
    losses = []
    for i in range(update_iter):
        # Get data from buffer
        states, actions, rewards, next_states, dones = memory.sample_chunk(batch_size, chunk_size)

        hidden = q.init_hidden(batch_size)
        target_hidden = q_target.init_hidden(batch_size)
        loss = 0
        for step_i in range(chunk_size):
            # TODO: Pick states and hiddens for non-terminated agents
            out: tuple[torch.Tensor, torch.Tensor] = q(states[:, step_i], hidden)  # [batch_size, num_agents, n_actions]
            q_out, hidden = out
            q_a = q_out.gather(2, actions[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)   # [batch_size, num_agents]: q values of actions taken
            sum_q = q_a.sum(dim=1, keepdims=True)   # [batch_size, 1]

            max_q_prime, target_hidden = q_target(next_states[:, step_i], target_hidden.detach())
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)  # [batch_size, num_agents]
            target_q = rewards[:, step_i, :].sum(dim=1, keepdims=True)  # [batch_size, 1]
            target_q += gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - dones[:, step_i])

            loss += F.smooth_l1_loss(sum_q, target_q.detach())

            # FIXME: may have error with dones
            done_mask = dones[:, step_i].squeeze(-1).bool()
            hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))
            target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()

    print('Loss:', losses[0], losses[-1])
    return losses


def run_episode(env: pettingzoo.ParallelEnv, q: VdnQNet, memory: Optional[ReplayBuffer] = None, random_rate: float=0, epsilon: float=0.1) -> float:
    """Run an episode in the environment
    :return: total score of the episode
    """
    observations: dict[str, np.ndarray] = env.reset()
    team_manager = TeamManager(env.agents)
    teams = team_manager.get_teams()
    my_team = team_manager.get_my_team()
    hidden = q.init_hidden()
    score = 0.0

    while not team_manager.has_terminated_teams():
        my_team_observations = team_manager.get_info_of_team(my_team, observations)
        # Get actions for each agent based on the team
        agent_actions: dict[str, Optional[int]] = {}  # {agent: action}
        for team in teams:
            if team == my_team:
                # TODO: Fill rows with zeros for terminated agents
                team_observations = torch.tensor(np.array([
                    my_team_observations[agent]
                    for agent in team_manager.get_team_agents(team)
                ])).unsqueeze(0)    # [batch_size=1, num_agents, n_obs]
                # TODO: team_hidden = team_manager.get_info_of_team(team, hidden)
                actions, team_hiddens = q.sample_action(team_observations, hidden, epsilon)
                team_actions = {
                    agent: action
                    for agent, action in zip(
                        my_team_observations.keys(), actions.squeeze(0).data.numpy().tolist()
                    )
                }

                ##############################
                #  Random agents Experiment  #
                ##############################
                if random_rate > 0:
                    for agent in team_manager.get_random_agents(random_rate):
                        team_actions[agent] = env.action_space(agent).sample()
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

        if memory is not None:
            memory.put((
                list(my_team_observations.values()),
                list(team_manager.get_info_of_team(my_team, agent_actions).values()),
                list(team_manager.get_info_of_team(my_team, agent_rewards, 0).values()),
                list(team_manager.get_info_of_team(my_team, observations).values()),
                [int(team_manager.has_terminated_teams())]  # TODO: Update done mask for each agent
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


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Value Decomposition Network (VDN)')
    parser.add_argument('--max-episodes', type=int, default=160, required=False)
    parser.add_argument('--batch', type=int, default=2048, required=False)
    parser.add_argument('--env', type=str, default='adversarial_pursuit', required=False)
    parser.add_argument('--load_model', type=str, default='vdn-2024-12-09-sota', required=False)
    parser.add_argument('--task', type=str, default='experiment', required=False, choices=['train', 'experiment'])

    # Process arguments
    args = parser.parse_args()
    env_cfg = envs_config[args.env]
    max_episodes = args.max_episodes
    batch_size = args.batch
    task = args.task
    save_name = f'vdn-{args.env}-{today}'
    loaded_model = 'vdn-2024-12-09-sota' # args.load_model

    # Hyperparameters
    hp = VdnHyperparameters(
        lr=0.001,
        gamma=0.99,
        batch_size=batch_size,
        buffer_limit=9000,
        log_interval=20,
        max_episodes=max_episodes,
        max_epsilon=0.1,
        min_epsilon=0.0,
        test_episodes=5,
        warm_up_steps=3000,
        update_iter=20,
        chunk_size=1,
        update_target_interval=20
    )

    # Create env
    env = env_cfg["module"].parallel_env(**env_cfg["args"])
    test_env = env_cfg["module"].parallel_env(**env_cfg["args"])
    env.reset(seed=seed)
    test_env.reset(seed=seed)
    team_manager = TeamManager(env.agents)

    # Create model
    q = VdnQNet(team_manager.get_my_agents(), env.observation_spaces, env.action_spaces)
    q_target = VdnQNet(team_manager.get_my_agents(), env.observation_spaces, env.action_spaces)

    # Do task
    if task == 'train':
        # Load model if exists
        if loaded_model is not None and is_model_found(loaded_model):
            q = load_model(q, loaded_model)
            test_score = evaluate_model(test_env, hp.test_episodes, q, run_episode)
            print("Pretrained Model loaded. Test score: ", test_score)

        # Train and test
        train_scores, test_scores = run_model_train_test(
            env, test_env, VdnQNet, q, q_target,
            save_name, team_manager, hp, train, run_episode
        )

        # Save data
        save_data(np.array(train_scores), f'{save_name}-train_scores')
        save_data(np.array(test_scores), f'{save_name}-test_scores')
    elif task == 'experiment':
        if loaded_model is None or not is_model_found(loaded_model):
            raise ValueError("Please provide a model to load for experiment.")

        q = load_model(q, loaded_model)

        # Run experiment
        rate_avg_scores, rate_scores = run_experiment(
            env, q, hp.test_episodes * 4, run_episode, num_tests=10
        )

        # Save data
        save_dict(rate_avg_scores, f'{save_name}-rate_avg_scores')
        save_dict(rate_scores, f'{save_name}-rate_scores')


