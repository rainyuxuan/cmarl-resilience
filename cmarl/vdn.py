import argparse
import time
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
        padding1, padding2 = 1, 1
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
            sum_q = (q_a * (1-dones[:, step_i])).sum(dim=1, keepdims=True)   # [batch_size, 1]

            max_q_prime, target_hidden = q_target(next_states[:, step_i], target_hidden.detach())
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)  # [batch_size, num_agents]
            target_q = rewards[:, step_i, :].sum(dim=1, keepdims=True)  # [batch_size, 1]
            target_q += gamma * ((1 - dones[:, step_i]) * max_q_prime).sum(dim=1, keepdims=True)

            loss += F.smooth_l1_loss(sum_q, target_q.detach())

            # FIXME: may have error with dones
            # Create a mask for each agent separately
            done_mask = dones[:, step_i].bool()  # Shape: (batch_size, num_agents)

            # Iterate over each agent
            for agent_i in range(q.num_agents):
                # Get the termination mask for this specific agent
                agent_done_mask = done_mask[:, agent_i]  # Shape: (batch_size,)

                # Number of terminated agents for this specific agent
                num_terminated = agent_done_mask.sum().item()

                if num_terminated > 0:  # Only process if there are terminated agents
                    # Generate hidden states for terminated agents
                    # Ensure `batch_size=num_terminated` for init_hidden
                    new_hidden = q.init_hidden(batch_size=num_terminated)  # Shape: (num_terminated, num_agents, hx_size)
                    new_target_hidden = q_target.init_hidden(batch_size=num_terminated)  # Same shape

                    # Extract hidden states for this specific agent
                    new_hidden_agent = new_hidden[:, agent_i, :]  # Shape: (num_terminated, hx_size)
                    new_target_hidden_agent = new_target_hidden[:, agent_i, :]  # Shape: (num_terminated, hx_size)

                    # Assign to hidden states only for the terminated agents of this specific agent
                    hidden[agent_done_mask, agent_i, :] = new_hidden_agent  # (num_terminated, hx_size)
                    target_hidden[agent_done_mask, agent_i, :] = new_target_hidden_agent  # (num_terminated, hx_size)

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
        # Fill rows with zeros for terminated agents
        for agent in team_manager.get_my_agents():
            if agent not in my_team_observations or my_team_observations[agent] is None:
                my_team_observations[agent] = np.zeros(q.n_obs, dtype=np.float32)
                team_manager.terminate_agent(agent)

        # Get actions for each agent based on the team
        agent_actions: dict[str, Optional[int]] = {}  # {agent: action}
        for team in teams:
            if team == my_team:
                team_observations = torch.tensor(np.array(list(my_team_observations.values()))).unsqueeze(0)    # [batch_size=1, num_agents, n_obs]
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

        # Special rewards for tiger_deer_v4
        if env.metadata['name'] == 'tiger_deer_v4':
            score -= sum(team_manager.get_info_of_team('deer', agent_rewards, 0).values())

        # Fill rows with zeros for terminated agents
        next_observations = [
            observations[agent]
            if agent in observations and observations[agent] is not None
            else np.zeros(q.n_obs, dtype=np.float32)
            for agent in team_manager.get_my_agents()
        ]
        my_team_actions = [
            agent_actions[agent]
            if agent in agent_actions and agent_actions[agent] is not None
            else 0
            for agent in team_manager.get_my_agents()
        ]

        if memory is not None:
            memory.put((
                list(my_team_observations.values()),
                my_team_actions,
                list(team_manager.get_info_of_team(my_team, agent_rewards, 0).values()),
                next_observations,
                list(team_manager.get_info_of_team(
                    my_team,
                    TeamManager.merge_terminates_truncates(agent_terminations, agent_truncations)).values())
            ))

        # Check for termination
        for agent, done in agent_terminations.items():
            if done:
                team_manager.terminate_agent(agent)
        for agent, done in agent_truncations.items():
            if done:
                team_manager.terminate_agent(agent)

        # Sleep for .5 seconds for visualization
        # time.sleep(0.05)
    print('Score:', score, 'num_random_agents:', len(team_manager.get_random_agents(random_rate)))
    return score


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Value Decomposition Network (VDN)')
    parser.add_argument('--max-episodes', type=int, default=160, required=False)
    parser.add_argument('--batch', type=int, default=2048, required=False)
    parser.add_argument('--env', type=str, default='adversarial_pursuit', required=False)
    parser.add_argument('--load_model', type=str, default=None, required=False)
    parser.add_argument('--task', type=str, default='train', required=False, choices=['train', 'experiment'])
    parser.add_argument('--name_suffix', type=str, default='', required=False)

    # Process arguments
    args = parser.parse_args()
    env_cfg = envs_config[args.env]
    max_episodes = args.max_episodes
    batch_size = args.batch
    task = args.task
    save_name = f'vdn-{args.env}-{today}{"-" + args.name_suffix if args.name_suffix else ""}'
    loaded_model = args.load_model

    # Hyperparameters
    hp = VdnHyperparameters(
        lr=0.001,
        gamma=0.99,
        batch_size=batch_size,
        buffer_limit=9000,
        log_interval=10,
        max_episodes=max_episodes,
        max_epsilon=0.9,
        min_epsilon=0.1,
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
        train_scores, test_scores, losses = run_model_train_test(
            env, test_env, VdnQNet, q, q_target,
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
        num_tests = len(team_manager.get_my_agents()) + 1
        rate_avg_scores, rate_scores = run_experiment(
            env, q, hp.test_episodes * 2, run_episode, num_tests=num_tests
        )

        # Save data
        save_dict(rate_avg_scores, f'{save_name}-rate_avg_scores')
        save_dict(rate_scores, f'{save_name}-rate_scores')


