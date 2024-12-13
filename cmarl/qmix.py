import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pettingzoo
import torch
import torch.optim as optim

from cmarl import vdn
from cmarl.runner import Hyperparameters, seed, evaluate_model, run_experiment, run_model_train_test
from cmarl.utils import compute_output_dim, today, TeamManager, is_model_found, load_model, save_data, save_dict
from cmarl.utils.buffer import ReplayBuffer
import gymnasium as gym

from cmarl.utils.env import envs_config

@dataclass
class QmixHyperparameters(Hyperparameters):
    recurrent: bool = False

import torch.nn as nn
import torch.nn.functional as F

class MixNet(nn.Module):
    def __init__(self, agents: list[str], observation_spaces: dict[str, gym.spaces.Space], hidden_dim=32, hx_size=64, recurrent=False):
        super(MixNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.hx_size = hx_size
        self.n_agents = len(agents)
        self.recurrent = recurrent
        self.n_obs = observation_spaces[agents[0]].shape  # observation space size of agents

        # Define CNN for preprocessing 3D observations
        stride1, stride2 = 1, 1
        padding1, padding2 = 1, 1
        kernel_size1, kernel_size2 = 3, 3
        pool_kernel_size, pool_stride = 2, 2
        out_channels = 32

        height = self.n_obs[0]  # n_obs is a tuple (height, width, channels)
        out_dim1 = compute_output_dim(height, kernel_size1, stride1, padding1) // pool_stride
        out_dim2 = compute_output_dim(out_dim1, kernel_size2, stride2, padding2) // pool_stride
        self.cnn_output_size = out_dim2 * out_dim2 * out_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(self.n_obs[2], 16, kernel_size1, stride1, padding1),
            nn.MaxPool2d(pool_kernel_size, pool_stride),
            nn.Conv2d(16, out_channels, kernel_size2, stride2, padding2),
            nn.MaxPool2d(pool_kernel_size, pool_stride),
            nn.Flatten()
        )

        # Define hyper network
        hyper_net_input_size = self.cnn_output_size * self.n_agents
        if self.recurrent:
            self.gru = nn.GRUCell(hyper_net_input_size, self.hx_size)
            hyper_net_input_size = self.hx_size
        self.hyper_net_weight_1 = nn.Linear(hyper_net_input_size, self.n_agents * hidden_dim)
        self.hyper_net_weight_2 = nn.Linear(hyper_net_input_size, hidden_dim)

        self.hyper_net_bias_1 = nn.Linear(hyper_net_input_size, hidden_dim)
        self.hyper_net_bias_2 = nn.Sequential(nn.Linear(hyper_net_input_size, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, 1))

    def forward(self, q_values: torch.Tensor, observations: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param q_values: (batch_size, n_agents)
        :param observations: (batch_size, n_agents, height, width, channels)
        :param hidden: (batch_size, hx_size)
        :return: (batch_size, 1), (batch_size, hx_size)
        """
        batch_size, n_agents, height, width, channels = observations.shape
        observations = observations.permute(0, 1, 4, 2, 3)  # (batch_size, n_agents, channels, height, width)
        observations = observations.reshape(batch_size * n_agents, channels, height, width) # (batch_size * n_agents, channels, height, width)
        cnn_output = self.cnn(observations) # (batch_size * n_agents, cnn_output_size)
        cnn_output = cnn_output.view(batch_size, n_agents, -1)  # (batch_size, n_agents, cnn_output_size)
        state = cnn_output.view(batch_size, n_agents * self.cnn_output_size)    # (batch_size, n_agents * cnn_output_size)

        x = state   # (batch_size, n_agents * cnn_output_size)
        if self.recurrent:
            hidden = self.gru(x, hidden)
            x = hidden  # (batch_size, hx_size)

        weight_1 = torch.abs(self.hyper_net_weight_1(x))    # (batch_size, n_agents * hidden_dim)
        weight_1 = weight_1.view(batch_size, self.hidden_dim, n_agents) # (batch_size, hidden_dim, n_agents)
        bias_1 = self.hyper_net_bias_1(x).unsqueeze(-1)     # (batch_size, hidden_dim, 1)
        weight_2 = torch.abs(self.hyper_net_weight_2(x))    # (batch_size, hidden_dim)
        bias_2 = self.hyper_net_bias_2(x)            # (batch_size, 1)

        x = torch.bmm(weight_1, q_values.unsqueeze(-1)) + bias_1    # (batch_size, hidden_dim, 1)
        x = torch.relu(x)   # (batch_size, hidden_dim, 1)
        x = (weight_2.unsqueeze(-1) * x).sum(dim=1) + bias_2    # (batch_size, 1)
        return x, hidden    # (batch_size, 1), (batch_size, hx_size)

    def init_hidden(self, batch_size=1):
        """
        :param batch_size:
        :return: (batch_size, hx_size)
        """
        return torch.zeros((batch_size, self.hx_size))


class QmixQNet(vdn.VdnQNet):
    model_name  = "QMIX"

    def __init__(self, agents: list[str], observation_spaces: dict[str, gym.spaces.Space], action_spaces: dict[str, gym.spaces.Space], recurrent=False):
        super(QmixQNet, self).__init__(agents, observation_spaces, action_spaces, recurrent)


def train(q: QmixQNet, q_target: QmixQNet, mix_net: MixNet, mix_net_target: MixNet,
          memory: ReplayBuffer, optimizer: optim.Optimizer,
          gamma: float, batch_size: int, update_iter=10, chunk_size=10, grad_clip_norm=5):
    q.train()
    q_target.eval()
    mix_net.train()
    mix_net_target.eval()
    chunk_size = chunk_size if q.recurrent else 1
    losses = []
    for i in range(update_iter):
        observations, actions, rewards, next_observations, dones = memory.sample_chunk(batch_size, chunk_size)

        hidden = q.init_hidden(batch_size)  # (batch_size, n_agents, hx_size)
        target_hidden = q_target.init_hidden(batch_size)    # (batch_size, n_agents, hx_size)
        mix_net_target_hidden = mix_net_target.init_hidden(batch_size)  # (batch_size, hx_size)
        mix_net_hidden = [torch.empty_like(mix_net_target_hidden) for _ in range(chunk_size + 1)]   # (chunk_size + 1) * (batch_size, hx_size)
        mix_net_hidden[0] = mix_net_target.init_hidden(batch_size)  # (batch_size, hx_size)

        loss = 0
        for step_i in range(chunk_size):
            q_out, hidden = q(observations[:, step_i], hidden)
            q_out: torch.Tensor  # (batch_size, n_agents, n_actions)
            hidden: torch.Tensor    # (batch_size, n_agents, hidden_dim)
            q_a = q_out.gather(2, actions[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)  # (batch_size, n_agents)
            pred_q, next_mix_net_hidden = mix_net(q_a, observations[:, step_i], mix_net_hidden[step_i]) # (batch_size, 1), (batch_size, hx_size)

            max_q_prime, target_hidden = q_target(next_observations[:, step_i], target_hidden.detach()) # (batch_size, n_agents, n_actions)
            max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1) # (batch_size, n_agents)
            q_prime_total, mix_net_target_hidden = mix_net_target(max_q_prime, next_observations[:, step_i],
                                                                  mix_net_target_hidden.detach())   # (batch_size, 1), (batch_size, hx_size)
            target_q = (rewards[:, step_i, :].sum(dim=1, keepdims=True)
                        + (gamma * q_prime_total
                           * (1 - dones[:, step_i, :].float()).min(dim=1, keepdim=True)[0]))    # (batch_size, 1)
            loss += F.smooth_l1_loss(pred_q, target_q.detach())

            done_mask = dones[:, step_i, :].any(dim=1).bool()   # (batch_size)
            hidden[done_mask] = q.init_hidden(len(hidden[done_mask]))   # (batch_size, n_agents, hx_size)
            target_hidden[done_mask] = q_target.init_hidden(len(target_hidden[done_mask]))  # (batch_size, n_agents, hx_size)
            mix_net_hidden[step_i + 1][~done_mask] = next_mix_net_hidden[~done_mask]    # (batch_size, hx_size)
            mix_net_hidden[step_i + 1][done_mask] = mix_net.init_hidden(len(mix_net_hidden[step_i][done_mask])) # (batch_size, hx_size)
            mix_net_target_hidden[done_mask] = mix_net_target.init_hidden(len(mix_net_target_hidden[done_mask]))    # (batch_size, hx_size)

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(mix_net.parameters(), grad_clip_norm, norm_type=2)
        optimizer.step()
    print('Loss:', losses[0], losses[-1])
    return losses



def run_episode(env: pettingzoo.ParallelEnv, q: QmixQNet, memory: Optional[ReplayBuffer] = None, random_rate: float=0, epsilon: float=0.1) -> float:
    return vdn.run_episode(env, q, memory, random_rate, epsilon)


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='QMIX')
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
    save_name = f'qmix-{args.env}-{today}'
    loaded_model = args.load_model

    # Hyperparameters
    hp = QmixHyperparameters(
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
        update_target_interval=20,
        recurrent=False,
    )

    # Create env
    env = env_cfg["module"].parallel_env(**env_cfg["args"])
    test_env = env_cfg["module"].parallel_env(**env_cfg["args"])
    env.reset(seed=seed)
    test_env.reset(seed=seed)
    team_manager = TeamManager(env.agents)

    # Create model
    q = QmixQNet(team_manager.get_my_agents(), env.observation_spaces, env.action_spaces)
    q_target = QmixQNet(team_manager.get_my_agents(), env.observation_spaces, env.action_spaces)

    mix_net = MixNet(team_manager.get_my_agents(), env.observation_spaces)
    mix_net_target = MixNet(team_manager.get_my_agents(), env.observation_spaces)

    # Do task
    if task == 'train':
        # Load model if exists
        if loaded_model is not None and is_model_found(loaded_model):
            q = load_model(q, loaded_model)
            if is_model_found(loaded_model+'-mix'):
                mix_net = load_model(mix_net, loaded_model+'-mix')
            test_score = evaluate_model(test_env, hp.test_episodes, q, run_episode)
            print("Pretrained Model loaded. Test score: ", test_score)

        # Train and test
        train_scores, test_scores, losses = run_model_train_test(
            env, test_env, QmixQNet, q, q_target,
            save_name, team_manager, hp, train, run_episode,
            mix_net=mix_net, mix_net_target=mix_net_target
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
