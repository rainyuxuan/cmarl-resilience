import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pettingzoo
from torch import nn, optim
from tqdm import tqdm

from cmarl.utils import reseed, TeamManager, is_model_found, load_model, save_model, save_data
from cmarl.utils.buffer import ReplayBuffer

seed = 42


@dataclass
class Hyperparameters:
    lr: float = 0.001
    gamma: float = 0.99
    batch_size: int = 2048
    update_iter: int = 20
    buffer_limit: int = 9000
    log_interval: int = 20
    update_target_interval: int = 20
    max_episodes: int = 160
    max_epsilon: float = 0.9
    min_epsilon: float = 0.1
    test_episodes: int = 10
    warm_up_steps: int = 3000
    chunk_size: int = 1


def run_experiment(
        env: pettingzoo.ParallelEnv, model: nn.Module,
        eval_iter: int,
        run_test_episode_fn: callable,
        random_agent_min_rate: float = 0.0,
        random_agent_max_rate: float = 1.0,
        num_tests: int = 16,
) -> tuple[dict, dict]:
    reseed(seed)
    random_rates = np.linspace(random_agent_min_rate, random_agent_max_rate, num_tests)
    rate_avg_scores = {}
    rate_scores = {}
    for rate in tqdm(random_rates):
        print(f"Testing with random rate: {rate}")
        env.reset(seed=seed)

        rate_scores[rate] = []
        for iter in range(eval_iter):
            env.reset()
            test_score = run_test_episode_fn(env, model, random_rate=rate, epsilon=0)
            rate_scores[rate].append(test_score)
        rate_avg_scores[rate] = np.mean(rate_scores[rate])
        print(f"Average score for random_rate={rate}: {rate_avg_scores[rate]}")
    return rate_avg_scores, rate_scores


def evaluate_model(env: pettingzoo.ParallelEnv, num_episodes: int, model: nn.Module, run_episode_fn: callable):
    """

    :param env: Environment
    :param num_episodes: How many episodes to test
    :param model: Trained model
    :param run_episode_fn: function to run an episode
    :return: average score over num_episodes
    """
    model.eval()
    score = 0
    for episode_i in range(num_episodes):
        score += run_episode_fn(env, model, epsilon=0)
    return score / num_episodes


def run_model_train_test(
        env: pettingzoo.ParallelEnv,
        test_env: pettingzoo.ParallelEnv,
        Model: callable,
        model: nn.Module,
        target_model: nn.Module,
        save_name: str,
        team_manager: TeamManager,
        hp: Hyperparameters,
        train_fn: callable,
        run_episode_fn: callable,
        mix_net: nn.Module = None,
        mix_net_target: nn.Module = None,
) -> (list[float], list[float]):
    """
    Run training and testing loop of a model

    :param env: Training environment
    :param test_env: Testing environment
    :param Model: Model class
    :param model: training model
    :param target_model: target model
    :param save_name: name to save the model
    :param team_manager: TeamManager
    :param hp: Hyperparameters
    :param train_fn: training function
    :param run_episode_fn: function to run an episode
    :return: train_scores, test_scores
    """
    reseed(seed)
    # create env.
    memory = ReplayBuffer(hp.buffer_limit)

    # Setup env
    test_env.reset(seed=seed)
    env.reset(seed=seed)
    my_team = team_manager.get_my_team()
    model_name = Model.model_name
    print("Training model: ", model_name)
    print("My team: ", my_team)

    # Load target model
    target_model.load_state_dict(model.state_dict())
    if mix_net is not None:
        mix_net_target.load_state_dict(mix_net.state_dict())

    score = 0
    test_score = 0
    train_scores = []
    test_scores = []
    losses: list[list[float]] = []

    # Train and test
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    if mix_net is not None:
        optimizer = optim.Adam([{'params': model.parameters()}, {'params': mix_net.parameters()}], lr=hp.lr)

    for episode_i in tqdm(range(hp.max_episodes)):
        # Collect data
        epsilon = max(hp.min_epsilon,
                      hp.max_epsilon - (hp.max_epsilon - hp.min_epsilon) * (episode_i / (0.6 * hp.max_episodes)))
        model.eval()
        score += run_episode_fn(env, model, memory, epsilon=epsilon)

        # Train
        if memory.size() > hp.warm_up_steps:
            print("Training phase:")
            model.train()

            if mix_net is not None:
                mix_net.train()
                episode_losses = train_fn(
                    model, target_model, mix_net, mix_net_target, memory, optimizer,
                    hp.gamma, hp.batch_size, hp.update_iter, hp.chunk_size
                )
            else:
                episode_losses = train_fn(
                    model, target_model, memory, optimizer,
                    hp.gamma, hp.batch_size, hp.update_iter, hp.chunk_size
                )
            losses.append(episode_losses)

        if episode_i % hp.update_target_interval == 0 and episode_i > 0:
            target_model.load_state_dict(model.state_dict())
            if mix_net is not None:
                mix_net_target.load_state_dict(mix_net.state_dict())

        # Test
        if ((episode_i == hp.max_episodes - 1)  # last episode
                or (episode_i % hp.log_interval == 0 and episode_i > 0)):
            print("Test phase:")
            prev_test_score = test_score
            model.eval()
            test_score = evaluate_model(test_env, hp.test_episodes, model, run_episode_fn)
            test_scores.append(test_score)
            if test_score > prev_test_score:
                save_model(model, save_name)
                if mix_net is not None:
                    save_model(mix_net, save_name + '-mix')
                print(f"Model saved as {save_name} at episode: {episode_i}")
            print("Test score: ", test_score)

            train_score = score / hp.log_interval
            train_scores.append(train_score)
            print("#{:<10}/{} episodes , avg train score : {:.1f}, test score: {:.1f} n_buffer : {}, epsilon : {:.1f}"
                  .format(episode_i, hp.max_episodes, train_score, test_score, memory.size(), epsilon))
            score = 0
            print('#' * 90)

    env.close()
    test_env.close()

    return train_scores, test_scores, losses
