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


def experiment(env: pettingzoo.ParallelEnv, test_env: pettingzoo.ParallelEnv, model: nn.Module, hp: Hyperparameters,
                run_episode_fn: callable):
    pass




def evaluate_model(env: pettingzoo.ParallelEnv, num_episodes: int, model: nn.Module, run_episode_fn: callable):
    """
    :param env: Environment
    :param num_episodes: How many episodes to test
    :param model: Trained model
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

    score = 0
    test_score = 0
    train_scores = []
    test_scores = []

    # Train and test
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    for episode_i in tqdm(range(hp.max_episodes)):
        # Collect data
        epsilon = max(hp.min_epsilon,
                      hp.max_epsilon - (hp.max_epsilon - hp.min_epsilon) * (episode_i / (0.6 * hp.max_episodes)))
        model.eval()
        score += run_episode_fn(env, model, memory, epsilon)

        # Train
        if memory.size() > hp.warm_up_steps:
            print("Training phase:")
            model.train()
            train_fn(model, target_model, memory, optimizer, hp.gamma, hp.batch_size, hp.update_iter, hp.chunk_size)

        if episode_i % hp.update_target_interval == 0 and episode_i > 0:
            target_model.load_state_dict(model.state_dict())

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

    return train_scores, test_scores