import matplotlib.pyplot as plt
import numpy as np


def plot_rewards_over_episodes(rewards: np.ndarray, title='Rewards', save_path: str=None):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_losses_over_episodes(losses: list[float], title='Losses', save_path: str=None):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Losses')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

