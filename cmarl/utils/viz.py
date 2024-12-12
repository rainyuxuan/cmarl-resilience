import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def cubic_smooth(x, y):
    f = interp1d(x, y, kind='cubic')
    x_new = np.linspace(min(x), max(x), 100)
    y_new = f(x_new)
    return x_new, y_new

def plot_rewards_over_episodes(rewards: np.ndarray, title='Rewards', save_path: str=None, episode_start=3, episode_end=160, step=1):
    episodes = np.arange(episode_start, episode_end + 1, step)
    # Smooth
    x_new, y_new = cubic_smooth(episodes, rewards)
    plt.plot(x_new, y_new)
    # Metadata
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_losses_over_episodes(losses: list[float], title='Losses', save_path: str=None, episode_start=3, episode_end=160, step=1):
    episodes = np.arange(episode_start, episode_end + 1, step)
    # Smooth
    x_new, y_new = cubic_smooth(episodes, losses)
    plt.plot(x_new, y_new)
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Losses')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_avg_scores_over_random_rates(scores: dict[float, float], title='Scores over Random Rates', save_path: str=None):
    rates = list(scores.keys())
    avg_scores = list(scores.values())
    # Smooth
    x_new, y_new = cubic_smooth(rates, avg_scores)
    plt.plot(x_new, y_new)
    # Metadata
    plt.title(title)
    plt.xlabel('Random Rates')
    plt.ylabel('Average Scores')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_scores_over_random_rates_with_variance(scores: dict[float, list[float]], title='Scores over Random Rates', save_path: str=None):
    rates = list(scores.keys())
    avg_scores = [np.mean(scores[rate]) for rate in rates]
    std_scores = [np.std(scores[rate]) for rate in rates]
    # Plot scores std with error bars
    plt.errorbar(rates, avg_scores, yerr=std_scores, fmt='o')
    # Plot smooth line for average scores
    x_new, y_new = cubic_smooth(rates, avg_scores)
    plt.plot(x_new, y_new)
    # Metadata
    plt.title(title)
    plt.xlabel('Random Rates')
    plt.ylabel('Average Scores')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_variance_over_random_rates(scores: dict[float, list[float]], title='Score Variance over Random Rates', save_path: str=None):
    rates = list(scores.keys())
    std_scores = [np.std(scores[rate]) for rate in rates]
    # Smooth
    x_new, y_new = cubic_smooth(rates, std_scores)
    plt.plot(x_new, y_new)
    # Metadata
    plt.title(title)
    plt.xlabel('Random Rates')
    plt.ylabel('Score Variance')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_stacked_rewards_over_episodes(rewards: dict[str, dict[str, any]], title='Rewards', save_path: str=None):
    """

    :param rewards: {name: {
        'rewards': np.ndarray,
        'color': str,
        'episode_start': int,
        'episode_end': int,
        'step': int
    }}
    :param title:
    :param save_path:
    :return:
    """
    for name, data in rewards.items():
        episodes = np.arange(data['episode_start'], data['episode_end'] + 1, data['step'])
        # Smooth
        x_new, y_new = cubic_smooth(episodes, data['rewards'])
        plt.plot(x_new, y_new, label=name, color=data['color'])
    # Metadata
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_stacked_scores_over_random_rates(scores: dict[str, dict[str, any]], title='Rewards', save_path: str=None):
    """

    :param scores: {name: {
        'scores': {rate: list[scores]},
        'color': str,
    }}
    :param title:
    :param save_path:
    :return:
    """
    for name, data in scores.items():
        rates = list(data['scores'].keys())
        avg_scores = [np.mean(data['scores'][rate]) for rate in rates]
        std_scores = [np.std(data['scores'][rate]) for rate in rates]
        # Plot scores std with error bars
        plt.errorbar(rates, avg_scores, yerr=std_scores, fmt='o', label=name, color=data['color'])
        # Plot smooth line for average scores
        x_new, y_new = cubic_smooth(rates, avg_scores)
        plt.plot(x_new, y_new, color=data['color'])
    # Metadata
    plt.title(title)
    plt.xlabel('Random Rates')
    plt.ylabel('Average Scores')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()