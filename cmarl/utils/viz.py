import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


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


def plot_avg_scores_over_random_rates(scores: dict[float, float], title='Scores over Random Rates', save_path: str=None):
    rates = list(scores.keys())
    avg_scores = list(scores.values())
    plt.plot(rates, avg_scores)
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
    x = np.linspace(min(rates), max(rates), 100)
    f = interp1d(rates, avg_scores, kind='cubic')
    plt.plot(x, f(x))

    # Metadata
    plt.title(title)
    plt.xlabel('Random Rates')
    plt.ylabel('Average Scores')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()