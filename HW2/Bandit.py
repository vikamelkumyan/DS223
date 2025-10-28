"""
Multi-armed bandit experiment framework providing Epsilon-Greedy and Thompson Sampling
implementations for Gaussian rewards, along with utilities to visualize and compare
their performance and to persist experiment results.

The script can be executed directly. It will:
1) Construct bandit arms with predefined true means.
2) Run Epsilon-Greedy and Thompson Sampling experiments.
3) Produce visualizations comparing per-arm performance and aggregate rewards/regret.
4) Save per-trial results and combined datasets to disk.

Artifacts:
- Visualisations/bandit_performance.png
- Visualisations/algorithms_comparison.png
- Datasets/epsilon_greedy_results.csv
- Datasets/thompson_sampling_results.csv
- Datasets/combined_results.csv
"""

from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('Visualisations', exist_ok=True)
os.makedirs('Datasets', exist_ok=True)


class Bandit(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.

    Subclasses must implement:
    - __init__: Initialize arm-specific parameters.
    - __repr__: Human-readable representation.
    - pull:     Sample a reward from the arm's reward distribution.
    - update:   Update arm/posterior estimates given an observed reward.
    - experiment: Run a full experiment over many trials.
    - report:   Log and/or persist experiment results.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize a bandit arm.

        Args:
            p: Parameter that defines the arm's underlying reward distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        """Return a concise, human-readable description of the arm."""
        raise NotImplementedError

    @abstractmethod
    def pull(self):
        """
        Draw a reward sample from the arm's true distribution.

        Returns:
            float: Observed reward.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, x):
        """
        Update internal estimates using an observed reward.

        Args:
            x (float): Observed reward.
        """
        raise NotImplementedError

    @abstractmethod
    def experiment(self, *args, **kwargs):
        """
        Run a complete experiment interacting with a set of bandit arms.

        Returns:
            Any: Implementation-specific results (e.g., arms, rewards, counts).
        """
        raise NotImplementedError

    @abstractmethod
    def report(self, *args, **kwargs):
        """
        Log metrics and optionally persist results from a completed experiment.
        """
        raise NotImplementedError


class Visualization:
    """
    Visualization utilities for comparing bandit algorithm performance.

    This class plots:
    - Per-arm average reward trajectories (linear and log scale) for both algorithms.
    - Side-by-side cumulative reward and cumulative regret comparisons.
    """

    def __init__(self, epsilon_greedy_bandits, thompson_sampling_bandits, eg_rewards, ts_rewards):
        """
        Initialize the visualization with results from both algorithms.

        Args:
            epsilon_greedy_bandits (list[EpsilonGreedy]): Trained Epsilon-Greedy arm instances.
            thompson_sampling_bandits (list[ThompsonSampling]): Trained Thompson Sampling arm instances.
            eg_rewards (np.ndarray): Per-trial rewards from the Epsilon-Greedy experiment.
            ts_rewards (np.ndarray): Per-trial rewards from the Thompson Sampling experiment.
        """
        self.eg_bandits = epsilon_greedy_bandits
        self.ts_bandits = thompson_sampling_bandits
        self.eg_rewards = eg_rewards
        self.ts_rewards = ts_rewards
        self.eg_optimal = max([b.m for b in epsilon_greedy_bandits])
        self.ts_optimal = max([b.true_mean for b in thompson_sampling_bandits])

    def plot1(self):
        """
        Plot per-arm average reward trajectories for both algorithms.

        Generates a 2x2 grid:
            Row 1: Epsilon-Greedy per-arm average rewards (linear and log x-scale)
            Row 2: Thompson Sampling per-arm average rewards (linear and log x-scale)

        Saves:
            Visualisations/bandit_performance.png
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        for i, bandit in enumerate(self.eg_bandits):
            if len(bandit.reward_history) > 0:
                rewards = np.array(bandit.reward_history)
                cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
                axes[0, 0].plot(cumulative_avg, label=f'Arm {i+1} (μ={bandit.m})', alpha=0.8)
                axes[0, 1].plot(cumulative_avg, label=f'Arm {i+1} (μ={bandit.m})', alpha=0.8)

        axes[0, 0].set_xlabel('Pull Number')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Epsilon-Greedy - Average Reward per Arm (Linear Scale)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel('Pull Number')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].set_title('Epsilon-Greedy - Average Reward per Arm (Log Scale)')
        axes[0, 1].set_xscale('log')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        for i, bandit in enumerate(self.ts_bandits):
            if len(bandit.reward_history) > 0:
                rewards = np.array(bandit.reward_history)
                cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
                axes[1, 0].plot(cumulative_avg, label=f'Arm {i+1} (μ={bandit.true_mean})', alpha=0.8)
                axes[1, 1].plot(cumulative_avg, label=f'Arm {i+1} (μ={bandit.true_mean})', alpha=0.8)

        axes[1, 0].set_xlabel('Pull Number')
        axes[1, 0].set_ylabel('Average Reward')
        axes[1, 0].set_title('Thompson Sampling - Average Reward per Arm (Linear Scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel('Pull Number')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].set_title('Thompson Sampling - Average Reward per Arm (Log Scale)')
        axes[1, 1].set_xscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('Visualisations/bandit_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Performance plot saved to 'Visualisations/bandit_performance.png'")

    def plot2(self):
        """
        Plot side-by-side cumulative rewards and cumulative regrets for both algorithms.

        Left subplot:
            Cumulative reward trajectories for Epsilon-Greedy and Thompson Sampling.
        Right subplot:
            Cumulative regret trajectories computed against the best true mean of each setup.

        Saves:
            Visualisations/algorithms_comparison.png
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        eg_cumulative_rewards = np.cumsum(self.eg_rewards)
        ts_cumulative_rewards = np.cumsum(self.ts_rewards)

        ax1.plot(eg_cumulative_rewards, label='Epsilon-Greedy', alpha=0.8, color='blue')
        ax1.plot(ts_cumulative_rewards, label='Thompson Sampling', alpha=0.8, color='green')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Cumulative Rewards Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        eg_regrets = np.array([self.eg_optimal - r for r in self.eg_rewards])
        ts_regrets = np.array([self.ts_optimal - r for r in self.ts_rewards])
        eg_cumulative_regrets = np.cumsum(eg_regrets)
        ts_cumulative_regrets = np.cumsum(ts_regrets)

        ax2.plot(eg_cumulative_regrets, label='Epsilon-Greedy', alpha=0.8, color='blue')
        ax2.plot(ts_cumulative_regrets, label='Thompson Sampling', alpha=0.8, color='green')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Cumulative Regret')
        ax2.set_title('Cumulative Regrets Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('Visualisations/algorithms_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Comparison plot saved to 'Visualisations/algorithms_comparison.png'")


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm for Gaussian rewards.

    Each arm maintains:
        - m:          True mean (unknown to the algorithm; used to generate rewards).
        - m_estimate: Estimated mean reward from samples.
        - N:          Number of times the arm has been pulled.
        - reward_history: List of observed rewards for visualization.

    Exploration probability decays as epsilon_t = epsilon_0 / t.
    """

    def __init__(self, m):
        """
        Initialize an Epsilon-Greedy arm.

        Args:
            m (float): True mean of the arm's Gaussian reward distribution.
        """
        self.m = m
        self.m_estimate = 0
        self.N = 0
        self.reward_history = []

    def __repr__(self):
        """Return a readable description of the arm with its true mean."""
        return f'An Arm with {self.m} Win Rate'

    def pull(self):
        """
        Sample a reward from N(m, 1).

        Returns:
            float: Observed reward.
        """
        reward = np.random.randn() + self.m
        self.reward_history.append(reward)
        return reward

    def update(self, x):
        """
        Update the running mean estimate given an observed reward.

        Args:
            x (float): Observed reward.
        """
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N

    def experiment(self, num_trials=20000, bandits=None, initial_epsilon=1.0):
        """
        Run an Epsilon-Greedy experiment over multiple arms.

        Args:
            num_trials (int): Number of interaction steps.
            bandits (list[EpsilonGreedy]): The arms to interact with.
            initial_epsilon (float): Initial exploration weight ε0 (decays as ε0 / t).

        Returns:
            tuple:
                - list[EpsilonGreedy]: The bandit arms after the experiment.
                - np.ndarray: Per-trial rewards.
                - int: Number of exploration steps.
                - int: Number of exploitation steps.
                - int: Number of times the optimal arm (by true mean) was selected.
        """
        if bandits is None:
            logger.error("Must provide list of bandits for experiment")
            return

        logger.info(f"Starting Epsilon-Greedy experiment with {num_trials} trials")

        rewards = np.zeros(num_trials)
        num_times_explored = 0
        num_times_exploited = 0
        num_optimal = 0
        bandit_selections = []

        optimal_j = np.argmax([b.m for b in bandits])
        optimal_reward = max([b.m for b in bandits])
        logger.info(f"Optimal bandit: {optimal_j} with reward {optimal_reward}")

        for i in range(num_trials):
            eps = initial_epsilon / (i + 1)

            if np.random.random() < eps:
                num_times_explored += 1
                j = np.random.randint(len(bandits))
            else:
                num_times_exploited += 1
                j = np.argmax([b.m_estimate for b in bandits])

            if j == optimal_j:
                num_optimal += 1

            x = bandits[j].pull()
            rewards[i] = x
            bandit_selections.append(j)
            bandits[j].update(x)

            if i % 5000 == 0 and i > 0:
                logger.info(
                    f"Trial {i}: epsilon={eps:.4f}, selected arm={j}, "
                    f"cumulative reward={np.sum(rewards[:i+1]):.2f}"
                )

        logger.info("Epsilon-Greedy experiment completed")
        logger.info(f"Times explored: {num_times_explored}")
        logger.info(f"Times exploited: {num_times_exploited}")
        logger.info(f"Times optimal: {num_optimal}")

        self.rewards = rewards
        self.bandit_selections = bandit_selections
        self.num_times_explored = num_times_explored
        self.num_times_exploited = num_times_exploited
        self.num_optimal = num_optimal
        self.optimal_reward = optimal_reward
        self.bandits = bandits

        return bandits, rewards, num_times_explored, num_times_exploited, num_optimal

    def report(self):
        """
        Log a detailed report of the Epsilon-Greedy experiment and save results to CSV.

        Saves:
            Datasets/epsilon_greedy_results.csv

        Logs include:
            - Total/average reward and regret.
            - Exploration vs exploitation counts.
            - Per-arm estimates and pull counts.
        """
        if not hasattr(self, 'rewards'):
            logger.error("No experiment data to report. Run experiment() first.")
            return

        total_reward = np.sum(self.rewards)
        avg_reward = np.mean(self.rewards)

        regret_per_trial = np.array([self.optimal_reward - r for r in self.rewards])
        total_regret = np.sum(regret_per_trial)
        avg_regret = np.mean(regret_per_trial)

        logger.info(f"{'='*60}")
        logger.info(f"EPSILON-GREEDY REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trials: {len(self.rewards)}")
        logger.info(f"Total Cumulative Reward: {total_reward:.2f}")
        logger.info(f"Average Reward per Trial: {avg_reward:.4f}")
        logger.info(f"Total Cumulative Regret: {total_regret:.2f}")
        logger.info(f"Average Regret per Trial: {avg_regret:.4f}")
        logger.info(f"\nExploration/Exploitation Statistics:")
        logger.info(f" # of Explored: {self.num_times_explored}")
        logger.info(f" # of Exploited: {self.num_times_exploited}")
        logger.info(f" # of times selected the optimal bandit: {self.num_optimal}")
        logger.info(f"\nArm Statistics:")
        for i, b in enumerate(self.bandits):
            logger.info(f"  Arm {i}: {b} -> Estimated: {b.m_estimate:.4f}, Pulls: {b.N}")
        logger.info(f"{'='*60}")

        df = pd.DataFrame({
            'Bandit': self.bandit_selections,
            'Reward': self.rewards,
            'Algorithm': 'EpsilonGreedy'
        })
        df.to_csv('Datasets/epsilon_greedy_results.csv', index=False)
        logger.info("Results saved to 'Datasets/epsilon_greedy_results.csv'")


class ThompsonSampling(Bandit):
    """
    Thompson Sampling (Bayesian) algorithm for Gaussian rewards with known unit variance.

    Posterior is maintained in a conjugate Normal-Normal form with:
        - m:       Posterior mean.
        - lambda_: Posterior precision (1/variance) of the mean.
        - tau:     Known observation precision (fixed to 1 here).
        - N:       Number of observations for this arm.
        - sum_x:   Sum of observed rewards (sufficient statistic).
        - reward_history: List of observed rewards for visualization.
    """

    def __init__(self, p):
        """
        Initialize a Thompson Sampling arm.

        Args:
            p (float): True mean of the arm's Gaussian reward distribution.
        """
        self.true_mean = p
        self.m = 0
        self.lambda_ = 1
        self.tau = 1
        self.N = 0
        self.sum_x = 0
        self.reward_history = []

    def __repr__(self):
        """Return a readable description including the arm's true mean."""
        return f'An Arm with true_mean={self.true_mean}'

    def pull(self):
        """
        Sample a reward from N(true_mean, 1).

        Returns:
            float: Observed reward.
        """
        reward = np.random.randn() / np.sqrt(self.tau) + self.true_mean
        self.reward_history.append(reward)
        return reward

    def sample(self):
        """
        Draw a sample from the current posterior over the mean.

        Returns:
            float: Posterior sample for the arm's mean.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        """
        Update posterior parameters given an observed reward.

        Args:
            x (float): Observed reward.
        """
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = (self.tau * self.sum_x) / self.lambda_
        self.N += 1

    def experiment(self, num_trials=20000, bandits=None):
        """
        Run a Thompson Sampling experiment over multiple arms.

        Args:
            num_trials (int): Number of interaction steps.
            bandits (list[ThompsonSampling]): The arms to interact with.

        Returns:
            tuple:
                - list[ThompsonSampling]: The bandit arms after the experiment.
                - np.ndarray: Per-trial rewards.
                - int: Number of times the optimal arm (by true mean) was selected.
        """
        if bandits is None:
            logger.error("Must provide list of bandits for experiment")
            return

        logger.info(f"Starting Thompson Sampling experiment with {num_trials} trials")

        rewards = np.zeros(num_trials)
        num_optimal = 0
        bandit_selections = []

        optimal_j = np.argmax([b.true_mean for b in bandits])
        optimal_reward = max([b.true_mean for b in bandits])
        logger.info(f"Optimal bandit: {optimal_j} with mean reward {optimal_reward}")

        for i in range(num_trials):
            j = np.argmax([b.sample() for b in bandits])

            if j == optimal_j:
                num_optimal += 1

            x = bandits[j].pull()
            bandits[j].update(x)

            rewards[i] = x
            bandit_selections.append(j)

            if i % 5000 == 0 and i > 0:
                logger.info(
                    f"Trial {i}: selected arm={j}, "
                    f"cumulative reward={np.sum(rewards[:i+1]):.2f}"
                )

        logger.info("Thompson Sampling experiment completed")
        logger.info(f"Times optimal: {num_optimal}")

        return bandits, rewards, num_optimal

    def report(self, bandits, rewards, num_optimal):
        """
        Log a detailed report of the Thompson Sampling experiment and save results to CSV.

        Args:
            bandits (list[ThompsonSampling]): The bandit arms used.
            rewards (np.ndarray): Per-trial rewards.
            num_optimal (int): Number of times the optimal arm was selected.

        Saves:
            Datasets/thompson_sampling_results.csv

        Logs include:
            - Total/average reward and regret.
            - Fraction of times the optimal arm was selected.
            - Posterior parameters and pull counts per arm.
        """
        optimal_reward = max([b.true_mean for b in bandits])

        total_reward = np.sum(rewards)
        avg_reward = np.mean(rewards)

        regret_per_trial = np.array([optimal_reward - r for r in rewards])
        total_regret = np.sum(regret_per_trial)
        avg_regret = np.mean(regret_per_trial)

        logger.info(f"{'='*60}")
        logger.info(f"THOMPSON SAMPLING REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trials: {len(rewards)}")
        logger.info(f"Total Cumulative Reward: {total_reward:.2f}")
        logger.info(f"Average Reward per Trial: {avg_reward:.4f}")
        logger.info(f"Total Cumulative Regret: {total_regret:.2f}")
        logger.info(f"Average Regret per Trial: {avg_regret:.4f}")
        logger.info(f"\nSelection Statistics:")
        logger.info(f"  Times Optimal Selected: {num_optimal} ({100*num_optimal/len(rewards):.2f}%)")
        logger.info(f"\nArm Statistics:")
        for i, b in enumerate(bandits):
            logger.info(
                f"  Arm {i}: {b} -> Posterior Mean (m): {b.m:.4f}, "
                f"Posterior Precision (λ): {b.lambda_:.4f}, Pulls: {b.N}"
            )
        logger.info(f"{'='*60}")

        df = pd.DataFrame({
            'Trial': range(len(rewards)),
            'Reward': rewards,
            'Algorithm': 'ThompsonSampling'
        })
        df.to_csv('Datasets/thompson_sampling_results.csv', index=False)
        logger.info("Results saved to 'Datasets/thompson_sampling_results.csv'")

        self.rewards = rewards
        self.optimal_reward = optimal_reward


def comparison(eg_bandits, ts_bandits, eg_rewards, ts_rewards):
    """
    Compare performance of Epsilon-Greedy and Thompson Sampling algorithms.

    This function:
        - Concatenates and persists per-trial results from both algorithms.
        - Produces per-arm and aggregate comparison plots.
        - Logs aggregate totals and regrets.

    Args:
        eg_bandits (list[EpsilonGreedy]): Trained EG arms after experiment.
        ts_bandits (list[ThompsonSampling]): Trained TS arms after experiment.
        eg_rewards (np.ndarray): Per-trial rewards from EG.
        ts_rewards (np.ndarray): Per-trial rewards from TS.
    """
    logger.info("Generating comparison visualizations...")

    df_eg = pd.read_csv('Datasets/epsilon_greedy_results.csv')
    df_ts = pd.read_csv('Datasets/thompson_sampling_results.csv')
    df_combined = pd.concat([df_eg, df_ts], ignore_index=True)
    df_combined.to_csv('Datasets/combined_results.csv', index=False)
    logger.info("Combined results saved to 'Datasets/combined_results.csv'")

    viz = Visualization(eg_bandits, ts_bandits, eg_rewards, ts_rewards)
    viz.plot1()
    viz.plot2()

    eg_optimal = max([b.m for b in eg_bandits])
    ts_optimal = max([b.true_mean for b in ts_bandits])

    eg_total = np.sum(eg_rewards)
    ts_total = np.sum(ts_rewards)
    eg_regret = np.sum([eg_optimal - r for r in eg_rewards])
    ts_regret = np.sum([ts_optimal - r for r in ts_rewards])

    logger.info(f"\n{'='*60}")
    logger.info(f"ALGORITHM COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"Epsilon-Greedy Total Reward: {eg_total:.2f}")
    logger.info(f"Thompson Sampling Total Reward: {ts_total:.2f}")
    logger.info(
        f"Winner: {'Thompson Sampling' if ts_total > eg_total else 'Epsilon-Greedy'} "
        f"(+{abs(ts_total - eg_total):.2f})"
    )
    logger.info(f"\nEpsilon-Greedy Total Regret: {eg_regret:.2f}")
    logger.info(f"Thompson Sampling Total Regret: {ts_regret:.2f}")
    logger.info(
        f"Lower Regret: {'Thompson Sampling' if ts_regret < eg_regret else 'Epsilon-Greedy'} "
        f"(-{abs(ts_regret - eg_regret):.2f})"
    )
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    np.random.seed(1)

    logger.info("\n" + "="*60)
    logger.info("MULTI-ARMED BANDIT EXPERIMENT")
    logger.info("="*60 + "\n")

    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000

    logger.info(f"True Bandit Rewards: {Bandit_Reward}")
    logger.info(f"Number of Trials: {NumberOfTrials}\n")

    logger.info("="*60)
    logger.info("Creating Epsilon-Greedy Bandits...")
    eg_bandits = [EpsilonGreedy(m) for m in Bandit_Reward]

    eg_bandits_result, eg_rewards, _, _, _ = eg_bandits[0].experiment(
        num_trials=NumberOfTrials, bandits=eg_bandits, initial_epsilon=1.0
    )
    eg_bandits[0].report()

    logger.info("\n" + "="*60)
    logger.info("Creating Thompson Sampling Bandits...")
    ts_bandits = [ThompsonSampling(p) for p in Bandit_Reward]

    ts_bandits_result, ts_rewards, ts_optimal = ts_bandits[0].experiment(
        num_trials=NumberOfTrials, bandits=ts_bandits
    )
    ts_bandits[0].report(ts_bandits_result, ts_rewards, ts_optimal)

    logger.info("\n" + "="*60)
    comparison(eg_bandits_result, ts_bandits_result, eg_rewards, ts_rewards)

    logger.info("\nExperiment completed successfully!")
