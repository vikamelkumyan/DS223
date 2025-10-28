############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create necessary directories
os.makedirs('Visualisations', exist_ok=True)
os.makedirs('Datasets', exist_ok=True)

class Bandit(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.
    """
    @abstractmethod
    def __init__(self, p):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    @abstractmethod
    def pull(self):
        pass
    
    @abstractmethod
    def update(self):
        pass
    
    @abstractmethod
    def experiment(self):
        pass
    
    @abstractmethod
    def report(self):
        pass

#--------------------------------------#
class Visualization():
    """
    Visualization class for comparing bandit algorithm performance.
    """
    def __init__(self, epsilon_greedy_bandits, thompson_sampling_bandits, eg_rewards, ts_rewards):
        """
        Initialize the visualization with bandit results.
        
        Args:
            epsilon_greedy_bandits: List of EpsilonGreedy bandit instances
            thompson_sampling_bandits: List of ThompsonSampling bandit instances
            eg_rewards: Array of rewards from EG experiment
            ts_rewards: Array of rewards from TS experiment
        """
        self.eg_bandits = epsilon_greedy_bandits
        self.ts_bandits = thompson_sampling_bandits
        self.eg_rewards = eg_rewards
        self.ts_rewards = ts_rewards
        self.eg_optimal = max([b.m for b in epsilon_greedy_bandits])
        self.ts_optimal = max([b.true_mean for b in thompson_sampling_bandits])
    
    def plot1(self):
        """
        Visualize cumulative rewards for each individual bandit arm in both algorithms.
        
        Creates a 2x2 grid of plots showing:
        - Top row: Epsilon-Greedy per-arm average rewards (linear and log scale)
        - Bottom row: Thompson Sampling per-arm average rewards (linear and log scale)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Epsilon-Greedy plots
        for i, bandit in enumerate(self.eg_bandits):
            if len(bandit.reward_history) > 0:
                rewards = np.array(bandit.reward_history)
                cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
                
                # Linear scale
                axes[0, 0].plot(cumulative_avg, label=f'Arm {i+1} (μ={bandit.m})', alpha=0.8)
                # Log scale
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
        
        # Thompson Sampling plots
        for i, bandit in enumerate(self.ts_bandits):
            if len(bandit.reward_history) > 0:
                rewards = np.array(bandit.reward_history)
                cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
                
                # Linear scale
                axes[1, 0].plot(cumulative_avg, label=f'Arm {i+1} (μ={bandit.true_mean})', alpha=0.8)
                # Log scale
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
        Compare Epsilon-Greedy and Thompson Sampling side-by-side.
        
        Creates a 1x2 plot showing:
        - Left: Cumulative rewards comparison between both algorithms
        - Right: Cumulative regrets comparison between both algorithms
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Calculate cumulative rewards
        eg_cumulative_rewards = np.cumsum(self.eg_rewards)
        ts_cumulative_rewards = np.cumsum(self.ts_rewards)
        
        # Plot cumulative rewards
        ax1.plot(eg_cumulative_rewards, label='Epsilon-Greedy', alpha=0.8, color='blue')
        ax1.plot(ts_cumulative_rewards, label='Thompson Sampling', alpha=0.8, color='green')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Cumulative Rewards Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Calculate cumulative regrets
        eg_regrets = np.array([self.eg_optimal - r for r in self.eg_rewards])
        ts_regrets = np.array([self.ts_optimal - r for r in self.ts_rewards])
        eg_cumulative_regrets = np.cumsum(eg_regrets)
        ts_cumulative_regrets = np.cumsum(ts_regrets)
        
        # Plot cumulative regrets
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

#--------------------------------------#
class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit algorithm implementation for Gaussian rewards.
    """
    
    def __init__(self, m):
        self.m = m
        self.m_estimate = 0
        self.N = 0
        self.reward_history = []  # Track rewards for this specific arm
    
    def __repr__(self):
        return f'An Arm with {self.m} Win Rate'
    
    def pull(self):
        reward = np.random.randn() + self.m
        self.reward_history.append(reward)  # Store reward for this arm
        return reward
    
    def update(self, x):
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N
    
    def experiment(self, num_trials=20000, bandits=None, initial_epsilon=1.0):
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
                logger.info(f"Trial {i}: epsilon={eps:.4f}, selected arm={j}, "
                          f"cumulative reward={np.sum(rewards[:i+1]):.2f}")
        
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

#--------------------------------------#
class ThompsonSampling(Bandit):
    """
    Thompson Sampling (Bayesian) bandit algorithm for Gaussian rewards.
    """
    
    def __init__(self, p):
        self.true_mean = p
        self.m = 0 
        self.lambda_ = 1 
        self.tau = 1 
        self.N = 0
        self.sum_x = 0
        self.reward_history = []  # Track rewards for this specific arm
    
    def __repr__(self):
        return f'An Arm with true_mean={self.true_mean}'
    
    def pull(self):
        reward = np.random.randn() / np.sqrt(self.tau) + self.true_mean
        self.reward_history.append(reward)  # Store reward for this arm
        return reward
    
    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.m
    
    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = (self.tau * self.sum_x) / self.lambda_
        self.N += 1
    
    def experiment(self, num_trials=20000, bandits=None):
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
                logger.info(f"Trial {i}: selected arm={j}, "
                          f"cumulative reward={np.sum(rewards[:i+1]):.2f}")
        
        logger.info("Thompson Sampling experiment completed")
        logger.info(f"Times optimal: {num_optimal}")
        
        return bandits, rewards, num_optimal
    
    def report(self, bandits, rewards, num_optimal):
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
            logger.info(f"  Arm {i}: {b} -> Posterior Mean (m): {b.m:.4f}, "
                       f"Posterior Precision (λ): {b.lambda_:.4f}, Pulls: {b.N}")
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
    logger.info(f"Winner: {'Thompson Sampling' if ts_total > eg_total else 'Epsilon-Greedy'} "
               f"(+{abs(ts_total - eg_total):.2f})")
    logger.info(f"\nEpsilon-Greedy Total Regret: {eg_regret:.2f}")
    logger.info(f"Thompson Sampling Total Regret: {ts_regret:.2f}")
    logger.info(f"Lower Regret: {'Thompson Sampling' if ts_regret < eg_regret else 'Epsilon-Greedy'} "
               f"(-{abs(ts_regret - eg_regret):.2f})")
    logger.info(f"{'='*60}")

if __name__=='__main__':
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