############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
    @abstractmethod
    def __init__(self, p):
        pass
    
    @abstractmethod
    def __repr__(self):
        pass
    
    @abstractmethod
    def pull(self):
        """ """
        pass
    
    @abstractmethod
    def update(self):
        """ """
        pass
    
    @abstractmethod
    def experiment(self):
        """ """
        pass
    
    @abstractmethod
    def report(self):
        """ """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#
class Visualization():
    """ """
    def __init__(self, epsilon_greedy, thompson_sampling):
        self.eg = epsilon_greedy
        self.ts = thompson_sampling
    
    def plot1(self):
        """Visualize the performance of each bandit: linear and log"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Epsilon-Greedy Linear
        axes[0, 0].plot(np.cumsum(self.eg.rewards))
        axes[0, 0].set_title('Epsilon-Greedy: Cumulative Reward (Linear)')
        axes[0, 0].set_xlabel('Trials')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].grid(True)
        
        # Epsilon-Greedy Log
        axes[0, 1].plot(np.cumsum(self.eg.rewards))
        axes[0, 1].set_title('Epsilon-Greedy: Cumulative Reward (Log Scale)')
        axes[0, 1].set_xlabel('Trials')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Thompson Sampling Linear
        axes[1, 0].plot(np.cumsum(self.ts.rewards))
        axes[1, 0].set_title('Thompson Sampling: Cumulative Reward (Linear)')
        axes[1, 0].set_xlabel('Trials')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].grid(True)
        
        # Thompson Sampling Log
        axes[1, 1].plot(np.cumsum(self.ts.rewards))
        axes[1, 1].set_title('Thompson Sampling: Cumulative Reward (Log Scale)')
        axes[1, 1].set_xlabel('Trials')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('Visualisations/bandit_performance.png', dpi=300, bbox_inches='tight')
        logger.info("Plot 1 saved as 'bandit_performance.png'")
        plt.show()
    
    def plot2(self):
        """Compare E-greedy and Thompson sampling cumulative rewards and regrets"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Cumulative Rewards Comparison
        axes[0].plot(np.cumsum(self.eg.rewards), label='Epsilon-Greedy', alpha=0.8)
        axes[0].plot(np.cumsum(self.ts.rewards), label='Thompson Sampling', alpha=0.8)
        axes[0].set_title('Cumulative Rewards Comparison')
        axes[0].set_xlabel('Trials')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].legend()
        axes[0].grid(True)
        
        # Cumulative Regrets Comparison
        eg_regret = np.array([self.eg.optimal_reward - r for r in self.eg.rewards])
        ts_regret = np.array([self.ts.optimal_reward - r for r in self.ts.rewards])
        
        axes[1].plot(np.cumsum(eg_regret), label='Epsilon-Greedy', alpha=0.8)
        axes[1].plot(np.cumsum(ts_regret), label='Thompson Sampling', alpha=0.8)
        axes[1].set_title('Cumulative Regrets Comparison')
        axes[1].set_xlabel('Trials')
        axes[1].set_ylabel('Cumulative Regret')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('Visualisations/algorithms_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("Plot 2 saved as 'algorithms_comparison.png'")
        plt.show()

#--------------------------------------#
class EpsilonGreedy(Bandit):
    """Individual Bandit Arm for Epsilon-Greedy algorithm"""
    
    def __init__(self, m):
        """
        Initialize a single bandit arm
        p: true win rate/probability for this arm
        """
        self.m = m
        self.m_estimate = 0
        self.N = 0
    
    def __repr__(self):
        return f'An Arm with {self.m} Win Rate'
    
    def pull(self):
        """Pull this arm and return reward"""
        return np.random.randn() + self.m
    
    def update(self, x):
        """Update the estimate for this arm based on observed reward x

        Args:
          x: 

        Returns:

        """
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N
    
    def experiment(self, num_trials=20000, bandits=None, initial_epsilon=1.0):
        """Run the epsilon-greedy experiment with multiple bandits
        This method should be called on one bandit instance but manages all bandits

        Args:
          num_trials:  (Default value = 20000)
          bandits:  (Default value = None)
          initial_epsilon:  (Default value = 1.0)

        Returns:

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
            # Decay epsilon by 1/t
            eps = initial_epsilon / (i + 1)
            
            if np.random.random() < eps:
                # Explore
                num_times_explored += 1
                j = np.random.randint(len(bandits))
            else:
                # Exploit
                num_times_exploited += 1
                j = np.argmax([b.m_estimate for b in bandits])
            
            if j == optimal_j:
                num_optimal += 1
            
            # Pull the arm for the bandit with the largest sample
            x = bandits[j].pull()
            
            # Update rewards log
            rewards[i] = x
            bandit_selections.append(j)
            
            # Update the distribution for the bandit whose arm we just pulled
            bandits[j].update(x)
            
            if i % 5000 == 0 and i > 0:
                logger.info(f"Trial {i}: epsilon={eps:.4f}, selected arm={j}, "
                          f"cumulative reward={np.sum(rewards[:i+1]):.2f}")
        
        logger.info("Epsilon-Greedy experiment completed")
        logger.info(f"Times explored: {num_times_explored}")
        logger.info(f"Times exploited: {num_times_exploited}")
        logger.info(f"Times optimal: {num_optimal}")
        
        # Store results in the bandit instance for reporting
        self.rewards = rewards
        self.bandit_selections = bandit_selections
        self.num_times_explored = num_times_explored
        self.num_times_exploited = num_times_exploited
        self.num_optimal = num_optimal
        self.optimal_reward = optimal_reward
        self.bandits = bandits
        
        return bandits, rewards, num_times_explored, num_times_exploited, num_optimal
    
    def report(self):
        """Generate report with statistics and save to CSV"""
        if not hasattr(self, 'rewards'):
            logger.error("No experiment data to report. Run experiment() first.")
            return
        
        total_reward = np.sum(self.rewards)
        avg_reward = np.mean(self.rewards)
        
        # Calculate regret
        regret_per_trial = np.array([self.optimal_reward - r for r in self.rewards])
        total_regret = np.sum(regret_per_trial)
        avg_regret = np.mean(regret_per_trial)  # it has length = num_trials, thus np.mean() will work as expected
        
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
        logger.info(f" # of times selected the optimal bandit: {self.num_optimal})")
        logger.info(f"\nArm Statistics:")
        for i, b in enumerate(self.bandits):
            logger.info(f"  {b} -> Estimated: {b.m_estimate:.4f}, Pulls: {b.N}")
        logger.info(f"{'='*60}")
        
        # Save to CSV
        df = pd.DataFrame({
            'Bandit': self.bandit_selections,
            'Reward': self.rewards,
            'Algorithm': 'EpsilonGreedy'
        })
        df.to_csv('Datasets/epsilon_greedy_results.csv', index=False)
        logger.info("Results saved to 'epsilon_greedy_results.csv'")

#--------------------------------------#
class ThompsonSampling(Bandit):
    """Individual Bandit Arm for Thompson Sampling algorithm"""
    
    def __init__(self, p):
        """
        Initialize a single bandit arm for Gaussian Thompson Sampling
        p: true mean reward (μ) for this arm
        Prior: N(0, 1)
        Known precision τ = 1
        """
        self.true_mean = p
        self.m = 0 
        self.lambda_ = 1 
        self.tau = 1 
        self.N = 0
        self.sum_x = 0
    
    def __repr__(self):
        return f'An Arm with true_mean={self.true_mean}'
    
    def pull(self):
        """Pull this arm and return reward from N(true_mean, 1/τ)"""
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean
    
    def sample(self):
        """Sample from the posterior distribution N(m, 1/λ)"""
        return np.random.randn() / np.sqrt(self.lambda_) + self.m
    
    def update(self, x):
        """Update the Gaussian posterior parameters based on observed reward x

        Args:
          x: 

        Returns:

        """
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = (self.tau * self.sum_x) / self.lambda_
        self.N += 1
    
    def experiment(self, num_trials=20000, bandits=None):
        """Run the Thompson Sampling experiment with multiple bandits
        This method should be called on one bandit instance but manages all bandits

        Args:
          num_trials:  (Default value = 20000)
          bandits:  (Default value = None)

        Returns:

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
                logger.info(f"Trial {i}: selected arm={j}, "
                          f"cumulative reward={np.sum(rewards[:i+1]):.2f}")
        
        logger.info("Thompson Sampling experiment completed")
        logger.info(f"Times optimal: {num_optimal}")
        
        return bandits, rewards, num_optimal
    
    def report(self, bandits, rewards, num_optimal):
        """

        Args:
          bandits: 
          rewards: 
          num_optimal: 

        Returns:

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
            logger.info(f"  {b} -> Posterior Mean (m): {b.m:.4f}, Posterior Precision (λ): {b.lambda_:.4f}, Pulls: {b.N}")
        logger.info(f"{'='*60}")
        
        # Save to CSV
        df = pd.DataFrame({
            'Trial': range(len(rewards)),
            'Reward': rewards,
            'Algorithm': 'ThompsonSampling'
        })
        df.to_csv('Datasets/thompson_sampling_results.csv', index=False)
        logger.info("Results saved to 'thompson_sampling_results.csv'")
        
        # Store for visualization
        self.rewards = rewards
        self.optimal_reward = optimal_reward

def comparison(eg_bandit, ts_bandit):
    """Compare the performances of the two algorithms visually

    Args:
      eg_bandit: 
      ts_bandit: 

    Returns:

    """
    logger.info("Generating comparison visualizations...")
    
    df_eg = pd.read_csv('Datasets/epsilon_greedy_results.csv')
    df_ts = pd.read_csv('Datasets/thompson_sampling_results.csv')
    df_combined = pd.concat([df_eg, df_ts], ignore_index=True)
    df_combined.to_csv('Datasets/combined_results.csv', index=False)
    logger.info("Combined results saved to 'combined_results.csv'")
    
    viz = Visualization(eg_bandit, ts_bandit)
    viz.plot1()
    viz.plot2()
    
    eg_total = np.sum(eg_bandit.rewards)
    ts_total = np.sum(ts_bandit.rewards)
    eg_regret = np.sum([eg_bandit.optimal_reward - r for r in eg_bandit.rewards])
    ts_regret = np.sum([ts_bandit.optimal_reward - r for r in ts_bandit.rewards])
    
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
    np.random.seed(42)
    
    logger.debug("Debug message - detailed diagnostic information")
    logger.info("Info message - general information about program execution")
    logger.warning("Warning message - something unexpected happened")
    logger.error("Error message - a serious problem occurred")
    logger.critical("Critical message - program may not be able to continue")
    
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
    for i, b in enumerate(eg_bandits):
        logger.debug(f"Bandit {i}: {b}")
    
    eg_bandits[0].experiment(num_trials=NumberOfTrials, bandits=eg_bandits, initial_epsilon=1.0)
    eg_bandits[0].report()
    
    logger.info("\n" + "="*60)
    logger.info("Creating Thompson Sampling Bandits...")
    ts_bandits = [ThompsonSampling(p) for p in Bandit_Reward]
    for i, b in enumerate(ts_bandits):
        logger.debug(f"Bandit {i}: {b}")
    
    bandits_ts, rewards_ts, optimal_ts = ts_bandits[0].experiment(
        num_trials=NumberOfTrials, bandits=ts_bandits
    )
    ts_bandits[0].report(bandits_ts, rewards_ts, optimal_ts)
    
    logger.info("\n" + "="*60)
    comparison(eg_bandits[0], ts_bandits[0])
    
    logger.info("\nExperiment completed successfully!")