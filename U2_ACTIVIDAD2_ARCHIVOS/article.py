import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class MultiArmedBandit:
    """Base class for Multi-Armed Bandit algorithms"""
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.rewards = [[] for _ in range(n_arms)]
        self.selections = []
        self.cumulative_reward = 0
        self.t = 0
    
    def select_arm(self) -> int:
        """Select an arm based on the algorithm strategy"""
        raise NotImplementedError
    
    def update(self, arm: int, reward: float):
        """Update the algorithm with observed reward"""
        self.rewards[arm].append(reward)
        self.selections.append(arm)
        self.cumulative_reward += reward
        self.t += 1

class UCBAlgorithm(MultiArmedBandit):
    """Upper Confidence Bound Algorithm"""
    
    def __init__(self, n_arms: int, confidence: float = 2.0):
        super().__init__(n_arms)
        self.confidence = confidence
        
    def select_arm(self) -> int:
        # If any arm hasn't been pulled, select it
        for arm in range(self.n_arms):
            if len(self.rewards[arm]) == 0:
                return arm
        
        # Calculate UCB for each arm
        ucb_values = []
        for arm in range(self.n_arms):
            mean_reward = np.mean(self.rewards[arm])
            n_selections = len(self.rewards[arm])
            confidence_interval = np.sqrt(
                (self.confidence * np.log(self.t)) / n_selections
            )
            ucb = mean_reward + confidence_interval
            ucb_values.append(ucb)
        
        return np.argmax(ucb_values)

class ThompsonSampling(MultiArmedBandit):
    """Thompson Sampling Algorithm using Beta-Bernoulli conjugate prior"""
    
    def __init__(self, n_arms: int, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(n_arms)
        self.alpha = [alpha] * n_arms
        self.beta = [beta] * n_arms
        
    def select_arm(self) -> int:
        # Sample from Beta distribution for each arm
        samples = []
        for arm in range(self.n_arms):
            sample = np.random.beta(self.alpha[arm], self.beta[arm])
            samples.append(sample)
        
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        # Update Beta parameters (assuming rewards are normalized to [0,1])
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)

class SolarBanditExperiment:
    """Experiment class for solar inverter optimization using Multi-Armed Bandits"""
    
    def __init__(self, data_path_gen: str, data_path_weather: str):
        self.generation_data = pd.read_csv(data_path_gen)
        self.weather_data = pd.read_csv(data_path_weather)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean the solar generation data"""
        # Convert DATE_TIME to datetime
        self.generation_data['DATE_TIME'] = pd.to_datetime(self.generation_data['DATE_TIME'])
        self.weather_data['DATE_TIME'] = pd.to_datetime(self.weather_data['DATE_TIME'])
        
        # Filter out zero power readings
        self.generation_data = self.generation_data[
            (self.generation_data['DC_POWER'] > 0) & 
            (self.generation_data['AC_POWER'] > 0)
        ].copy()
        
        # Calculate efficiency as reward
        self.generation_data['EFFICIENCY'] = (
            self.generation_data['AC_POWER'] / self.generation_data['DC_POWER']
        )
        
        # Normalize efficiency to [0,1] for Thompson Sampling
        max_eff = self.generation_data['EFFICIENCY'].max()
        min_eff = self.generation_data['EFFICIENCY'].min()
        self.generation_data['NORMALIZED_EFFICIENCY'] = (
            (self.generation_data['EFFICIENCY'] - min_eff) / (max_eff - min_eff)
        )
        
        # Get unique inverters (arms)
        self.inverters = sorted(self.generation_data['SOURCE_KEY'].unique())
        self.n_arms = len(self.inverters)
        self.inverter_to_idx = {inv: idx for idx, inv in enumerate(self.inverters)}
        
        print(f"Dataset prepared:")
        print(f"- Number of inverters (arms): {self.n_arms}")
        print(f"- Total observations: {len(self.generation_data)}")
        print(f"- Date range: {self.generation_data['DATE_TIME'].min()} to {self.generation_data['DATE_TIME'].max()}")
        
    def create_daily_aggregated_data(self):
        """Create daily aggregated data for bandit experiment"""
        # Group by date and inverter
        daily_data = self.generation_data.groupby([
            self.generation_data['DATE_TIME'].dt.date, 'SOURCE_KEY'
        ]).agg({
            'EFFICIENCY': 'mean',
            'NORMALIZED_EFFICIENCY': 'mean',
            'AC_POWER': 'sum',
            'DC_POWER': 'sum'
        }).reset_index()
        
        daily_data.columns = ['DATE', 'SOURCE_KEY', 'AVG_EFFICIENCY', 
                             'NORMALIZED_EFFICIENCY', 'TOTAL_AC_POWER', 'TOTAL_DC_POWER']
        
        return daily_data
    
    def run_bandit_experiment(self, n_days: int = None, algorithms: List[str] = ['UCB', 'Thompson']):
        """Run the bandit experiment"""
        daily_data = self.create_daily_aggregated_data()
        
        if n_days is None:
            n_days = len(daily_data['DATE'].unique())
        
        unique_dates = sorted(daily_data['DATE'].unique())[:n_days]
        
        # Initialize algorithms
        results = {}
        if 'UCB' in algorithms:
            results['UCB'] = {
                'algorithm': UCBAlgorithm(self.n_arms),
                'cumulative_reward': [0],
                'regret': [0],
                'selections': []
            }
        
        if 'Thompson' in algorithms:
            results['Thompson'] = {
                'algorithm': ThompsonSampling(self.n_arms),
                'cumulative_reward': [0],
                'regret': [0],
                'selections': []
            }
        
        # Calculate optimal reward for regret calculation
        optimal_rewards = []
        for date in unique_dates:
            day_data = daily_data[daily_data['DATE'] == date]
            if len(day_data) > 0:
                optimal_reward = day_data['NORMALIZED_EFFICIENCY'].max()
                optimal_rewards.append(optimal_reward)
        
        # Run experiment for each day
        for day_idx, date in enumerate(unique_dates):
            day_data = daily_data[daily_data['DATE'] == date]
            
            if len(day_data) == 0:
                continue
                
            # Get available arms for this day
            available_arms = [self.inverter_to_idx[inv] for inv in day_data['SOURCE_KEY'].unique()]
            
            for algo_name, result in results.items():
                algorithm = result['algorithm']
                
                # Select arm
                selected_arm = algorithm.select_arm()
                
                # Ensure selected arm has data for this day
                if selected_arm not in available_arms:
                    selected_arm = np.random.choice(available_arms)
                
                # Get reward
                inverter_name = self.inverters[selected_arm]
                arm_data = day_data[day_data['SOURCE_KEY'] == inverter_name]
                
                if len(arm_data) > 0:
                    reward = arm_data['NORMALIZED_EFFICIENCY'].iloc[0]
                else:
                    reward = 0
                
                # Update algorithm
                algorithm.update(selected_arm, reward)
                
                # Record results
                result['selections'].append(selected_arm)
                result['cumulative_reward'].append(algorithm.cumulative_reward)
                
                # Calculate regret
                if day_idx < len(optimal_rewards):
                    instantaneous_regret = optimal_rewards[day_idx] - reward
                    total_regret = result['regret'][-1] + instantaneous_regret
                    result['regret'].append(total_regret)
                else:
                    result['regret'].append(result['regret'][-1])
        
        return results, unique_dates[:len(optimal_rewards)]
    
    def plot_results(self, results: Dict, dates: List):
        """Plot experiment results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot cumulative reward
        axes[0,0].set_title('Cumulative Reward Over Time')
        for algo_name, result in results.items():
            axes[0,0].plot(result['cumulative_reward'], label=algo_name, linewidth=2)
        axes[0,0].set_xlabel('Day')
        axes[0,0].set_ylabel('Cumulative Reward')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot cumulative regret
        axes[0,1].set_title('Cumulative Regret Over Time')
        for algo_name, result in results.items():
            axes[0,1].plot(result['regret'], label=algo_name, linewidth=2)
        axes[0,1].set_xlabel('Day')
        axes[0,1].set_ylabel('Cumulative Regret')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot arm selection frequency
        axes[1,0].set_title('Inverter Selection Frequency')
        for algo_name, result in results.items():
            selections = result['selections']
            unique, counts = np.unique(selections, return_counts=True)
            axes[1,0].bar(unique, counts, alpha=0.7, label=algo_name)
        axes[1,0].set_xlabel('Inverter Index')
        axes[1,0].set_ylabel('Selection Count')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot average efficiency by inverter
        axes[1,1].set_title('Average Efficiency by Inverter')
        daily_data = self.create_daily_aggregated_data()
        inverter_efficiency = daily_data.groupby('SOURCE_KEY')['AVG_EFFICIENCY'].mean()
        inverter_indices = [self.inverter_to_idx[inv] for inv in inverter_efficiency.index]
        axes[1,1].bar(inverter_indices, inverter_efficiency.values, alpha=0.7)
        axes[1,1].set_xlabel('Inverter Index')
        axes[1,1].set_ylabel('Average Efficiency')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_report(self, results: Dict, dates: List):
        """Generate experimental report"""
        print("=" * 60)
        print("SOLAR INVERTER BANDIT OPTIMIZATION REPORT")
        print("=" * 60)
        
        print(f"\nExperimental Setup:")
        print(f"- Number of inverters: {self.n_arms}")
        print(f"- Evaluation period: {len(dates)} days")
        print(f"- Date range: {dates[0]} to {dates[-1]}")
        
        print(f"\nAlgorithm Performance:")
        for algo_name, result in results.items():
            final_reward = result['cumulative_reward'][-1]
            final_regret = result['regret'][-1]
            print(f"\n{algo_name}:")
            print(f"  - Final cumulative reward: {final_reward:.4f}")
            print(f"  - Final cumulative regret: {final_regret:.4f}")
            print(f"  - Average reward per day: {final_reward/len(dates):.4f}")
            
            # Most selected inverter
            selections = result['selections']
            if selections:
                most_selected = max(set(selections), key=selections.count)
                selection_count = selections.count(most_selected)
                print(f"  - Most selected inverter: {self.inverters[most_selected]} ({selection_count} times)")
        
        # Statistical comparison
        if len(results) == 2:
            algo_names = list(results.keys())
            rewards_1 = np.diff(results[algo_names[0]]['cumulative_reward'])
            rewards_2 = np.diff(results[algo_names[1]]['cumulative_reward'])
            
            # Perform t-test
            if len(rewards_1) > 1 and len(rewards_2) > 1:
                t_stat, p_value = stats.ttest_ind(rewards_1, rewards_2)
                print(f"\nStatistical Comparison:")
                print(f"- T-statistic: {t_stat:.4f}")
                print(f"- P-value: {p_value:.4f}")
                print(f"- Significance: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")

# Main execution
def main():
    """Main function to run the solar bandit experiment"""
    print("Solar Inverter Multi-Armed Bandit Optimization")
    print("=" * 50)
    
    # Initialize experiment (you need to provide the correct file paths)
    # experiment = SolarBanditExperiment('Plant_1_Generation_Data.csv', 'Plant_1_Weather_Sensor_Data.csv')
    
    # For demonstration, we'll create synthetic data
    print("Creating synthetic solar data for demonstration...")
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-05-15', '2020-06-17', freq='D')
    inverters = [f'INV_{i:03d}' for i in range(22)]  # 22 inverters as in your data
    
    synthetic_data = []
    for date in dates:
        for inv in inverters:
            # Simulate efficiency with some inverters being better than others
            base_efficiency = np.random.normal(0.85, 0.05)  # Base efficiency around 85%
            # Add some inverter-specific bias
            inv_bias = hash(inv) % 100 / 1000  # Small bias based on inverter
            efficiency = max(0.1, min(0.95, base_efficiency + inv_bias))
            
            synthetic_data.append({
                'DATE_TIME': date,
                'SOURCE_KEY': inv,
                'DC_POWER': np.random.uniform(100, 1000),
                'AC_POWER': np.random.uniform(80, 800),
                'EFFICIENCY': efficiency,
                'NORMALIZED_EFFICIENCY': efficiency  # Already normalized for this example
            })
    
    df = pd.DataFrame(synthetic_data)
    
    # Create experiment class manually for synthetic data
    class SyntheticSolarExperiment(SolarBanditExperiment):
        def __init__(self, synthetic_df):
            self.generation_data = synthetic_df
            self.inverters = sorted(df['SOURCE_KEY'].unique())
            self.n_arms = len(self.inverters)
            self.inverter_to_idx = {inv: idx for idx, inv in enumerate(self.inverters)}
            print(f"Synthetic dataset created:")
            print(f"- Number of inverters (arms): {self.n_arms}")
            print(f"- Total observations: {len(self.generation_data)}")
    
    experiment = SyntheticSolarExperiment(df)
    
    # Run experiment
    print("\nRunning Multi-Armed Bandit experiment...")
    results, dates = experiment.run_bandit_experiment(n_days=30)
    
    # Generate report
    experiment.generate_report(results, dates)
    
    # Plot results
    print("\nGenerating plots...")
    fig = experiment.plot_results(results, dates)
    
    print("\nExperiment completed successfully!")
    
    return experiment, results, dates

if __name__ == "__main__":
    experiment, results, dates = main()