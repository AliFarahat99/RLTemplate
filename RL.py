import gym
from gym import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.current_step = 0
        self.positions = 0  # 1 for long, -1 for short, 0 for neutral
        self.trade_history = []

        # Define action and observation space
        self.action_space = spaces.Discrete(5)  # 0: Hold, 1: Buy, 2: Sell, 3: Short, 4: Cover
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32)  # Asset price, %K, %D, position
        
    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.current_step = 0
        self.positions = 0
        self.trade_history = []
        return self._next_observation()
    
    def _next_observation(self):
        obs = np.array([
            self.df['Close'].iloc[self.current_step],
            self.df['%K'].iloc[self.current_step],
            self.df['%D'].iloc[self.current_step],
            self.positions
        ])
        return obs
    
    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step]
        previous_net_worth = self.net_worth
        if action == 1:  # Buy
            if self.positions == 0:
                self.positions = 1
                self.trade_history.append(('buy', current_price))
            elif self.positions == -1:  # Cover short
                self.positions = 0
                self.trade_history.append(('cover', current_price))
    
        elif action == 2:  # Sell
            if self.positions == 1:
                self.positions = 0
                self.trade_history.append(('sell', current_price))
    
        elif action == 3:  # Short
            if self.positions == 0:
                self.positions = -1
                self.trade_history.append(('short', current_price))
    
        # Update balance and net worth
        if self.positions == 1:
            self.net_worth = self.balance * (current_price / self.trade_history[-1][1])
        elif self.positions == -1:
            self.net_worth = self.balance * (self.trade_history[-1][1] / current_price)
        else:
            self.net_worth = self.balance
    
        # Penalize the agent for losing wealth
        reward = self.net_worth - self.initial_balance

        if ( ((previous_net_worth - self.net_worth)/previous_net_worth) > 0.06): 
            reward -= -10000
        if ( ((self.net_worth - previous_net_worth)/previous_net_worth) > 0.01): 
            reward += 100000

        # Check if net worth becomes zero or negative
        if self.net_worth <= 0:
            done = True
            reward = -10000000000  # Penalize heavily for failure
        else:
            done = False
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth        
        trailing_stop = 0.90 * self.max_net_worth
        if self.net_worth < trailing_stop:
            reward -= 50000  # Penalty for hitting trailing stop-loss

        self.balance = self.net_worth
    
        self.current_step += 1
    
        if self.current_step >= len(self.df) - 1:
            done = True
    
        obs = self._next_observation()
    
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Positions: {self.positions}')
        print(f'Trade History: {self.trade_history}')

stock_symbol = 'AAPL'  # Example: Apple Inc.
start_date = '2020-01-01'
end_date = '2023-01-01'

# Fetch historical data using yfinance
data = yf.download(stock_symbol, start=start_date, end=end_date)

df=data

df['L14'] = df['Low'].rolling(window=14).min()
df['H14'] = df['High'].rolling(window=14).max()
df['%K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14']))
df['%D'] = df['%K'].rolling(window=3).mean()
df = df.dropna()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the environments for training and testing
env_train = TradingEnv(train_df)
env_train = DummyVecEnv([lambda: env_train])

env_test = TradingEnv(test_df)
env_test = DummyVecEnv([lambda: env_test])

model = PPO('MlpPolicy', env_train, verbose=1)

# Train the agent
model.learn(total_timesteps=25000)

model.save("ppo_trading_model")

# Load the trained model
model = PPO.load("ppo_trading_model", env=env_test)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env_test, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward}')

# Plotting the performance of the agent
obs = env_test.reset()
net_worths = []
for _ in range(len(df)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env_test.step(action)
    net_worths.append(env_test.envs[0].net_worth)
    if done:
        break

plt.plot(net_worths)
plt.xlabel('Steps')
plt.ylabel('Net Worth')
plt.title('Agent Net Worth Over Time')
plt.savefig('RLOUTPUT.png')  # Save the plot as a PNG file
plt.close()  # Close the plot to free up resources