import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CarRacing-v3", continuous=True)

model = PPO("CnnPolicy", env, device="cpu", verbose=1, tensorboard_log="./tensorboard_carracing/")
model.learn(total_timesteps=10000)
model.save("ppo_carracing")

print("Done Training")
