import gymnasium as gym
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

import torch.nn as nn
import numpy as np
import time
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.distributions import Normal

env = gym.make("CarRacing-v3", continuous=True)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            flat_size = self.cnn(dummy).shape[1]
        self.shared = nn.Linear(flat_size, 256)
        self.actor_mean = nn.Linear(256, 3)
        self.actor_log_std = nn.Parameter(torch.zeros(3))
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.relu(self.shared(x))
        mean = torch.tanh(self.actor_mean(x))
        std = self.actor_log_std.exp()
        value = self.critic(x)
        return mean, std, value


def rescale_action(raw_action):
    steer = raw_action[0]
    gas = (raw_action[1] + 1) / 2
    brake = (raw_action[2] + 1) / 2
    return np.array([steer, gas, brake], dtype=np.float32)


def preprocess(obs):
    return torch.tensor(obs / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def collect_rollout(model, env, T):
    states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
    obs, _ = env.reset()
    for t in range(T):
        obs_t = preprocess(obs)
        mean, std, value = model(obs_t)
        dist = Normal(mean, std)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(-1)
        action = rescale_action(raw_action.squeeze(0).detach().numpy())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(obs_t)
        actions.append(raw_action)
        rewards.append(reward)
        values.append(value.item())
        log_probs.append(log_prob)
        dones.append(done)
        obs = next_obs if not done else env.reset()[0]
    return states, actions, rewards, values, log_probs, dones


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def ppo_update(model, optimizer, states, actions, old_log_probs, advantages, returns,
               clip_eps=0.2, c1=1.0, c2=0.01, epochs=4, batch_size=32):
    states_batch = torch.cat(states, dim=0)
    actions_batch = torch.cat(actions, dim=0)
    old_log_probs_batch = torch.cat(old_log_probs).detach()
    advantages_batch = torch.tensor(advantages, dtype=torch.float32)
    returns_batch = torch.tensor(returns, dtype=torch.float32)
    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

    T = states_batch.shape[0]
    policy_losses, value_losses, entropy_terms = [], [], []

    for epoch in range(epochs):
        indices = torch.randperm(T)
        for start in range(0, T, batch_size):
            idx = indices[start:start + batch_size]
            mean, std, value = model(states_batch[idx])
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions_batch[idx]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            ratio = torch.exp(new_log_probs - old_log_probs_batch[idx])
            surr1 = ratio * advantages_batch[idx]
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_batch[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((value.squeeze(-1) - returns_batch[idx]) ** 2).mean()
            loss = policy_loss + c1 * value_loss - c2 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_terms.append(entropy.item())

    return float(np.mean(policy_losses)), float(np.mean(value_losses)), float(np.mean(entropy_terms))


model = ActorCritic()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

T = 256
N_ITERATIONS = 500
log_rows = []

print(f"starting training: {N_ITERATIONS} iterations, T={T}", flush=True)
for it in range(N_ITERATIONS):
    start = time.time()
    states, actions, rewards, values, log_probs, dones = collect_rollout(model, env, T)
    advantages, returns = compute_gae(rewards, values, dones)
    p_loss, v_loss, ent = ppo_update(model, optimizer, states, actions, log_probs, advantages, returns)
    elapsed = time.time() - start

    mean_reward = sum(rewards) / T
    log_rows.append([it, mean_reward, p_loss, v_loss, ent, elapsed])
    print(f"iter {it}: mean_reward={mean_reward:.3f} p_loss={p_loss:.4f} v_loss={v_loss:.4f} "
          f"entropy={ent:.3f} time={elapsed:.1f}s", flush=True)

with open("training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "mean_reward", "policy_loss", "value_loss", "entropy", "seconds"])
    writer.writerows(log_rows)

print("calibration done", flush=True)

print("training done, generating plots", flush=True)

iterations = [row[0] for row in log_rows]
mean_rewards = [row[1] for row in log_rows]
policy_losses = [row[2] for row in log_rows]
value_losses = [row[3] for row in log_rows]
entropies = [row[4] for row in log_rows]

plt.figure()
plt.plot(iterations, mean_rewards)
plt.xlabel("Iteration")
plt.ylabel("Mean reward per step")
plt.title("PPO on CarRacing-v3, from scratch: reward over training")
plt.savefig("reward_plot.png")
plt.close()

plt.figure()
plt.plot(iterations, policy_losses, label="policy loss")
plt.plot(iterations, value_losses, label="value loss")
plt.plot(iterations, entropies, label="entropy")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.title("PPO on CarRacing-v3, from scratch: losses over training")
plt.legend()
plt.savefig("loss_plot.png")
plt.close()

print("saved reward_plot.png and loss_plot.png", flush=True)
