import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from minesweeper_env import MinesweeperEnv

from ppo_network import ActorCritic

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    # rewards: T
    # values: T+1, 最后一个 T+1 表示未来的 value, termination 时 value=0
    # dones: T, 每步是否完成
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0

    # 倒着往回算，最后一个 value 是已知的
    for t in reversed(range(T)):
        next_none_terminal = 1 - dones[t] # 如果是 T，那么 V(s_T)=0，且没有 V(s_{T+1})，所以要乘一个系数删除这一步
        delta = rewards[t] + gamma * values[t + 1] * next_none_terminal - values[t]

        gae = delta + lam * gamma * next_none_terminal * advantages[t + 1]
        advantages[t] = gae

        returns = advantages + values[:-1] # R = A - V，移项即可，用于网络训练出更好的 values
        return advantages, returns

def rollout(env, policy, device, T=256):
    states, actions, rewards, dones, old_log_probs, old_values = [], [], [], [], [], []

    state, _ = env.reset()
    episode_rewards = []

    for _ in range(T):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, value = policy(state_tensor)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        old_log_probs.append(log_prob.item())
        old_values.append(value)
        episode_rewards.append(reward)

        if done:
            state, _ = env.reset()
        else:
            state = next_state

    # 最后一个状态值
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
    with torch.no_grad():
        _, next_value = policy(state_tensor)
    old_values.append(next_value)

    return {
        "states": torch.tensor(np.array(states), dtype=torch.float32).to(device),
        "actions": torch.tensor(actions, dtype=torch.long).to(device),
        "rewards": np.array(rewards, dtype=torch.float32),
        "dones": np.array(dones, dtype=np.float32),
        "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32).to(device),
        "old_values": np.array(old_values, dtype=np.float32),
        "episode_rewardds": np.array(episode_rewards),
    }

def ppo_update(policy, optimizer, batch, clip_eps=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4, device="cpu"):
    states = batch["states"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probs"]

    advantages, returns = compute_gae(batch["rewards"], batch["values"], batch["dones"])
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0

    for _ in range(epochs):
        logits, values = policy(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(returns, values)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / epochs, total_value_loss / epochs
   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
BOARD_WIDTH = 9
BOARD_HEIGHT = 9
NUM_ACTIONS = BOARD_WIDTH * BOARD_HEIGHT * 2
ROLLOUT_STEPS = 2048
LR = 3e-4
TOTAL_TIMESTEPS = 500000
GAMMA = 0.99
LAMBDA = 0.95
EPOCHS_PER_ROLLOUT = 4
BATCH_SIZE = 256
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.5
MAX_GRAD_NORM = 0.5

def train(env, policy, optimizer, num_epochs=500, steps_per_epoch=256, device="cpu"):
    for epoch in range(num_epochs):
        batch = rollout(env, policy, device, T = steps_per_epoch)
        p_loss, v_loss = ppo_update(policy, optimizer, batch, device=device)

        if epoch % 10 == 0:
            ep_rewards = batch["episode_rewards"]
            avg_ep_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            num_eps = len(ep_rewards)
            print(f"epoch {epoch:4d} | avg_ep_reward {avg_ep_reward:8.2f} | episodes {num_eps:3d} | p_loss {p_loss:.4f} | v_loss {v_loss:.4f}")

    return policy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    env = MinesweeperEnv()
    policy = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    policy = train(env, policy, optimizer, num_epochs=500, device=device)

    torch.save(policy.state_dict(), "ppo_minesweeper.pth")
    print("saved ppo_minesweeper.pth")