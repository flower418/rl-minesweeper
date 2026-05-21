from dataclasses import dataclass, field
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import wandb

from minesweeper_env import MinesweeperEnv
from ppo_network import ActorCritic


# ============================================================
#  Config
# ============================================================

@dataclass
class PPOConfig:
    # ---- RL 超参 ----
    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    steps_per_epoch: int = 256
    num_epochs: int = 2000

    # ---- 环境 ----
    board_size: int = 9
    num_mines: int = 10
    max_steps: int = 200

    # ---- 奖励 ----
    reward_win: float = 20.0
    reward_lose: float = -20.0
    reward_reveal: float = 1.0
    reward_flag_toggle: float = -0.02
    reward_flag_right: float = 2 
    reward_flag_wrong: float = -2
    reward_invalid: float = -0.1

    # ---- wandb ----
    use_wandb: bool = True
    wandb_project: str = "rl-minesweeper"
    wandb_name: str = "test"


# ============================================================
#  GAE
# ============================================================

def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nxt = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nxt - values[t]
        gae = delta + gamma * lam * nxt * gae
        adv[t] = gae
    return adv, adv + values[:-1]


# ============================================================
#  Rollout
# ============================================================

def rollout(env, policy, device, n_steps):
    states, actions, rewards, dones = [], [], [], []
    log_probs, values = [], []
    episode_rewards, ep_reward = [], 0.0

    s, _ = env.reset()
    for _ in range(n_steps):
        t = torch.tensor(s, dtype=torch.float32, device=device)
        with torch.no_grad():
            logit, v = policy(t)

        dist = Categorical(logits=logit)
        a = dist.sample()

        ns, r, term, trunc, _ = env.step(a.item())
        done = term or trunc

        states.append(s)
        actions.append(a.item())
        rewards.append(r)
        dones.append(done)
        log_probs.append(dist.log_prob(a).item())
        values.append(v.item())
        ep_reward += r

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            s, _ = env.reset()
        else:
            s = ns

    # bootstrap value
    t = torch.tensor(s, dtype=torch.float32, device=device)
    with torch.no_grad():
        _, next_v = policy(t)
    values.append(next_v.item())

    return (
        torch.tensor(np.array(states), dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.long, device=device),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        torch.tensor(log_probs, dtype=torch.float32, device=device),
        np.array(values, dtype=np.float32),
        episode_rewards,
    )


# ============================================================
#  PPO-Clip Update
# ============================================================

def ppo_update(policy, optimizer, cfg, states, actions, rewards, dones,
               old_log_probs, values):
    adv, ret = compute_gae(rewards, values, dones, cfg.gamma, cfg.lam)
    adv_t = torch.tensor(adv, dtype=torch.float32, device=states.device)
    ret_t = torch.tensor(ret, dtype=torch.float32, device=states.device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    p_loss = v_loss = 0.0
    for _ in range(cfg.update_epochs):
        logits, v_pred = policy(states)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(actions)

        ratio = torch.exp(new_lp - old_log_probs)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(v_pred, ret_t)
        entropy = dist.entropy().mean()
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p_loss += policy_loss.item()
        v_loss += value_loss.item()

    return p_loss / cfg.update_epochs, v_loss / cfg.update_epochs


# ============================================================
#  Train
# ============================================================

def train(env, policy, optimizer, cfg, device):
    for epoch in range(cfg.num_epochs):
        states, actions, rewards, dones, old_lp, values, ep_rewards = rollout(
            env, policy, device, cfg.steps_per_epoch
        )
        p_loss, v_loss = ppo_update(
            policy, optimizer, cfg, states, actions, rewards, dones, old_lp, values
        )

        avg_r = np.mean(ep_rewards) if ep_rewards else 0.0
        wins = sum(1 for r in ep_rewards if r > 0)

        if cfg.use_wandb:
            wandb.log({
                "avg_reward": avg_r,
                "wins": wins,
                "episodes": len(ep_rewards),
                "policy_loss": p_loss,
                "value_loss": v_loss,
            }, step=epoch)

        if epoch % 50 == 0:
            print(f"epoch {epoch:4d} | avg_reward: {avg_r:8.2f} "
                  f"| wins: {wins} | eps: {len(ep_rewards):3d} "
                  f"| p_loss: {p_loss:.4f} | v_loss: {v_loss:.4f}")


# ============================================================
#  Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="wandb run name")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PPOConfig()

    if args.name:
        cfg.wandb_name = args.name
    if args.no_wandb:
        cfg.use_wandb = False

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg.__dict__)

    env = MinesweeperEnv(
        width=cfg.board_size, height=cfg.board_size,
        num_mines=cfg.num_mines, max_steps=cfg.max_steps,
        reward_win=cfg.reward_win, reward_lose=cfg.reward_lose,
        reward_reveal=cfg.reward_reveal,
        reward_flag_toggle=cfg.reward_flag_toggle,
        reward_flag_right=cfg.reward_flag_right,
        reward_flag_wrong=cfg.reward_flag_wrong,
        reward_invalid=cfg.reward_invalid,
    )
    policy = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    train(env, policy, optimizer, cfg, device)

    os.makedirs("exp", exist_ok=True)
    torch.save(policy.state_dict(), "exp/ppo_minesweeper.pth")
    print("saved exp/ppo_minesweeper.pth")

    if cfg.use_wandb:
        wandb.finish()