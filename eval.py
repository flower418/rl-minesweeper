"""加载训练好的模型，在扫雷环境上推理，结果写入日志文件。"""
import argparse
import os
import torch
from datetime import datetime
from minesweeper_env import MinesweeperEnv
from ppo_network import ActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="exp/ppo_minesweeper.pth")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--logdir", type=str, default="exp")
    parser.add_argument("--render", action="store_true", help="also print to stdout")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def log(msg=""):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
        if args.render:
            print(msg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = ActorCritic()
    policy.load_state_dict(torch.load(args.model, map_location=device))
    policy.to(device)
    policy.eval()

    env = MinesweeperEnv(render_mode="ansi")

    wins = 0
    for ep in range(args.games):
        s, _ = env.reset()
        done = False
        total_r = 0.0
        step = 0
        episode_log = []

        episode_log.append(f"{'='*30}")
        episode_log.append(f"  Episode {ep + 1}/{args.games}")
        episode_log.append(f"{'='*30}")
        episode_log.append(env.render())

        while not done:
            t = torch.tensor(s, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = policy(t)
            a = torch.argmax(logits, dim=-1).item()

            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_r += r
            step += 1

            act = "F" if a >= 81 else "R"
            pos = a % 81
            episode_log.append(f"step {step:3d}  {act}({pos//9},{pos%9})  r={r:+.1f}")
            episode_log.append(env.render())

        if total_r > 0:
            episode_log.append("  >>> WIN <<<")
            wins += 1
        else:
            episode_log.append("  >>> LOSE <<<")

        log("\n".join(episode_log))

    log(f"\nwins: {wins}/{args.games}")
    log(f"log saved to {log_path}")


if __name__ == "__main__":
    main()
