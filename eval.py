"""加载 checkpoint 推理，输出胜率，详细过程写入日志。"""
import argparse
import os
import torch
from datetime import datetime
from minesweeper_env import MinesweeperEnv
from ppo_network import ActorCritic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path to checkpoint.pth")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--logdir", type=str, default="eval")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    with open(log_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"games: {args.games}\n\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = ActorCritic()
    policy.load_state_dict(torch.load(args.checkpoint, map_location=device))
    policy.to(device)
    policy.eval()

    env = MinesweeperEnv(render_mode="ansi")

    wins = 0
    for ep in range(args.games):
        s, _ = env.reset()
        done = False
        total_r = 0.0
        step = 0

        lines = []
        lines.append(f"Episode {ep + 1}")
        lines.append(env.render())

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
            lines.append(f"step {step:3d}  {act}({pos//9},{pos%9})  r={r:+.1f}")
            lines.append(env.render())

        if total_r > 0:
            lines.append("WIN")
            wins += 1
        else:
            lines.append("LOSE")
        lines.append("")

        with open(log_path, "a") as f:
            f.write("\n".join(lines))

        print(f"[{ep + 1:3d}/{args.games}]  wins: {wins}")

    print(f"\nwin rate: {wins}/{args.games} = {wins/args.games*100:.1f}%")
    print(f"log saved to {log_path}")


if __name__ == "__main__":
    main()
