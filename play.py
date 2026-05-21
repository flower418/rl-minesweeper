from minesweeper_env import MinesweeperEnv

env = MinesweeperEnv(width=9, height=9, num_mines=10, render_mode="ansi")
obs, _ = env.reset()

print("=== 扫雷 ===")
print("输入格式:  row col   (翻开)")
print("输入格式:  f row col (标旗/取消)")
print("输入格式:  q         (退出)")
print()

while True:
    print(env.render())
    print()
    cmd = input("> ").strip().lower()
    if cmd == "q":
        break

    parts = cmd.split()
    if len(parts) < 2:
        print("格式: row col  或  f row col")
        continue

    if parts[0] == "f":
        row, col = int(parts[1]), int(parts[2])
        action = 1 * (env.width * env.height) + row * env.width + col
    else:
        row, col = int(parts[0]), int(parts[1])
        action = 0 * (env.width * env.height) + row * env.width + col

    obs, reward, terminated, truncated, info = env.step(action)
    print(f"reward: {reward}")
    if terminated:
        print(env.render())
        print("你赢了!" if info.get("is_win", False) else "踩雷了!")
        break

env.close()
