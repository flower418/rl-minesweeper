import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MinesweeperEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, width=9, height=9, num_mines=10, render_mode=None,
                 reward_win=10.0, reward_lose=-10.0, reward_reveal=0.3,
                 reward_flag=0.0, reward_invalid=-0.5):
        super().__init__()
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.render_mode = render_mode
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        self.reward_reveal = reward_reveal
        self.reward_flag = reward_flag
        self.reward_invalid = reward_invalid

        # 162 个离散动作: 81 个位置 × 2 种操作 (翻开 / 标旗)
        self.action_space = spaces.Discrete(width * height * 2)

        # 状态: 3 通道 × height × width, 值域 [0, 8]
        self.observation_space = spaces.Box(
            low=0, high=8,
            shape=(3, height, width),
            dtype=np.int8
        )

        # 内部状态
        self.mine_grid = None       # -1=雷, 0~8=周围雷数
        self.revealed = None        # bool 网格
        self.flagged = None         # bool 网格
        self.first_click = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mine_grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.revealed = np.zeros((self.height, self.width), dtype=bool)
        self.flagged = np.zeros((self.height, self.width), dtype=bool)
        self.first_click = True
        return self._get_obs(), {}

    def step(self, action):
        action_type = action // (self.width * self.height)
        pos_action = action % (self.width * self.height)
        row, col = divmod(pos_action, self.width)

        if action_type == 1:
            return self._handle_flag(row, col)

        # action_type == 0: 翻开
        if self.revealed[row, col]:
            return self._get_obs(), self.reward_invalid, False, False, {}
        if self.flagged[row, col]:
            return self._get_obs(), self.reward_invalid, False, False, {}

        if self.first_click:
            self._place_mines(safe_row=row, safe_col=col)
            self.first_click = False

        if self.mine_grid[row, col] == -1:
            self.revealed[row, col] = True
            return self._get_obs(), self.reward_lose, True, False, {}

        self._flood_fill(row, col)

        all_safe_revealed = np.all(self.revealed | (self.mine_grid == -1))
        if all_safe_revealed:
            return self._get_obs(), self.reward_win, True, True, {}

        return self._get_obs(), self.reward_reveal, False, False, {}

    def _handle_flag(self, row, col):
        if self.revealed[row, col]:
            return self._get_obs(), self.reward_invalid, False, False, {}
        self.flagged[row, col] = not self.flagged[row, col]
        return self._get_obs(), self.reward_flag, False, False, {}

    # ---------- 核心逻辑 ----------

    def _place_mines(self, safe_row, safe_col):
        safe_set = {(safe_row, safe_col)}
        for nr, nc in self._neighbors(safe_row, safe_col):
            safe_set.add((nr, nc))
        candidates = [(r, c) for r in range(self.height)
                      for c in range(self.width)
                      if (r, c) not in safe_set]
        mine_positions = self.np_random.choice(
            len(candidates), self.num_mines, replace=False
        )
        for idx in mine_positions:
            r, c = candidates[idx]
            self.mine_grid[r, c] = -1

        for r in range(self.height):
            for c in range(self.width):
                if self.mine_grid[r, c] == -1:
                    continue
                count = sum(
                    1 for nr, nc in self._neighbors(r, c)
                    if self.mine_grid[nr, nc] == -1
                )
                self.mine_grid[r, c] = count

    def _flood_fill(self, row, col):
        if self.revealed[row, col]:
            return
        self.revealed[row, col] = True
        if self.mine_grid[row, col] == 0:
            for nr, nc in self._neighbors(row, col):
                self._flood_fill(nr, nc)

    def _neighbors(self, row, col):
        dirs = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]
        return [
            (row + dr, col + dc) for dr, dc in dirs
            if 0 <= row + dr < self.height and 0 <= col + dc < self.width
        ]

    # ---------- 观测 ----------

    def _get_obs(self):
        obs = np.zeros((3, self.height, self.width), dtype=np.int8)
        revealed_vals = np.where(self.revealed, self.mine_grid, 0)
        revealed_vals = np.clip(revealed_vals, 0, 8)
        obs[0] = revealed_vals
        obs[1] = self.flagged.astype(np.int8)
        obs[2] = (~self.revealed & ~self.flagged).astype(np.int8)
        return obs

    # ---------- 渲染 ----------

    def render(self):
        if self.render_mode == "ansi":
            lines = []
            col_header = "   " + " ".join(str(c) for c in range(self.width))
            lines.append(col_header)
            for r in range(self.height):
                row_str = f"{r:2d} "
                for c in range(self.width):
                    if self.revealed[r, c]:
                        if self.mine_grid[r, c] == -1:
                            row_str += "* "
                        else:
                            row_str += f"{self.mine_grid[r, c]} "
                    elif self.flagged[r, c]:
                        row_str += "F "
                    else:
                        row_str += ". "
                lines.append(row_str)
            return "\n".join(lines)

    def close(self):
        pass
