import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    # action space: 81*2，共 81 个格子，每个格子有 flag 或点开两种选择
    def __init__(self, board_width=9, board_height=9, num_actions=162):
        super().__init__()

        # 输入通道是 1 个通道，只有一个 9*9
        # 输出通道 16 维，hyperparameter
        # kernel_size=3，因为扫雷只需要关注周围 3*3
        # padding=1，保证每个通道都保持 9*9
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        conv_out_size = 32 * board_width * board_height # 总共 32 张这样的图片

        self.fc = nn.Linear(conv_out_size, 128)
        
        # 二者共享一个 conv+fc 架构
        # actor 用于输出 action space 中的每个动作
        # critic 用于输出对 value function 的估计
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        # x: (board_width, board_height)
        x.unsqueeze(1) # 在第 1 维添加，因为 cnn 要求的维数：(batch, in_channels, width, height) (1, 9, 9)

        x = F.relu(self.conv1(x)) # (batch, 16, 9, 9)
        x = F.relu(self.conv2(x)) # (batch, 32, 9, 9)

        x.flatten(start_dim=1) # 从第一维开始 flatten, (batch, 32*9*9)
        x = self.fc(x) # (batch, 128)

        actions = self.actor(x) # (batch, num_actions)
        value = self.critic(x) # (batch, 1)

        return logits, value
