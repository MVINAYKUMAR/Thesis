import torch
import torch.nn as nn

class PoseEncoder(nn.Module):
    """
    Simple MLP encoder for pose sequences.
    Input: (B, T, 66)  -> temporal mean -> (B, 66) -> MLP -> (B, D)
    """
    def __init__(self, input_dim=66, hidden_dim=128, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pose_seq):
        # pose_seq: (B, T, 66)
        x = pose_seq.mean(dim=1)  # (B, 66)
        return self.mlp(x)        # (B, out_dim)
