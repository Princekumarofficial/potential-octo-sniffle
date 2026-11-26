from torch import nn
import torch

class BigMoveAwareLoss(nn.Module):
    """
    Directional loss function that heavily penalizes sign mismatches (wrong direction)
    and penalizes missing large tail-risk market moves (moves >= sigma_threshold).
    """
    def __init__(self, dir_penalty=0.5, mag_penalty=2.0, sigma_threshold=1.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.dir_penalty = dir_penalty
        self.mag_penalty = mag_penalty
        self.sigma_threshold = sigma_threshold

    def forward(self, y_pred, y_true):
        sq_error = self.mse(y_pred, y_true)

        weight = 1.0 + self.mag_penalty * torch.abs(y_true)
        big_move_mask = (torch.abs(y_true) >= self.sigma_threshold).float()
        weight = weight + (big_move_mask * self.mag_penalty * 2.0)

        weighted_mse = sq_error * weight

        sign_mismatch = torch.relu(-torch.sign(y_pred) * torch.sign(y_true))
        dir_weight = torch.abs(y_true) + (big_move_mask * 2.0)
        dir_loss = sign_mismatch * dir_weight * self.dir_penalty

        return torch.mean(weighted_mse + dir_loss)