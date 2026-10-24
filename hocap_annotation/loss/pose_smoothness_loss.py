import torch
import torch.nn as nn


class PoseSmoothnessLoss(nn.Module):
    def __init__(self, win_size=1, w_vel_r=1.0, w_vel_t=1.0, w_acc_r=1.0, w_acc_t=1.0):
        super(PoseSmoothnessLoss, self).__init__()
        self._win_size = win_size
        self._w_vel_r = w_vel_r
        self._w_vel_t = w_vel_t
        self._w_acc_r = w_acc_r
        self._w_acc_t = w_acc_t
        self.register_buffer("_zero", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, poses, subset=None, valid_mask=None):
        """
        Args:
            poses: list of torch.Tensor, each tensor has shape (batch_size, num_objs, pose_dim)
            subset: list of int, the indices of the selected poses
        """

        if subset is None:
            subset = list(range(len(poses)))

        P = torch.stack([poses[i] for i in subset], dim=1)
        if valid_mask is None:
            valid = torch.ones(P.shape[:2], dtype=torch.bool, device=P.device)
        else:
            valid = valid_mask[subset].T
            if valid.shape != P.shape[:2]:
                raise ValueError('Expected hand/frame validity matching the poses')

        def penalty(values, mask):
            selected = values[mask].reshape(-1, 3)
            return torch.linalg.vector_norm(selected, dim=-1).mean() if selected.numel() else P.reshape(-1)[:0].sum()

        loss_vel_r = self._zero.clone()
        loss_vel_t = self._zero.clone()
        loss_acc_r = self._zero.clone()
        loss_acc_t = self._zero.clone()

        for j in range(1, self._win_size + 1):
            if len(P) <= j:
                continue
            pair_valid = valid[:-j].clone()
            for offset in range(1, j + 1):
                pair_valid &= valid[offset:len(P)-j+offset]
            acc_valid = pair_valid[1:] & pair_valid[:-1]
            vel_forward = P[j:] - P[:-j]
            acc_forward = vel_forward[1:] - vel_forward[:-1]
            loss_vel_r += self._w_vel_r * penalty(vel_forward[..., :-3], pair_valid)
            loss_vel_t += self._w_vel_t * penalty(vel_forward[..., -3:], pair_valid)
            loss_acc_r += self._w_acc_r * penalty(acc_forward[..., :-3], acc_valid)
            loss_acc_t += self._w_acc_t * penalty(acc_forward[..., -3:], acc_valid)

        total_loss = loss_vel_r + loss_vel_t + loss_acc_r + loss_acc_t
        total_loss /= self._win_size
        return total_loss
