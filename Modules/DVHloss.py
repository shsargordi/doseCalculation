import torch
import torch.nn as nn
import torch.nn.functional as F

class DVHGlobalLoss(nn.Module):
    def __init__(self, num_bins=500, dose_min=0.0, dose_max=75.0):
        """
        :param num_bins: Number of bins for the DVH curve
        :param dose_min: Minimum dose considered in DVH (e.g., 0 Gy)
        :param dose_max: Maximum dose considered in DVH (e.g., 80 Gy)
        """
        super().__init__()
        self.num_bins = num_bins
        self.register_buffer("dose_bins", torch.linspace(dose_min, dose_max, num_bins))

    def forward(self, d_pred, d_gt, mask):
        """
        :param d_pred: predicted dose tensor, shape [B, 1, H, W, D]
        :param d_gt: ground truth dose tensor, shape [B, 1, H, W, D]
        :param mask: binary mask tensor, shape [B, 1, H, W, D] (possible dose mask Md)
        :return: scalar DVH global loss
        """
        # Flatten spatial dims: [B, 1, H, W, D] -> [B, N]
        B = d_pred.shape[0]
        d_pred_flat = d_pred.view(B, -1)
        d_gt_flat = d_gt.view(B, -1)
        mask_flat = mask.view(B, -1)

        # Only keep masked voxels
        d_pred_masked = d_pred_flat * mask_flat
        d_gt_masked = d_gt_flat * mask_flat

        dvh_pred = self.compute_dvh(d_pred_masked, mask_flat)
        dvh_gt = self.compute_dvh(d_gt_masked, mask_flat)

        # L2 loss between DVHs
        loss = F.mse_loss(dvh_pred, dvh_gt, reduction="mean")
        return loss

    def compute_dvh(self, dose, mask):
        """
        Computes the cumulative DVH over volume (masked).
        :param dose: [B, N] dose values (already masked)
        :param mask: [B, N] binary mask
        :return: [B, num_bins] cumulative DVH
        """
        B = dose.shape[0]
        dvh = []

        for t in self.dose_bins:  # t = threshold
            # DVH[v]: fraction of voxels >= t
            above_thresh = (dose >= t).float() * mask
            frac = above_thresh.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
            dvh.append(frac)

        return torch.stack(dvh, dim=1)  # [B, num_bins]

class CombinedDoseLoss(nn.Module):
    def __init__(self, alpha=10.0, beta=10.0, **dvh_kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dvh_loss = DVHGlobalLoss(**dvh_kwargs)
        self.mse_loss = nn.MSELoss()

    def forward(self, d_pred, d_gt, mask):
        """
        :param d_pred: predicted dose tensor [B, 1, H, W, D]
        :param d_gt: ground truth dose tensor [B, 1, H, W, D]
        :param mask: possible dose mask [B, 1, H, W, D]
        """

        lmse = self.mse_loss(d_pred * mask, d_gt * mask)
        ldvh = self.dvh_loss(d_pred, d_gt, mask)
        return self.alpha * lmse + self.beta * ldvh