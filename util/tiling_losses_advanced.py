"""
tiling_losses_advanced.py
Advanced losses for tiling/seamless texture preservation.

Scientific improvements:
1. SeamlessLoss - explicit wrap-around continuity loss
2. LatentPeriodLoss - work directly in latent space (faster, no VAE decode)
3. CrossPatchConsistencyLoss - enforce consistent period across patches
4. GradientPeriodLoss - periodicity in gradient domain (often cleaner)
5. MultiScaleSpectralLoss - pyramid analysis for multi-scale tiling
"""
from __future__ import annotations
from typing import Optional, Tuple, List
import torch
import torch.nn.functional as F
from torch import nn

from toolkit.util.texture_helpers import (
    _EPS,
    rgb_to_luma,
    fft_amp,
    radial_profile,
    soft_peak,
    feather,
)


class SeamlessLoss(nn.Module):
    """
    Explicit seamless/wrap-around loss.
    
    Idea: For a tileable texture, opposite edges should match perfectly.
    We compare:
    - Left edge vs Right edge
    - Top edge vs Bottom edge
    - Diagonal corners (top-left vs bottom-right, top-right vs bottom-left)
    - Diagonal strips (roll by diagonal offset to detect internal diagonal seams)
    
    This directly penalizes seams that would appear when tiling in any direction.
    """
    
    def __init__(
        self,
        edge_width: int = 16,  # how many pixels from edge to compare
        use_gradient: bool = True,  # compare gradients instead of raw pixels (more robust)
        multi_scale: bool = True,  # analyze at multiple scales
        check_diagonals: bool = True,  # check diagonal continuity (for stripe artifacts)
        diagonal_weight: float = 0.5,  # weight for diagonal checks relative to axis-aligned
        w: float = 1.0,
    ):
        super().__init__()
        self.edge_width = edge_width
        self.use_gradient = use_gradient
        self.multi_scale = multi_scale
        self.check_diagonals = check_diagonals
        self.diagonal_weight = diagonal_weight
        self.w = w
        
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude. x: Bx1xHxW -> Bx1xHxW"""
        # Pad for same-size output
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(x_pad, self.sobel_x.to(x.device, x.dtype))
        gy = F.conv2d(x_pad, self.sobel_y.to(x.device, x.dtype))
        return (gx ** 2 + gy ** 2 + _EPS).sqrt()
    
    def _edge_loss(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None, k: Optional[int] = None) -> torch.Tensor:
        """
        Compare opposite edges for seamlessness.
        img: BxCxHxW, mask: Bx1xHxW
        """
        B, C, H, W = img.shape
        if k is None:
            k = self.edge_width
        
        # Extract edges
        left = img[..., :k]        # BxCxHxk
        right = img[..., -k:]      # BxCxHxk
        top = img[..., :k, :]      # BxCxkxW
        bottom = img[..., -k:, :]  # BxCxkxW
        
        # For seamless tiling, left should match right, top should match bottom
        # We compare with wrap-around: left edge of tile should match right edge
        loss_lr = F.mse_loss(left, right, reduction='none')
        loss_tb = F.mse_loss(top, bottom, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            mask_left = mask[..., :k]
            mask_right = mask[..., -k:]
            mask_top = mask[..., :k, :]
            mask_bottom = mask[..., -k:, :]
            
            # Weight by mask coverage
            w_lr = (mask_left * mask_right).mean(dim=(2, 3), keepdim=True).clamp_min(_EPS)
            w_tb = (mask_top * mask_bottom).mean(dim=(2, 3), keepdim=True).clamp_min(_EPS)
            
            loss_lr = (loss_lr * mask_left * mask_right).sum() / (w_lr.sum() * C * H * k + _EPS)
            loss_tb = (loss_tb * mask_top * mask_bottom).sum() / (w_tb.sum() * C * W * k + _EPS)
        else:
            loss_lr = loss_lr.mean()
            loss_tb = loss_tb.mean()
        
        return loss_lr + loss_tb
    
    def _diagonal_loss(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None, k: Optional[int] = None) -> torch.Tensor:
        """
        Check diagonal continuity - catches stripe artifacts at 45° and 135°.
        Compares:
        1. Corner regions (top-left vs bottom-right, top-right vs bottom-left)
        2. Diagonal strips via rolling (detects internal diagonal seams)
        """
        B, C, H, W = img.shape
        if k is None:
            k = self.edge_width
        
        # === Corner comparison ===
        # For diagonal tiling, corners should match
        corner_tl = img[..., :k, :k]
        corner_br = img[..., -k:, -k:]
        corner_tr = img[..., :k, -k:]
        corner_bl = img[..., -k:, :k]
        
        if mask is not None:
            m_tl = mask[..., :k, :k]
            m_br = mask[..., -k:, -k:]
            m_tr = mask[..., :k, -k:]
            m_bl = mask[..., -k:, :k]
            
            # Weighted corner losses
            w1 = (m_tl * m_br).mean(dim=(2, 3), keepdim=True).clamp_min(_EPS)
            w2 = (m_tr * m_bl).mean(dim=(2, 3), keepdim=True).clamp_min(_EPS)
            
            loss_diag1 = ((corner_tl - corner_br).pow(2) * m_tl * m_br).sum() / (w1.sum() * C * k * k + _EPS)
            loss_diag2 = ((corner_tr - corner_bl).pow(2) * m_tr * m_bl).sum() / (w2.sum() * C * k * k + _EPS)
        else:
            loss_diag1 = F.mse_loss(corner_tl, corner_br)
            loss_diag2 = F.mse_loss(corner_tr, corner_bl)
        
        # === Diagonal strip check (roll and compare) ===
        # This catches internal diagonal seams that corner check might miss
        shift = k
        loss_strip = img.new_zeros(())
        
        if H > 2 * shift and W > 2 * shift:
            # Roll by (shift, shift) - detects 45° stripes
            rolled_d1 = torch.roll(img, shifts=(shift, shift), dims=(2, 3))
            center_orig = img[..., shift:-shift, shift:-shift]
            center_d1 = rolled_d1[..., shift:-shift, shift:-shift]
            
            # Roll by (shift, -shift) - detects 135° stripes
            rolled_d2 = torch.roll(img, shifts=(shift, -shift), dims=(2, 3))
            center_d2 = rolled_d2[..., shift:-shift, shift:-shift]
            
            if mask is not None:
                m_center = mask[..., shift:-shift, shift:-shift]
                m_rolled_d1 = torch.roll(mask, shifts=(shift, shift), dims=(2, 3))[..., shift:-shift, shift:-shift]
                m_rolled_d2 = torch.roll(mask, shifts=(shift, -shift), dims=(2, 3))[..., shift:-shift, shift:-shift]
                
                w_d1 = (m_center * m_rolled_d1).mean(dim=(2, 3), keepdim=True).clamp_min(_EPS)
                w_d2 = (m_center * m_rolled_d2).mean(dim=(2, 3), keepdim=True).clamp_min(_EPS)
                
                loss_strip_d1 = ((center_orig - center_d1).pow(2) * m_center * m_rolled_d1).sum() / (w_d1.sum() * C * center_orig.shape[-2] * center_orig.shape[-1] + _EPS)
                loss_strip_d2 = ((center_orig - center_d2).pow(2) * m_center * m_rolled_d2).sum() / (w_d2.sum() * C * center_orig.shape[-2] * center_orig.shape[-1] + _EPS)
            else:
                loss_strip_d1 = F.mse_loss(center_orig, center_d1)
                loss_strip_d2 = F.mse_loss(center_orig, center_d2)
            
            loss_strip = 0.5 * (loss_strip_d1 + loss_strip_d2)
        
        return loss_diag1 + loss_diag2 + loss_strip
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred, target: BxCxHxW in [-1, 1] or [0, 1]
        mask: Bx1xHxW
        """
        # Normalize to [0, 1]
        pred_0 = (pred / 2 + 0.5).clamp(0, 1)
        tgt_0 = (target / 2 + 0.5).clamp(0, 1)
        
        # Convert to luma for analysis
        pred_luma = rgb_to_luma(pred_0)
        tgt_luma = rgb_to_luma(tgt_0)
        
        if self.use_gradient:
            pred_luma = self._compute_gradient(pred_luma)
            tgt_luma = self._compute_gradient(tgt_luma)
        
        losses = []
        
        if self.multi_scale:
            # Pyramid analysis
            scales = [1.0, 0.5, 0.25]
            for scale in scales:
                if scale < 1.0:
                    new_h = max(32, int(pred_luma.shape[2] * scale))
                    new_w = max(32, int(pred_luma.shape[3] * scale))
                    p = F.interpolate(pred_luma, (new_h, new_w), mode='bilinear', align_corners=False)
                    t = F.interpolate(tgt_luma, (new_h, new_w), mode='bilinear', align_corners=False)
                    m = F.interpolate(mask, (new_h, new_w), mode='nearest')
                    edge_w = max(4, int(self.edge_width * scale))
                else:
                    p, t, m = pred_luma, tgt_luma, mask
                    edge_w = self.edge_width
                
                # Axis-aligned edge loss
                loss_axis = self._edge_loss(p, m, edge_w)
                
                # Diagonal loss (catches stripe artifacts at 45°/135°)
                if self.check_diagonals:
                    loss_diag = self._diagonal_loss(p, m, edge_w)
                    losses.append(loss_axis + self.diagonal_weight * loss_diag)
                else:
                    losses.append(loss_axis)
        else:
            loss_axis = self._edge_loss(pred_luma, mask)
            if self.check_diagonals:
                loss_diag = self._diagonal_loss(pred_luma, mask)
                losses.append(loss_axis + self.diagonal_weight * loss_diag)
            else:
                losses.append(loss_axis)
        
        return self.w * sum(losses) / len(losses)


class LatentPeriodLoss(nn.Module):
    """
    Period consistency loss directly in latent space.
    
    Advantages:
    - No need to decode VAE (faster, less memory)
    - Works with the actual training signal
    - Periodicity in latent space ≈ periodicity in image space
    
    We analyze the predicted latent vs target latent for period consistency.
    """
    
    def __init__(
        self,
        r_bins: int = 64,  # fewer bins for latent (lower resolution)
        tau: float = 0.08,
        min_bin: int = 2,
        max_bin: Optional[int] = None,
        w: float = 1.0,
    ):
        super().__init__()
        self.r_bins = r_bins
        self.tau = tau
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.w = w
        
        grid = (torch.arange(r_bins).float() + 0.5) / r_bins
        self.register_buffer("grid", grid)
    
    def _get_period(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Extract dominant period from latent. latent: BxCxHxW -> B (period as frequency)
        """
        B, C, H, W = latent.shape
        
        # Analyze each channel and average
        periods = []
        for c in range(min(C, 4)):  # analyze first 4 channels
            x = latent[:, c:c+1]  # Bx1xHxW
            
            # FFT amplitude
            x_norm = x - x.mean(dim=(2, 3), keepdim=True)
            x_norm = x_norm / (x_norm.std(dim=(2, 3), keepdim=True) + _EPS)
            
            F2 = torch.fft.fft2(x_norm.squeeze(1), norm="ortho")
            F2 = torch.fft.fftshift(F2, dim=(-2, -1))
            A = (F2.real ** 2 + F2.imag ** 2 + _EPS).sqrt()
            A = A.unsqueeze(1)
            
            # Radial profile
            max_bin = self.max_bin if self.max_bin is not None else self.r_bins - 1
            S = radial_profile(A, r_bins=self.r_bins, min_bin=self.min_bin, 
                              max_bin=max_bin, log_scale=False)
            S = S / (S.sum(dim=-1, keepdim=True) + _EPS)
            S_log = torch.log(S + _EPS)
            
            _, f_norm = soft_peak(S_log, tau=self.tau, lo=self.min_bin, hi=max_bin)
            periods.append(f_norm)
        
        return torch.stack(periods, dim=1).mean(dim=1)  # B
    
    def forward(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        mask_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred_latent, target_latent: BxCxHxW (latent space)
        mask_latent: Bx1xHxW (optional, in latent resolution)
        """
        # Get dominant periods
        f_pred = self._get_period(pred_latent)
        f_target = self._get_period(target_latent)
        
        # Loss: periods should match
        loss = (torch.log(f_pred + _EPS) - torch.log(f_target + _EPS)).abs().mean()
        
        return self.w * loss


class CrossPatchConsistencyLoss(nn.Module):
    """
    Enforce consistent period across all patches of the image.
    
    Problem: Local losses may allow different patches to have different periods,
    leading to "period drift" across the image.
    
    Solution: Compute period for each patch, then penalize variance.
    """
    
    def __init__(
        self,
        patch_size: int = 128,
        stride: int = 64,
        r_bins: int = 64,
        tau: float = 0.08,
        w: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.r_bins = r_bins
        self.tau = tau
        self.w = w
    
    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: BxCxHxW -> (B*N)xCxPxP"""
        B, C, H, W = x.shape
        u = F.unfold(x, kernel_size=self.patch_size, stride=self.stride)
        N = u.shape[-1]
        patches = u.transpose(1, 2).contiguous().view(B * N, C, self.patch_size, self.patch_size)
        return patches, N
    
    def _get_patch_periods(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get dominant period for each patch.
        img: BxCxHxW, mask: Bx1xHxW
        Returns: periods (B*N,), valid_mask (B*N,)
        """
        # Extract patches
        img_patches, N = self._extract_patches(img)  # (B*N)xCxPxP
        mask_patches, _ = self._extract_patches(mask)  # (B*N)x1xPxP
        
        # Filter by mask coverage
        coverage = mask_patches.flatten(1).mean(dim=1)  # (B*N,)
        valid = coverage > 0.5  # at least 50% coverage
        
        if not valid.any():
            return torch.zeros(1, device=img.device), torch.zeros(1, device=img.device, dtype=torch.bool)
        
        # Get periods for valid patches
        valid_patches = img_patches[valid]  # Mx C x P x P
        valid_masks = mask_patches[valid]  # Mx1xPxP
        
        # Convert to luma
        if valid_patches.shape[1] == 3:
            luma = rgb_to_luma((valid_patches / 2 + 0.5).clamp(0, 1))
        else:
            luma = valid_patches[:, :1]
        
        # FFT and find period
        luma = luma * feather(valid_masks, ksize=5)
        luma = luma - luma.mean(dim=(2, 3), keepdim=True)
        
        F2 = torch.fft.fft2(luma.squeeze(1), norm="ortho")
        F2 = torch.fft.fftshift(F2, dim=(-2, -1))
        A = (F2.real ** 2 + F2.imag ** 2 + _EPS).sqrt().unsqueeze(1)
        
        S = radial_profile(A, r_bins=self.r_bins, min_bin=2, max_bin=self.r_bins - 1, log_scale=False)
        S = S / (S.sum(dim=-1, keepdim=True) + _EPS)
        S_log = torch.log(S + _EPS)
        
        _, periods = soft_peak(S_log, tau=self.tau, lo=2, hi=self.r_bins - 1)
        
        return periods, valid
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Penalize variance in periods across patches.
        """
        # Get periods for prediction
        pred_periods, valid_pred = self._get_patch_periods(pred, mask)
        tgt_periods, valid_tgt = self._get_patch_periods(target, mask)
        
        if pred_periods.numel() < 2 or tgt_periods.numel() < 2:
            # Return zero that preserves gradient graph
            return pred.new_zeros(())
        
        # Variance of log-periods (more stable)
        log_pred = torch.log(pred_periods + _EPS)
        log_tgt = torch.log(tgt_periods + _EPS)
        
        # Target: pred should have same variance as target (or lower)
        var_pred = log_pred.var()
        var_tgt = log_tgt.var()
        
        # Also: mean period should match
        mean_loss = (log_pred.mean() - log_tgt.mean()).abs()
        
        # Variance should not be higher than target
        var_loss = F.relu(var_pred - var_tgt)
        
        return self.w * (mean_loss + var_loss)


class GradientPeriodLoss(nn.Module):
    """
    Analyze periodicity in gradient domain.
    
    Rationale: Gradients often show cleaner periodicity than raw pixels,
    especially for textures with varying brightness but consistent structure.
    
    This is complementary to pixel-space spectral analysis.
    """
    
    def __init__(
        self,
        r_bins: int = 128,
        tau: float = 0.06,
        band_sigma: float = 0.15,
        w: float = 1.0,
    ):
        super().__init__()
        self.r_bins = r_bins
        self.tau = tau
        self.band_sigma = band_sigma
        self.w = w
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        grid = (torch.arange(r_bins).float() + 0.5) / r_bins
        self.register_buffer("grid", grid)
    
    def _compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """x: Bx1xHxW -> Bx1xHxW gradient magnitude"""
        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        gx = F.conv2d(x_pad, self.sobel_x.to(x.device, x.dtype))
        gy = F.conv2d(x_pad, self.sobel_y.to(x.device, x.dtype))
        return (gx ** 2 + gy ** 2 + _EPS).sqrt()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compare periodicity in gradient domain.
        """
        # Normalize and convert to luma
        pred_0 = (pred / 2 + 0.5).clamp(0, 1)
        tgt_0 = (target / 2 + 0.5).clamp(0, 1)
        
        pred_luma = rgb_to_luma(pred_0)
        tgt_luma = rgb_to_luma(tgt_0)
        
        # Compute gradients
        grad_pred = self._compute_gradient_magnitude(pred_luma)
        grad_tgt = self._compute_gradient_magnitude(tgt_luma)
        
        # Apply mask
        m = feather(mask, ksize=7)
        grad_pred = grad_pred * m
        grad_tgt = grad_tgt * m
        
        # Normalize
        grad_pred = grad_pred - grad_pred.mean(dim=(2, 3), keepdim=True)
        grad_tgt = grad_tgt - grad_tgt.mean(dim=(2, 3), keepdim=True)
        
        # FFT
        F_pred = torch.fft.fft2(grad_pred.squeeze(1), norm="ortho")
        F_pred = torch.fft.fftshift(F_pred, dim=(-2, -1))
        A_pred = (F_pred.real ** 2 + F_pred.imag ** 2 + _EPS).sqrt().unsqueeze(1)
        
        F_tgt = torch.fft.fft2(grad_tgt.squeeze(1), norm="ortho")
        F_tgt = torch.fft.fftshift(F_tgt, dim=(-2, -1))
        A_tgt = (F_tgt.real ** 2 + F_tgt.imag ** 2 + _EPS).sqrt().unsqueeze(1)
        
        # Radial profiles
        max_bin = int(0.4 * self.r_bins)
        S_pred = radial_profile(A_pred, r_bins=self.r_bins, min_bin=2, max_bin=max_bin, log_scale=False)
        S_tgt = radial_profile(A_tgt, r_bins=self.r_bins, min_bin=2, max_bin=max_bin, log_scale=False)
        
        S_pred = S_pred / (S_pred.sum(dim=-1, keepdim=True) + _EPS)
        S_tgt = S_tgt / (S_tgt.sum(dim=-1, keepdim=True) + _EPS)
        
        # Log space for comparison
        S_pred_log = torch.log(S_pred + _EPS)
        S_tgt_log = torch.log(S_tgt + _EPS)
        
        # Peak frequency matching
        _, f_pred = soft_peak(S_pred_log, tau=self.tau, lo=2, hi=max_bin)
        _, f_tgt = soft_peak(S_tgt_log, tau=self.tau, lo=2, hi=max_bin)
        
        loss_peak = (torch.log(f_pred + _EPS) - torch.log(f_tgt + _EPS)).abs().mean()
        
        # Band matching
        grid = self.grid.to(pred.device, pred.dtype)
        c = f_tgt.detach().clamp(1e-3, 1 - 1e-3)
        log_grid = torch.log(grid + _EPS)[None, :]
        log_c = torch.log(c + _EPS)[:, None]
        w = torch.exp(-(log_grid - log_c) ** 2 / (2 * self.band_sigma ** 2))
        w = w / (w.sum(dim=-1, keepdim=True) + _EPS)
        
        loss_band = (w * (S_pred_log - S_tgt_log).abs()).sum(dim=-1).mean()
        
        return self.w * (loss_peak + loss_band)


class MultiScaleSpectralLoss(nn.Module):
    """
    Pyramid-based multi-scale spectral analysis.
    
    Idea: Analyze tiling at multiple scales to catch:
    - Large-scale period (overall tile repeat)
    - Medium-scale patterns
    - Fine texture details
    
    This helps when texture has multi-level structure.
    """
    
    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        r_bins: int = 128,
        tau: float = 0.06,
        w: float = 1.0,
    ):
        super().__init__()
        self.scales = scales
        self.r_bins = r_bins
        self.tau = tau
        self.w = w
    
    def _spectral_loss_at_scale(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spectral period loss at given resolution."""
        # FFT amplitude
        A_pred = fft_amp(pred, mask)
        A_tgt = fft_amp(target, mask)
        
        max_bin = int(0.4 * self.r_bins)
        
        S_pred = radial_profile(A_pred, r_bins=self.r_bins, min_bin=2, max_bin=max_bin, log_scale=False)
        S_tgt = radial_profile(A_tgt, r_bins=self.r_bins, min_bin=2, max_bin=max_bin, log_scale=False)
        
        S_pred = S_pred / (S_pred.sum(dim=-1, keepdim=True) + _EPS)
        S_tgt = S_tgt / (S_tgt.sum(dim=-1, keepdim=True) + _EPS)
        
        S_pred_log = torch.log(S_pred + _EPS)
        S_tgt_log = torch.log(S_tgt + _EPS)
        
        _, f_pred = soft_peak(S_pred_log, tau=self.tau, lo=2, hi=max_bin)
        _, f_tgt = soft_peak(S_tgt_log, tau=self.tau, lo=2, hi=max_bin)
        
        return (torch.log(f_pred + _EPS) - torch.log(f_tgt + _EPS)).abs().mean()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Multi-scale spectral analysis."""
        # Normalize
        pred_0 = (pred / 2 + 0.5).clamp(0, 1)
        tgt_0 = (target / 2 + 0.5).clamp(0, 1)
        
        pred_luma = rgb_to_luma(pred_0)
        tgt_luma = rgb_to_luma(tgt_0)
        
        losses = []
        for scale in self.scales:
            if scale < 1.0:
                new_h = max(32, int(pred_luma.shape[2] * scale))
                new_w = max(32, int(pred_luma.shape[3] * scale))
                p = F.interpolate(pred_luma, (new_h, new_w), mode='bilinear', align_corners=False)
                t = F.interpolate(tgt_luma, (new_h, new_w), mode='bilinear', align_corners=False)
                m = F.interpolate(mask, (new_h, new_w), mode='nearest')
            else:
                p, t, m = pred_luma, tgt_luma, mask
            
            loss = self._spectral_loss_at_scale(p, t, m)
            losses.append(loss)
        
        return self.w * sum(losses) / len(losses)


# ============================================================================
# Combined Advanced Tiling Loss
# ============================================================================

class AdvancedTilingLoss(nn.Module):
    """
    Combined advanced tiling loss with all scientific improvements.
    
    Use this as a replacement or addition to LocalTextureRegularityLoss.
    """
    
    def __init__(
        self,
        # Component weights
        w_seamless: float = 1.0,
        w_gradient: float = 0.5,
        w_cross_patch: float = 0.3,
        w_multiscale: float = 0.5,
        
        # Seamless params
        seamless_edge_width: int = 16,
        seamless_use_gradient: bool = True,
        
        # Gradient params
        gradient_r_bins: int = 128,
        
        # Cross-patch params
        cross_patch_size: int = 128,
        cross_patch_stride: int = 64,
        
        # Multi-scale params
        multiscale_scales: List[float] = [1.0, 0.5, 0.25],
    ):
        super().__init__()
        
        self.w_seamless = w_seamless
        self.w_gradient = w_gradient
        self.w_cross_patch = w_cross_patch
        self.w_multiscale = w_multiscale
        
        if w_seamless > 0:
            self.seamless_loss = SeamlessLoss(
                edge_width=seamless_edge_width,
                use_gradient=seamless_use_gradient,
            )
        
        if w_gradient > 0:
            self.gradient_loss = GradientPeriodLoss(
                r_bins=gradient_r_bins,
            )
        
        if w_cross_patch > 0:
            self.cross_patch_loss = CrossPatchConsistencyLoss(
                patch_size=cross_patch_size,
                stride=cross_patch_stride,
            )
        
        if w_multiscale > 0:
            self.multiscale_loss = MultiScaleSpectralLoss(
                scales=multiscale_scales,
            )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute combined advanced tiling loss.
        """
        total_loss = pred.new_zeros(())
        
        if self.w_seamless > 0:
            total_loss = total_loss + self.w_seamless * self.seamless_loss(pred, target, mask)
        
        if self.w_gradient > 0:
            total_loss = total_loss + self.w_gradient * self.gradient_loss(pred, target, mask)
        
        if self.w_cross_patch > 0:
            total_loss = total_loss + self.w_cross_patch * self.cross_patch_loss(pred, target, mask)
        
        if self.w_multiscale > 0:
            total_loss = total_loss + self.w_multiscale * self.multiscale_loss(pred, target, mask)
        
        return total_loss


class GlobalPeriodAnchorLoss(nn.Module):
    """
    Anchors prediction's period to target's GLOBAL period.
    
    Problem: When dataset has varying tile sizes, model learns to generate
    different periods in different parts of the same image (period drift).
    
    Solution: 
    1. Compute ONE global dominant period from the ENTIRE target image
    2. For EACH patch of prediction, compute its local period
    3. Penalize deviation of each prediction patch from target's global period
    
    This forces the model to use the SAME period as target, preventing drift.
    """
    
    def __init__(
        self,
        patch_size: int = 128,
        stride: int = 64,
        r_bins: int = 128,
        tau: float = 0.06,
        min_bin: int = 3,
        w: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.r_bins = r_bins
        self.tau = tau
        self.min_bin = min_bin
        self.w = w
    
    def _get_global_period(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute single dominant period for the ENTIRE image.
        img: BxCxHxW, mask: Bx1xHxW
        Returns: (B,) tensor of periods
        """
        B = img.shape[0]
        
        # Convert to luma
        img_01 = (img / 2 + 0.5).clamp(0, 1)
        if img_01.shape[1] == 3:
            luma = rgb_to_luma(img_01)
        else:
            luma = img_01[:, :1]
        
        # Apply mask
        m = feather(mask, ksize=7)
        luma = luma * m
        luma = luma - luma.mean(dim=(2, 3), keepdim=True)
        
        # FFT on entire image
        F2 = torch.fft.fft2(luma.squeeze(1), norm="ortho")
        F2 = torch.fft.fftshift(F2, dim=(-2, -1))
        A = (F2.real ** 2 + F2.imag ** 2 + _EPS).sqrt().unsqueeze(1)
        
        # Radial profile
        max_bin = int(0.5 * self.r_bins)
        S = radial_profile(A, r_bins=self.r_bins, min_bin=self.min_bin, max_bin=max_bin, log_scale=False)
        S = S / (S.sum(dim=-1, keepdim=True) + _EPS)
        S_log = torch.log(S + _EPS)
        
        # Find dominant period
        _, periods = soft_peak(S_log, tau=self.tau, lo=self.min_bin, hi=max_bin)
        
        return periods  # (B,)
    
    def _extract_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Extract overlapping patches. Returns (B*N, C, P, P), N"""
        B, C, H, W = x.shape
        P = self.patch_size
        S = self.stride
        
        patches = x.unfold(2, P, S).unfold(3, P, S)  # B x C x nH x nW x P x P
        nH, nW = patches.shape[2], patches.shape[3]
        N = nH * nW
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * N, C, P, P)
        return patches, N
    
    def _get_patch_periods(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get period for each patch.
        Returns: periods (B*N,), valid_mask (B*N,), batch_size B
        """
        B = img.shape[0]
        
        # Extract patches
        img_patches, N = self._extract_patches(img)  # (B*N)xCxPxP
        mask_patches, _ = self._extract_patches(mask)  # (B*N)x1xPxP
        
        # Filter by mask coverage
        coverage = mask_patches.flatten(1).mean(dim=1)
        valid = coverage > 0.3
        
        if not valid.any():
            return torch.zeros(B * N, device=img.device), valid, B
        
        # Get luma for valid patches
        valid_patches = img_patches  # Keep all for indexing
        
        img_01 = (valid_patches / 2 + 0.5).clamp(0, 1)
        if img_01.shape[1] == 3:
            luma = rgb_to_luma(img_01)
        else:
            luma = img_01[:, :1]
        
        # Apply mask
        m = feather(mask_patches, ksize=5)
        luma = luma * m
        luma = luma - luma.mean(dim=(2, 3), keepdim=True)
        
        # FFT
        F2 = torch.fft.fft2(luma.squeeze(1), norm="ortho")
        F2 = torch.fft.fftshift(F2, dim=(-2, -1))
        A = (F2.real ** 2 + F2.imag ** 2 + _EPS).sqrt().unsqueeze(1)
        
        # Radial profile (use smaller bins for patches)
        patch_r_bins = min(self.r_bins, self.patch_size // 2)
        max_bin = int(0.5 * patch_r_bins)
        S = radial_profile(A, r_bins=patch_r_bins, min_bin=2, max_bin=max_bin, log_scale=False)
        S = S / (S.sum(dim=-1, keepdim=True) + _EPS)
        S_log = torch.log(S + _EPS)
        
        _, periods = soft_peak(S_log, tau=self.tau, lo=2, hi=max_bin)
        
        # Normalize periods to be comparable (scale by patch vs image ratio)
        # patch_period / patch_size should equal global_period / image_size
        # So: normalized_patch_period = patch_period * (image_size / patch_size)
        
        return periods, valid, B
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Penalize prediction patches that deviate from target's global period.
        """
        B, C, H, W = pred.shape
        
        # Get global period from TARGET (this is the anchor)
        target_global_period = self._get_global_period(target, mask)  # (B,)
        
        # Normalize to [0,1] range based on image size
        # Period in frequency domain: higher = finer detail, lower = coarser
        # We convert to spatial period: image_size / freq_bin
        target_spatial_period = H / (target_global_period + _EPS)  # pixels per tile
        
        # Get periods for each PREDICTION patch
        pred_periods, valid, B_check = self._get_patch_periods(pred, mask)
        
        if not valid.any():
            return pred.new_zeros(())
        
        # Number of patches per image
        P = self.patch_size
        S = self.stride
        nH = (H - P) // S + 1
        nW = (W - P) // S + 1
        N = nH * nW
        
        # Convert patch periods to spatial (pixels per tile in patch)
        pred_spatial = P / (pred_periods + _EPS)
        
        # Scale patch spatial period to image scale
        # If patch sees period of X pixels, the same pattern at image scale is X * (H/P)
        # Actually, we compare in log-space for better stability
        log_pred = torch.log(pred_spatial + _EPS)
        
        # Expand target period to match patches: each patch should match its image's global period
        # target_spatial_period is (B,), we need (B*N,)
        target_expanded = target_spatial_period.unsqueeze(1).expand(B, N).reshape(B * N)
        log_target = torch.log(target_expanded + _EPS)
        
        # Loss: penalize deviation from target period
        # Only count valid patches
        diff = (log_pred - log_target).abs()
        diff_valid = diff[valid]
        
        if diff_valid.numel() == 0:
            return pred.new_zeros(())
        
        # Hard mining: focus on worst patches (biggest deviations)
        k = max(1, int(0.3 * diff_valid.numel()))
        top_k_diff, _ = torch.topk(diff_valid, k)
        
        loss = top_k_diff.mean()
        
        return self.w * loss
