# local_texture_regularizer.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from toolkit.util.texture_losses import SpectralPeriodLoss, LogPolarAlignLoss, ACFPeriodLoss
from toolkit.util.phase_correlation_loss import PhaseCorrelationLoss


def _extract_patches_unfold(
    x: torch.Tensor,
    patch: int,
    stride: int,
    return_bidx: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    x: BxCxHxW
    return:
      patches: (B*N) x C x patch x patch
      bidx:    (B*N,)  индекс исходного B для каждого патча (если return_bidx=True)
    """
    B, C, H, W = x.shape
    u = F.unfold(x, kernel_size=patch, stride=stride)  # B x (C*patch*patch) x N
    N = u.shape[-1]
    patches = u.transpose(1, 2).contiguous().view(B * N, C, patch, patch)

    if not return_bidx:
        return patches

    bidx = torch.arange(B, device=x.device).repeat_interleave(N)  # (B*N,)
    return patches, bidx

def _random_offset_align(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    stride: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Случайно сдвигаем "сётку" патчей, не меняя итоговый размер (crop + pad обратно).
    Важно: одинаково для pred/target/mask.
    """
    if stride <= 1:
        return pred, target, mask

    B, _, H, W = pred.shape
    ox = int(torch.randint(0, stride, (1,), device=pred.device).item())
    oy = int(torch.randint(0, stride, (1,), device=pred.device).item())

    if ox == 0 and oy == 0:
        return pred, target, mask

    pred2   = pred[..., ox:, oy:]
    target2 = target[..., ox:, oy:]
    mask2   = mask[..., ox:, oy:]

    # pad обратно до (H,W): паддим справа и снизу
    pad = (0, oy, 0, ox)  # (left,right,top,bottom)
    pred2   = F.pad(pred2,   pad, mode="reflect")
    target2 = F.pad(target2, pad, mode="reflect")
    mask2   = F.pad(mask2,   pad, mode="replicate")  # маску лучше не отражать

    # safety: если reflect вдруг дал +1 пиксель из-за нечётности — обрежем
    pred2   = pred2[..., :H, :W]
    target2 = target2[..., :H, :W]
    mask2   = mask2[..., :H, :W]

    return pred2, target2, mask2


def roll_batch(pred: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor,
               p: float = 1.0,
               max_shift: tuple[int, int] | None = None):
    """
    pred/target: BxCxHxW, mask: Bx1xHxW
    Делает одинаковый циклический сдвиг всем тензорам.
    """
    if p <= 0:
        return pred, target, mask

    if torch.rand((), device=pred.device) > p:
        return pred, target, mask

    H, W = pred.shape[-2:]
    if max_shift is None:
        max_shift = (H, W)

    sy = int(torch.randint(0, max_shift[0], (1,), device=pred.device).item())
    sx = int(torch.randint(0, max_shift[1], (1,), device=pred.device).item())

    pred   = torch.roll(pred,   shifts=(sy, sx), dims=(-2, -1))
    target = torch.roll(target, shifts=(sy, sx), dims=(-2, -1))
    mask   = torch.roll(mask,   shifts=(sy, sx), dims=(-2, -1))
    return pred, target, mask



def extract_masked_patches(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    stride: int,
    min_coverage: float = 0.45,
    random_offset: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pred, target: BxCxHxW
    mask: Bx1xHxW (0..1)
    Return:
      pred_p, tgt_p, mask_p: (M) x C/1 x patch x patch
      cover: (M,)
      patch_bidx: (M,) индекс исходного B для каждого патча
    """
    if random_offset:
        pred, target, mask = _random_offset_align(pred, target, mask, stride=stride)

    pred_p, bidx = _extract_patches_unfold(pred, patch_size, stride, return_bidx=True)
    tgt_p        = _extract_patches_unfold(target, patch_size, stride, return_bidx=False)
    m_p          = _extract_patches_unfold(mask, patch_size, stride, return_bidx=False)

    cover = m_p.flatten(1).mean(dim=1)  # (B*N,)
    keep = cover >= float(min_coverage)

    if keep.any():
        pred_p = pred_p[keep]
        tgt_p  = tgt_p[keep]
        m_p    = m_p[keep]
        cover  = cover[keep]
        bidx   = bidx[keep]
    else:
        pred_p = pred_p[:1]
        tgt_p  = tgt_p[:1]
        m_p    = m_p[:1]
        cover  = cover[:1]
        bidx   = bidx[:1]

    return pred_p, tgt_p, m_p, cover, bidx



class LocalTextureRegularityLoss(nn.Module):
    """
    Локальный регуляризатор тайлинга:
      - режем на патчи в области маски
      - делаем soft-mask внутри патчей (анти-ring)
      - hard-mining: берём худшие top_k патчей (чтобы добить “20% проблем”)
      - считаем спектр/ACF/phase/log и усредняем
    """

    def __init__(
        self,
        patch_size: int = 192,
        stride: int = 96,
        min_coverage: float = 0.45,

        # hard mining
        hard_k_frac: float = 0.30,   # берём худшие 30% патчей
        hard_k_min: int = 4,         # минимум патчей

        # weights
        w_spectral: float = 1.0,
        w_afc: float = 0.4,
        w_phase: float = 0.25,
        w_log: float = 0.15,

        # PhaseCorrelationLoss params
        phase_center_weight: float = 1.0,
        phase_pce_weight: float = 0.05,
        phase_temperature: float = 0.03,
        phase_use_hann: bool = True,
        phase_demean: bool = True,
        phase_exclude_radius: int = 7,

        # SpectralPeriodLoss params (локально делаем более “мягко”)
        spectral_r_bins: int = 128,
        spectral_tau: float = 0.08,
        spectral_band_sigma: float = 0.22,
        spectral_w_peak: float = 0.8,
        spectral_w_band: float = 0.35,
        spectral_min_bin: int = 3,
        spectral_max_bin: int | None = None,

        # ACF params
        afc_r_bins: int = 128,
        afc_tau: float = 0.10,
        afc_min_rel: float = 0.04,
        afc_max_rel: float = 0.55,
        afc_w: float = 0.25,

        # LogPolar params
        log_out_r: int = 96,
        log_out_t: int = 180,
        log_r_min: float = 0.02,
        log_w: float = 0.08,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.min_coverage = min_coverage

        self.hard_k_frac = hard_k_frac
        self.hard_k_min = hard_k_min

        self.phase = PhaseCorrelationLoss(
            center_weight=phase_center_weight,
            pce_weight=phase_pce_weight,
            temperature=phase_temperature,
            use_hann=phase_use_hann,
            demean=phase_demean,
            exclude_radius=phase_exclude_radius,
        )

        self.spectral = SpectralPeriodLoss(
            r_bins=spectral_r_bins,
            tau=spectral_tau,
            band_sigma=spectral_band_sigma,
            w_peak=spectral_w_peak,
            w_band=spectral_w_band,
            min_bin=spectral_min_bin,
            max_bin=spectral_max_bin,
        )

        self.afc = ACFPeriodLoss(
            r_bins=afc_r_bins,
            tau=afc_tau,
            min_rel=afc_min_rel,
            max_rel=afc_max_rel,
            w=afc_w,
        )

        self.log = LogPolarAlignLoss(
            out_r=log_out_r,
            out_t=log_out_t,
            r_min=log_r_min,
            w=log_w,
        )

        self.w_spectral = w_spectral
        self.w_afc = w_afc
        self.w_phase = w_phase
        self.w_log = w_log

    def _soften_mask(self, m: torch.Tensor) -> torch.Tensor:
        # анти-ring: чуть смягчаем границу патча
        m = m.clamp(0, 1)
        m = F.avg_pool2d(m, kernel_size=3, stride=1, padding=1)
        return m.clamp(0, 1)

    def _hard_mine(
        self,
        pred_p: torch.Tensor,
        tgt_p: torch.Tensor,
        m_p: torch.Tensor,
        patch_bidx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hard-mining по seam-прокси (а не по L1).
        Проверяем разрывы во всех направлениях: X, Y и диагонали.
        Это ловит полосы в любом направлении, не только горизонтальные/вертикальные.
        """
        with torch.no_grad():
            k_edge = 12  # 8..16 ок; 12 обычно хорошо

            # === Axis-aligned seams (X/Y) ===
            # веса на краях (среднее маски двух сравниваемых сторон)
            w_lr = 0.5 * (m_p[..., :k_edge] + m_p[..., -k_edge:])
            w_tb = 0.5 * (m_p[..., :k_edge, :] + m_p[..., -k_edge:, :])

            # разрыв по X (левый/правый край патча)
            sx = ((pred_p[..., :k_edge] - pred_p[..., -k_edge:]).abs() * w_lr).mean(dim=(1, 2, 3))
            # разрыв по Y (верх/низ края патча)
            sy = ((pred_p[..., :k_edge, :] - pred_p[..., -k_edge:, :]).abs() * w_tb).mean(dim=(1, 2, 3))

            # === Diagonal seams (45° and 135°) ===
            # Для диагоналей сравниваем углы патча
            # Угол top-left должен соответствовать bottom-right (для диагонального тайлинга)
            corner_tl = pred_p[..., :k_edge, :k_edge]
            corner_br = pred_p[..., -k_edge:, -k_edge:]
            m_tl = m_p[..., :k_edge, :k_edge]
            m_br = m_p[..., -k_edge:, -k_edge:]
            w_diag1 = 0.5 * (m_tl + m_br)
            s_diag1 = ((corner_tl - corner_br).abs() * w_diag1).mean(dim=(1, 2, 3))

            # Угол top-right должен соответствовать bottom-left
            corner_tr = pred_p[..., :k_edge, -k_edge:]
            corner_bl = pred_p[..., -k_edge:, :k_edge]
            m_tr = m_p[..., :k_edge, -k_edge:]
            m_bl = m_p[..., -k_edge:, :k_edge]
            w_diag2 = 0.5 * (m_tr + m_bl)
            s_diag2 = ((corner_tr - corner_bl).abs() * w_diag2).mean(dim=(1, 2, 3))

            # === Diagonal strips (detect internal diagonal seams) ===
            # Сдвигаем патч по диагонали и сравниваем - ловит полосы внутри патча
            H, W = pred_p.shape[-2:]
            shift = k_edge
            
            # Diagonal shift 1: (shift, shift) - detects 45° stripes
            rolled_d1 = torch.roll(pred_p, shifts=(shift, shift), dims=(2, 3))
            m_rolled_d1 = torch.roll(m_p, shifts=(shift, shift), dims=(2, 3))
            # Compare center region (avoid edge wrap artifacts)
            margin = shift
            if H > 2 * margin and W > 2 * margin:
                center_orig = pred_p[..., margin:-margin, margin:-margin]
                center_rolled = rolled_d1[..., margin:-margin, margin:-margin]
                m_center = 0.5 * (m_p[..., margin:-margin, margin:-margin] + 
                                  m_rolled_d1[..., margin:-margin, margin:-margin])
                s_strip_d1 = ((center_orig - center_rolled).abs() * m_center).mean(dim=(1, 2, 3))
            else:
                s_strip_d1 = torch.zeros_like(sx)

            # Diagonal shift 2: (shift, -shift) - detects 135° stripes
            rolled_d2 = torch.roll(pred_p, shifts=(shift, -shift), dims=(2, 3))
            m_rolled_d2 = torch.roll(m_p, shifts=(shift, -shift), dims=(2, 3))
            if H > 2 * margin and W > 2 * margin:
                center_rolled2 = rolled_d2[..., margin:-margin, margin:-margin]
                m_center2 = 0.5 * (m_p[..., margin:-margin, margin:-margin] + 
                                   m_rolled_d2[..., margin:-margin, margin:-margin])
                s_strip_d2 = ((center_orig - center_rolled2).abs() * m_center2).mean(dim=(1, 2, 3))
            else:
                s_strip_d2 = torch.zeros_like(sx)

            # === Combined score ===
            # Axis-aligned + diagonal corners + diagonal strips
            # Diagonal weight slightly lower (0.7) since they overlap with axis checks
            score = sx + sy + 0.7 * (s_diag1 + s_diag2) + 0.5 * (s_strip_d1 + s_strip_d2)

            N = score.numel()
            k = max(int(self.hard_k_frac * N), self.hard_k_min, 1)
            k = min(k, N)
            idx = torch.topk(score, k=k, largest=True).indices

        return pred_p[idx], tgt_p[idx], m_p[idx], patch_bidx[idx]


    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 1) global "центр периода" на ИЗОБРАЖЕНИЕ (убирает локальный дрейф масштаба)
        # get_peak_center уже @torch.no_grad()
        pred, target, mask = roll_batch(pred, target, mask, p=1.0)
        ft_global = self.spectral.get_peak_center(target, mask)  # (B,)

        # 2) патчи (с random offset)
        pred_p, tgt_p, m_p, _cover, patch_bidx = extract_masked_patches(
            pred, target, mask,
            patch_size=self.patch_size,
            stride=self.stride,
            min_coverage=self.min_coverage,
            random_offset=False,
        )

        # 3) soft mask внутри патчей
        m_p = self._soften_mask(m_p)

        # 4) hard-mining по seam-прокси
        pred_p, tgt_p, m_p, patch_bidx = self._hard_mine(pred_p, tgt_p, m_p, patch_bidx)

        # 5) раздаём global ft на каждый патч
        ft = ft_global[patch_bidx]  # (N_patches,)

        # 6) лоссы
        L_spec  = self.spectral(pred_p, tgt_p, m_p, center_override=ft)
        L_afc   = self.afc(pred_p, tgt_p, m_p)
        L_phase = self.phase(pred_p, tgt_p, m_p)
        L_phase = torch.clamp(L_phase, min=0.0)
        L_log   = self.log(pred_p, tgt_p, m_p)

        return (
            self.w_spectral * L_spec +
            self.w_afc      * L_afc +
            self.w_phase    * L_phase +
            self.w_log      * L_log
    )

