"""
Quality Loss for Stage 3
Pushes SNN outputs toward HQ prompt and away from LQ prompt
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityLoss(nn.Module):
    """
    Quality-guided loss from SpikeCLIP paper (Eq. 8):

    L_prompt = -log(exp(sim(I, T_hq)) / (exp(sim(I, T_hq)) + exp(sim(I, T_lq))))

    This is softmax cross-entropy pushing images toward T_hq
    """

    def __init__(self, temperature=0.07):
        """
        :param temperature: Softmax temperature for similarity scaling
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features_hq, text_features_lq):
        """
        Compute quality loss (Paper Eq. 8).

        :param image_features: (B, 512) normalized image features from CLIP
        :param text_features_hq: (512,) normalized HQ prompt features
        :param text_features_lq: (512,) normalized LQ prompt features
        :return: quality loss (scalar)
        """
        # Compute similarities
        sim_hq = (image_features @ text_features_hq) / self.temperature  # (B,)
        sim_lq = (image_features @ text_features_lq) / self.temperature  # (B,)

        # Stack for softmax: (B, 2) where [:, 0]=LQ, [:, 1]=HQ
        logits = torch.stack([sim_lq, sim_hq], dim=1)

        # Target: all images should be HQ (label=1)
        labels = torch.ones(len(logits), dtype=torch.long, device=logits.device)

        # Cross-entropy loss (paper Eq. 8)
        loss = F.cross_entropy(logits, labels)

        return loss

class CombinedStage3Loss(nn.Module):
    """
    Combined loss for Stage 3:
    L_total = λ_quality * L_quality + λ_recon * L_recon + λ_infonce * L_infonce

    - L_quality: Push toward HQ prompt
    - L_recon: MSE reconstruction loss (preserve content)
    - L_infonce: Original contrastive loss (semantic alignment)
    """

    def __init__(
            self,
            lambda_quality=1.0,
            lambda_recon=0.5,
            lambda_infonce=0.5,
            temperature=0.07
    ):
        super().__init__()
        self.lambda_quality = lambda_quality
        self.lambda_recon = lambda_recon
        self.lambda_infonce = lambda_infonce
        self.temperature = temperature

        self.quality_loss = QualityLoss(temperature)
        self.recon_loss = nn.MSELoss()

    def forward(
            self,
            snn_output,
            gt_image,
            image_features,
            text_features_hq,
            text_features_lq,
            text_features_class=None
    ):
        """
        Compute combined Stage 3 loss.

        :param snn_output: (B, 1, H, W) SNN reconstruction
        :param gt_image: (B, 1, H, W) Ground truth image
        :param image_features: (B, 512) CLIP features of SNN output
        :param text_features_hq: (512,) HQ prompt features
        :param text_features_lq: (512,) LQ prompt features
        :param text_features_class: (B, 512) Class text features (optional, for InfoNCE)
        :return: total loss, dict of individual losses
        """
        losses = {}

        # 1. Quality Loss
        loss_quality = self.quality_loss(image_features, text_features_hq, text_features_lq)
        losses['quality'] = loss_quality.item()

        # 2. Reconstruction Loss (resize if needed)
        if snn_output.shape != gt_image.shape:
            gt_resized = F.interpolate(gt_image, size=snn_output.shape[-2:], mode='bilinear', align_corners=False)
        else:
            gt_resized = gt_image
        loss_recon = self.recon_loss(snn_output, gt_resized)
        losses['recon'] = loss_recon.item()

        # 3. InfoNCE Loss (optional)
        loss_infonce = torch.tensor(0.0, device=snn_output.device)
        if text_features_class is not None:
            # Contrastive loss between image and class text
            logits = (image_features @ text_features_class.T) / self.temperature
            labels = torch.arange(len(logits), device=logits.device)
            loss_infonce = F.cross_entropy(logits, labels)
            losses['infonce'] = loss_infonce.item()

        # Combined loss
        total_loss = (
                self.lambda_quality * loss_quality +
                self.lambda_recon * loss_recon +
                self.lambda_infonce * loss_infonce
        )
        losses['total'] = total_loss.item()

        return total_loss, losses
