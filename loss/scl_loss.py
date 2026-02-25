import logging
import torch
import torch.nn as nn

from utils import logger
from utils.model_utils import is_main_process

class SynergisticContrastiveLoss(nn.Module):
    """
    A collaborative contrastive loss function that integrates two strategies:
    1. Global MMC Loss (Base Stabilizer Loss): Performs contrastive learning across the entire batch to learn macro-level feature separation.
    2. Rank Loss: Specifically targets defective samples to forcibly enlarge the distance between positive samples and the hardest negative samples.

    L_scl = L_mmc + gamma * L_rank

    Args:
        temperature (float): The temperature coefficient used in the base MMC loss.
        margin (float): The desired minimum gap in the hard negative margin loss (Rank Loss).
        gamma (float): The weight coefficient used to balance the Rank Loss.
    """
    def __init__(self, temperature: float = 0.1, margin: float = 0.3, gamma: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.gamma = gamma
        logger = logging.getLogger("SFR-Net.train")
        if is_main_process():
            logger.info(f"SynergisticContrastiveLoss initialized with: T={self.temperature}, margin={self.margin}, gamma={self.gamma}")

    def _base_mmc_loss(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the MMC loss by flattening the entire batch to involve B * C - 1 negative samples per anchor.
        """
        mask = mask.float()
        logits, mask = logits.reshape(-1), mask.reshape(-1)

        logits = logits / self.temperature
        
        logits_max = torch.max(logits)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        log_denominator = torch.log(exp_logits.sum())
        
        log_prob = logits - log_denominator
        
        num_pos_pairs = mask.sum()
        if num_pos_pairs < 1e-6:
            return torch.tensor(0.0, device=logits.device)
            
        mean_log_prob_pos = (mask * log_prob).sum() / num_pos_pairs
        
        loss = -mean_log_prob_pos
        return loss

    def _hard_margin_loss(self, logits_abnormal: torch.Tensor, mask_abnormal: torch.Tensor) -> torch.Tensor:
        mask_abnormal = mask_abnormal.float()
        
        # --- Step 1: Identify the score of the hardest negative sample for each instance (row) ---
        # To avoid affecting original logits, create a copy and isolate negative sample logits.
        neg_logits = torch.where(mask_abnormal == 0, logits_abnormal, torch.tensor(-1e9, device=logits_abnormal.device))
        # hard_neg_per_sample shape: [B_abnormal, 1]
        hard_neg_per_sample, _ = neg_logits.max(dim=1, keepdim=True)

        # --- Step 2: Compute potential losses for all positions ---
        # hard_neg_per_sample is broadcasted to [B_abnormal, C].
        # loss_matrix shape: [B_abnormal, C]
        loss_matrix = self.margin + hard_neg_per_sample - logits_abnormal

        # --- Step 3: Apply ReLU to retain only positive loss values ---
        loss_matrix = torch.clamp(loss_matrix, min=0)

        # --- Step 4: Mask and aggregate losses specifically at positive sample positions ---
        masked_loss = loss_matrix * mask_abnormal

        # --- Step 5: Compute the average loss ---
        num_pos_pairs = mask_abnormal.sum()
        if num_pos_pairs < 1e-6:
            return torch.tensor(0.0, device=logits_abnormal.device)
        
        total_hard_loss = masked_loss.sum() / num_pos_pairs
        return total_hard_loss

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss_base = self._base_mmc_loss(logits, mask)

        if torch.isnan(loss_base):
            print("FATAL: Base MMC Loss is NaN! Check learning rate or model stability.")
            return loss_base

        is_abnormal_mask = (mask.sum(dim=1) > 0)
        loss_hard = torch.tensor(0.0, device=logits.device)
        if is_abnormal_mask.any():
            logits_abnormal = logits[is_abnormal_mask]
            mask_abnormal = mask[is_abnormal_mask]
            loss_hard = self._hard_margin_loss(logits_abnormal, mask_abnormal)

        total_loss = loss_base + self.gamma * loss_hard
        return total_loss