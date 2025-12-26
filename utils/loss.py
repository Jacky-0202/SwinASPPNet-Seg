import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=255):
        """
        Focal Loss for addressing class imbalance.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 1. Compute standard Cross Entropy Loss (pixel-wise, no reduction yet)
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index
        )
        
        # 2. Get probabilities (pt) from log-probabilities (ce_loss)
        pt = torch.exp(-ce_loss)
        
        # 3. Compute Focal term: alpha * (1 - pt)^gamma * ce_loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 4. Return mean loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-6, ignore_index=255):
        """
        Dice Loss for optimizing overlap (IoU).
        Handles ignore_index by masking.
        """
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Apply Softmax to get probabilities (N, C, H, W)
        probs = torch.softmax(logits, dim=1)
        
        # --- One-Hot Encoding & Masking ---
        if self.ignore_index is not None:
            # Create mask for valid pixels (1 = valid, 0 = ignore)
            valid_mask = (targets != self.ignore_index).float()
            
            # Clamp targets to avoid error in one_hot (ignored pixels become 0 temporarily)
            targets_clamped = targets.clone()
            targets_clamped[targets == self.ignore_index] = 0
            
            # Convert to One-Hot: (N, H, W) -> (N, H, W, C) -> (N, C, H, W)
            targets_one_hot = F.one_hot(targets_clamped, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
            
            # Apply mask to both probs and targets (Zero out ignored areas)
            # Unsqueeze adds channel dim: (N, 1, H, W)
            mask = valid_mask.unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask
            
        else:
            # Standard One-Hot conversion without masking
            targets_one_hot = F.one_hot(targets, num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        # --- Dice Calculation ---
        # Flatten tensors: (C, N*H*W) to compute Dice per class
        probs_flat = probs.contiguous().view(self.n_classes, -1)
        targets_flat = targets_one_hot.contiguous().view(self.n_classes, -1)
        
        # Intersection & Union
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # Dice Score per class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - Mean Dice
        return 1 - dice.mean()

class SegmentationLoss(nn.Module):
    def __init__(self, n_classes, weight_focal=1.0, weight_dice=1.0, ignore_index=255):
        """
        Combined Loss: Focal Loss + Dice Loss
        Strong strategy for Cityscapes (Class Imbalance + Region Overlap).
        """
        super(SegmentationLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        
        # Initialize sub-losses
        self.focal = FocalLoss(gamma=2.0, ignore_index=ignore_index)
        self.dice = DiceLoss(n_classes=n_classes, ignore_index=ignore_index)

    def forward(self, logits, targets):
        # 1. Calculate Focal Loss (inputs: logits, targets: labels)
        loss_focal = self.focal(logits, targets.long())
        
        # 2. Calculate Dice Loss (inputs: logits, targets: labels)
        loss_dice = self.dice(logits, targets.long())
        
        # 3. Weighted Sum
        return (self.weight_focal * loss_focal) + (self.weight_dice * loss_dice)