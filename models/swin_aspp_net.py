import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.blocks import DoubleConv, OutConv, ASPP

class SwinASPPNet(nn.Module):
    def __init__(self, n_classes=151, img_size=224, backbone_name='swin_tiny_patch4_window7_224', pretrained=True):
        """
        SwinASPPNet with Multi-Level Fusion (Based on User Diagram).
        
        Architecture Flow:
        1. Inputs: 
           - c1: Stage 1 (Stride 4)  - Details
           - c3: Stage 3 (Stride 16) - Mid-level Context
           - c4: Stage 4 (Stride 32) - Global Semantics
           
        2. Processing:
           - c3 -> ASPP
           - c4 -> 1x1 Conv (Project) -> Upsample
           
        3. Fusion 1 (UP2 in diagram context, logical high-level fusion):
           - Concat(ASPP_Out, Projected_c4_Up) -> DoubleConv
           
        4. Fusion 2 (UP1 in diagram context, logical low-level fusion):
           - Fusion1_Out -> Upsample
           - c1 -> 1x1 Conv (Project)
           - Concat(Fusion1_Up, Projected_c1) -> DoubleConv
           
        5. Output -> Upsample x4
        """
        super(SwinASPPNet, self).__init__()
        
        # -----------------------------------------------------------------
        # 1. Backbone: Swin Transformer
        # -----------------------------------------------------------------
        # out_indices=(0, 2, 3) corresponds to:
        # Index 0: Stage 1 (Stride 4)
        # Index 2: Stage 3 (Stride 16)
        # Index 3: Stage 4 (Stride 32)
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 2, 3), 
            img_size=img_size,
            strict_img_size=False
        )
        
        # Determine embedding dimension
        if 'tiny' in backbone_name:
            embed_dim = 96
        elif 'small' in backbone_name:
            embed_dim = 96
        elif 'base' in backbone_name:
            embed_dim = 128
        elif 'large' in backbone_name:
            embed_dim = 192
        else:
            embed_dim = 96 

        # Channel calculation based on Swin architecture [C, 2C, 4C, 8C]
        # c1 (idx 0) = C
        # c3 (idx 2) = 4C
        # c4 (idx 3) = 8C
        c1_channels = embed_dim
        c3_channels = embed_dim * 4
        c4_channels = embed_dim * 8
        
        # -----------------------------------------------------------------
        # 2. Middle Flow (ASPP on Stage 3)
        # -----------------------------------------------------------------
        # Diagram: Stage 3 -> ASPP
        self.aspp = ASPP(in_channels=c3_channels, atrous_rates=[6, 12, 18], out_channels=256)
        
        # -----------------------------------------------------------------
        # 3. Deep Fusion Branch (Stage 4)
        # -----------------------------------------------------------------
        # Project Stage 4 to same channels as ASPP output for balanced concatenation
        self.c4_project = nn.Sequential(
            nn.Conv2d(c4_channels, 256, 1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ReLU(inplace=True)
        )
        
        # Fusion 1 Block (Combines ASPP(c3) + c4)
        # Input: 256 (ASPP) + 256 (c4) = 512
        self.fusion_deep = DoubleConv(in_channels=512, out_channels=256, mid_channels=256)
        
        # -----------------------------------------------------------------
        # 4. Shallow Fusion Branch (Stage 1)
        # -----------------------------------------------------------------
        # Project Stage 1 to reduce computation (Standard DeepLab practice)
        self.c1_project = nn.Sequential(
            nn.Conv2d(c1_channels, 48, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=48),
            nn.ReLU(inplace=True)
        )
        
        # Fusion 2 Block (Combines Deep_Fusion + c1)
        # Input: 256 (Deep Fusion) + 48 (c1) = 304
        self.fusion_final = DoubleConv(in_channels=304, out_channels=256, mid_channels=256)
        
        # -----------------------------------------------------------------
        # 5. Output Head
        # -----------------------------------------------------------------
        self.outc = OutConv(256, n_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        
        # --- 1. Encoder ---
        features_raw = self.backbone(x)
        
        # Fix Permutation (N, H, W, C) -> (N, C, H, W)
        features = []
        for f in features_raw:
            if len(f.shape) == 4 and f.shape[1] != f.shape[-1]: 
               f = f.permute(0, 3, 1, 2).contiguous()
            features.append(f)
            
        c1 = features[0] # Stride 4
        c3 = features[1] # Stride 16 (Note: out_indices is (0,2,3), so index 1 in list is stage 3)
        c4 = features[2] # Stride 32
        
        # --- 2. Process Deep Layers ---
        
        # Path A: Stage 3 -> ASPP
        x_aspp = self.aspp(c3) # (N, 256, H/16, W/16)
        
        # Path B: Stage 4 -> Project -> Upsample
        x_c4 = self.c4_project(c4) # (N, 256, H/32, W/32)
        x_c4_up = F.interpolate(x_c4, size=x_aspp.shape[2:], mode='bilinear', align_corners=True) # (H/16, W/16)
        
        # --- 3. First Fusion (Deep + Mid) ---
        # Combine Global (c4) with Context (ASPP)
        x_deep_cat = torch.cat([x_aspp, x_c4_up], dim=1) # 256+256 = 512 channels
        x_deep_fused = self.fusion_deep(x_deep_cat)      # Output: 256 channels
        
        # --- 4. Process Shallow Layers ---
        
        # Upsample Deep Fused features to match c1
        x_up_mid = F.interpolate(x_deep_fused, size=c1.shape[2:], mode='bilinear', align_corners=True) # (H/4, W/4)
        
        # Project c1
        x_c1_low = self.c1_project(c1) # (N, 48, H/4, W/4)
        
        # --- 5. Final Fusion ---
        x_final_cat = torch.cat([x_up_mid, x_c1_low], dim=1) # 256+48 = 304 channels
        x_decode = self.fusion_final(x_final_cat)          # Output: 256 channels
        
        # --- 6. Prediction ---
        logits = self.outc(x_decode)
        logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)
        
        return logits