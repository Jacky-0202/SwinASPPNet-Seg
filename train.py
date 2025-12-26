# train.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F  # Required for resizing (interpolation)
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# --- Project Setup ---
# Add the project root directory to sys.path to ensure modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- Imports ---
import config as config
from models.swin_aspp_net import SwinASPPNet 
from utils.logger import CSVLogger
from utils.plot import plot_history
from utils.loss import SegmentationLoss
from utils.metrics import calculate_miou
from utils.dataset import CityscapesDataset 

# --- 3. Setup Functions ---
def get_loaders():
    """Initializes and returns Train and Validation DataLoaders."""
    print(f"📂 Dataset Root: {config.DATASET_ROOT}")
    
    # Initialize Train Dataset
    # behavior: Random scaling and cropping to config.IMG_SIZE (e.g., 512x512 or 768x768)
    train_ds = CityscapesDataset(
        img_dir=config.TRAIN_IMG_DIR,   
        mask_dir=config.TRAIN_MASK_DIR, 
        mode='train',
        img_size=config.IMG_SIZE
    )
    
    # Initialize Val Dataset
    # behavior: returns the original full-resolution image (e.g., 1024x2048)
    val_ds = CityscapesDataset(
        img_dir=config.VAL_IMG_DIR,     
        mask_dir=config.VAL_MASK_DIR,   
        mode='val',
        img_size=config.IMG_SIZE # Not actively used for cropping in val mode
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    # Batch Size set to 1 for Validation to handle potentially large full images
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader

def get_model_components(device):
    """
    Initializes the model, loss, optimizer, and scheduler.
    
    [Feature Added] Differential Learning Rates:
    - Backbone (Swin): Uses 0.1x Learning Rate (e.g., 1e-5) to preserve pretrained knowledge.
    - Decoder/Head: Uses 1.0x Learning Rate (e.g., 1e-4) to learn new task features quickly.
    """
    print(f"🏗️ Building Model: SwinASPPNet ({config.BACKBONE})...")
    
    # Initialize the model
    model = SwinASPPNet(
        n_classes=config.NUM_CLASSES, 
        img_size=config.IMG_SIZE, # Used for Swin Transformer position embedding init
        backbone_name=config.BACKBONE,
        pretrained=True
    ).to(device)
    
    # Loss Function (ignoring the void class, usually 255)
    loss_fn = SegmentationLoss(n_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
    
    # --- Optimizer Setup (Differential Learning Rates) ---
    
    # 1. Identify Backbone Parameters (Swin Transformer)
    backbone_ids = list(map(id, model.backbone.parameters()))
    
    # 2. Identify Head/Decoder Parameters (Everything else: ASPP, Decoder, OutConv)
    base_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    # 3. Create Parameter Groups
    optimizer = optim.AdamW([
        # Group 1: Decoder/Head -> Use full LR from config (e.g., 1e-4)
        {'params': base_params, 'lr': config.LEARNING_RATE},
        
        # Group 2: Backbone -> Use 10% of LR (e.g., 1e-5) to avoid destroying pretrained weights
        {'params': model.backbone.parameters(), 'lr': config.LEARNING_RATE * 0.1}
    ], weight_decay=1e-2)
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler()
    
    # Learning Rate Scheduler (Cosine Annealing with Warm Restarts)
    # Note: The scheduler automatically scales the LR for each group based on their initial values.
    scheduler = CosineAnnealingWarmRestarts(optimizer,
                                            T_0=config.SCHEDULER_T0,
                                            T_mult=config.SCHEDULER_T_MULT,
                                            eta_min=config.SCHEDULER_ETA_MIN)
    
    return model, loss_fn, optimizer, scaler, scheduler

# --- 4. Core Loops ---
def run_epoch(loader, model, optimizer, loss_fn, scaler, device, mode='train'):
    """
    Runs one epoch of training or validation.
    
    - Training: Uses Random Crops (handled by Dataset).
    - Validation: Resizes whole image to config.INFER_SIZE, predicts, then resizes back.
    """
    model.train() if mode == 'train' else model.eval()
    loop = tqdm(loader, desc=mode.capitalize(), leave=False)
    
    total_loss = 0.0
    total_miou = 0.0
    
    with torch.set_grad_enabled(mode == 'train'):
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)

            if mode == 'train':
                # --- Training Phase (Random Scale + Crop) ---
                # The data here is already augmented by the Dataset logic
                
                with torch.amp.autocast('cuda'):
                    predictions = model(data)
                    loss = loss_fn(predictions, targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            else:
                # --- Validation Phase (Whole Image Resize) ---
                
                # 1. Store original dimensions (e.g., 1024, 2048)
                orig_h, orig_w = data.shape[-2:]
                
                # 2. Downsample input to Inference Size (e.g., 512, 1024)
                # 'bilinear' is used for images. align_corners=True is standard.
                data_resized = F.interpolate(
                    data, 
                    size=config.INFER_SIZE, 
                    mode='bilinear', 
                    align_corners=True
                )
                
                with torch.amp.autocast('cuda'):
                    # 3. Model Inference on resized image
                    logits = model(data_resized)
                    
                    # 4. Upsample result back to Original Size
                    # We interpolate the logits to match the Ground Truth size
                    predictions = F.interpolate(
                        logits, 
                        size=(orig_h, orig_w), 
                        mode='bilinear', 
                        align_corners=True
                    )
                    
                    # Calculate loss against original targets
                    loss = loss_fn(predictions, targets)

            # --- Metrics Calculation ---
            preds_detached = predictions.detach()
            
            # Calculate mIoU (Mean Intersection over Union)
            miou_val = calculate_miou(
                preds_detached, 
                targets, 
                n_classes=config.NUM_CLASSES, 
                ignore_index=config.IGNORE_INDEX
            ).item()
            
            loss_val = loss.item()
            
            total_loss += loss_val
            total_miou += miou_val
            
            # Update progress bar
            loop.set_postfix(loss=f"{loss_val:.4f}", miou=f"{miou_val:.4f}")
            
    # Return average loss and mIoU for the epoch
    return total_loss / len(loader), 0.0, total_miou / len(loader)

# --- 5. Main Execution ---
def main():
    print(f"--- SwinASPPNet Training Setup ---")
    print(f"--- Device:     {config.DEVICE}")
    print(f"--- Backbone:   {config.BACKBONE}")
    print(f"--- Train Size: {config.IMG_SIZE}x{config.IMG_SIZE} (Augmented)")
    print(f"--- Infer Size: {config.INFER_SIZE} (Resize)")
    
    # Create directories for saving models and logs
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Initialize Logger
    logger = CSVLogger(save_dir=config.SAVE_DIR, filename='training_log.csv')
    
    # Get DataLoaders
    train_loader, val_loader = get_loaders()
    
    # Get Model components
    model, loss_fn, optimizer, scaler, scheduler = get_model_components(config.DEVICE)

    best_miou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': []}

    print(f"\n🚀 Starting Training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        # Log the learning rate of the first group (Decoder)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] LR (Head): {current_lr:.6f}")

        # Run Training Loop
        train_loss, _, train_miou = run_epoch(
            train_loader, model, optimizer, loss_fn, scaler, config.DEVICE, mode='train'
        )
        
        # Run Validation Loop
        val_loss, _, val_miou = run_epoch(
            val_loader, model, None, loss_fn, None, config.DEVICE, mode='val'
        )
        
        # Step the Scheduler
        scheduler.step()
        
        # Log metrics to CSV
        logger.log([epoch+1, current_lr, train_loss, 0, train_miou, val_loss, 0, val_miou])
        
        # Print metrics
        print(f"\tTrain Loss: {train_loss:.4f} | mIoU: {train_miou:.4f}")
        print(f"\tVal Loss:   {val_loss:.4f} | mIoU: {val_miou:.4f}")
        
        # Store history for plotting
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou); history['val_miou'].append(val_miou)

        # Save Best Model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"💾 Best Model Saved! ({best_miou:.4f})")
            
        # Save Last Model (Checkpoint)
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)

    print("\n🎉 Training Complete!")
    
    # Plot training history curves
    plot_history(
        history['train_loss'], history['val_loss'], 
        [], [], # Dice scores (empty for now)
        history['train_miou'], history['val_miou'], 
        save_dir=config.SAVE_DIR
    )

if __name__ == "__main__":
    main()