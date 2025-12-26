# utils/plot.py

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def to_cpu_numpy(data):
    """
    Helper function to convert tensors or lists of tensors to CPU numpy arrays.
    Handles various input types safely.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return np.array([x.detach().cpu().numpy() for x in data])
        else:
            return np.array(data)
    return np.array(data)

def plot_history(train_losses, val_losses, train_dice, val_dice, train_miou, val_miou, save_dir):
    """
    Plots Training and Validation curves for Loss, Dice, and mIoU.
    
    Includes safety checks to skip plotting empty metrics (e.g., if Dice was not calculated).
    """
    # 1. Convert all inputs to numpy arrays
    train_losses = to_cpu_numpy(train_losses)
    val_losses = to_cpu_numpy(val_losses)
    train_dice = to_cpu_numpy(train_dice)
    val_dice = to_cpu_numpy(val_dice)
    train_miou = to_cpu_numpy(train_miou)
    val_miou = to_cpu_numpy(val_miou)
    
    # 2. Define X-axis (Epochs) based on the length of training loss
    epochs = range(1, len(train_losses) + 1)

    # 3. Setup Figure (Wide format for 3 subplots)
    plt.figure(figsize=(18, 5))

    # --- Subplot 1: Loss Curve ---
    plt.subplot(1, 3, 1)
    if len(train_losses) > 0:
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No Loss Data', ha='center', va='center')

    # --- Subplot 2: Dice Score Curve ---
    # SAFETY CHECK: Only plot if data is available to prevent ValueError
    plt.subplot(1, 3, 2)
    if len(train_dice) > 0 and len(val_dice) > 0:
        plt.plot(epochs, train_dice, 'b-', label='Train Dice')
        plt.plot(epochs, val_dice, 'g-', label='Val Dice')
        plt.title('Dice Score Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.legend()
        plt.grid(True)
    else:
        # Display N/A message if data is empty
        plt.text(0.5, 0.5, 'Dice Score N/A\n(Not Calculated)', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.title('Dice Score (N/A)')
        plt.grid(False)
    
    # --- Subplot 3: mIoU Score Curve ---
    # SAFETY CHECK: Only plot if data is available
    plt.subplot(1, 3, 3)
    if len(train_miou) > 0 and len(val_miou) > 0:
        plt.plot(epochs, train_miou, 'b-', label='Train mIoU')
        plt.plot(epochs, val_miou, 'm-', label='Val mIoU')
        plt.title('mIoU Score Curve')
        plt.xlabel('Epochs')
        plt.ylabel('mIoU')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'mIoU N/A', 
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)
        plt.title('mIoU Score (N/A)')
        plt.grid(False)

    # 4. Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"📊 Training curves saved at: {save_path}")