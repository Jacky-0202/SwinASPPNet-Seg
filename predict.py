# predict.py

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

# --- Project Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config as config
from models.swin_aspp_net import SwinASPPNet 

# --- 1. Settings ---
INPUT_DIR = 'test_data'         # Directory for input test images
OUTPUT_DIR = 'test_results'     # Directory for output results

# Automatically construct model path
# MODEL_PATH = os.path.join(config.SAVE_DIR, 'best_model.pth')
MODEL_PATH = "/home/tec/Desktop/Project/SwinASPPNet-Seg/checkpoints/Swin_Large_384_Cityscapes/best_model.pth"

# [MODIFIED] Manually set Inference Size (Height, Width)
# We strictly use 512x1024 regardless of config settings.
# Format: (Height, Width) -> matches PyTorch F.interpolate convention
# INFER_SIZE = config.INFER_SIZE
INFER_SIZE = (512, 1024)

# --- 2. Cityscapes Palette Generator (19 Classes) ---
def get_cityscapes_palette():
    """
    Cityscapes standard 19-class color definition (RGB).
    """
    palette = [
        [128, 64, 128],   # 0: Road
        [244, 35, 232],   # 1: Sidewalk
        [70, 70, 70],     # 2: Building
        [102, 102, 156],  # 3: Wall
        [190, 153, 153],  # 4: Fence
        [153, 153, 153],  # 5: Pole
        [250, 170, 30],   # 6: Traffic Light
        [220, 220, 0],    # 7: Traffic Sign
        [107, 142, 35],   # 8: Vegetation
        [152, 251, 152],  # 9: Terrain
        [70, 130, 180],   # 10: Sky
        [220, 20, 60],    # 11: Person
        [255, 0, 0],      # 12: Rider
        [0, 0, 142],      # 13: Car
        [0, 0, 70],       # 14: Truck
        [0, 60, 100],     # 15: Bus
        [0, 80, 100],     # 16: Train
        [0, 0, 230],      # 17: Motorcycle
        [119, 11, 32]     # 18: Bicycle
    ]
    return np.array(palette, dtype=np.uint8)

CITY_PALETTE = get_cityscapes_palette()

# --- 3. Resize Inference Function ---
def process_image(img_path, model, device, transform):
    """
    Pipeline: 
    1. Load Image (Original Size)
    2. Resize to INFER_SIZE (512x1024)
    3. Predict
    4. Upsample Logits back to Original Size
    """
    # 1. Load original image
    original_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = original_img.size # Record original size (PIL uses W, H)
    
    # 2. Preprocessing (ToTensor + Normalize)
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # 3. Resize input to Inference Size (512, 1024)
    # 'align_corners=True' is standard for image interpolation
    input_resized = F.interpolate(input_tensor, size=INFER_SIZE, mode='bilinear', align_corners=True)
    
    # 4. Model Inference
    with torch.no_grad():
        logits = model(input_resized)
    
    # 5. Upsample logits back to Original Size
    # Note: We resize logits BEFORE argmax to preserve precision
    output = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    
    # 6. Get final Mask (Argmax)
    pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
    return pred_mask, original_img

def save_blend_result(pred_mask, original_img_pil, save_path, alpha=0.9):
    """
    Colorize the prediction mask and blend it with the original image.
    """
    mask_copy = pred_mask.copy()
    # Prevent index out of bounds error
    mask_copy[mask_copy >= len(CITY_PALETTE)] = 0 
    
    color_mask = CITY_PALETTE[mask_copy]
    orig_np = np.array(original_img_pil)
    
    # Safety check: Ensure dimensions match
    if orig_np.shape[:2] != color_mask.shape[:2]:
        color_mask = cv2.resize(color_mask, (orig_np.shape[1], orig_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Blend images
    blend = (orig_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    
    # Save result (OpenCV uses BGR)
    cv2.imwrite(save_path, cv2.cvtColor(blend, cv2.COLOR_RGB2BGR))

def main():
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"📁 Created '{INPUT_DIR}'. Please put test images here.")
        return

    device = config.DEVICE
    print(f"🚀 Loading Model from: {MODEL_PATH}")
    print(f"   Backbone: {config.BACKBONE}")
    print(f"   Inference Size: {INFER_SIZE} (Manually Set)")

    # Initialize Model
    # Note: img_size here is for initialization only; it doesn't restrict input size due to strict_img_size=False
    model = SwinASPPNet(
        n_classes=config.NUM_CLASSES, 
        img_size=config.IMG_SIZE, 
        backbone_name=config.BACKBONE, 
        pretrained=False
    ).to(device)
    
    # Load Weights
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            # Handle potential 'model_state_dict' key
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("✅ Model weights loaded.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return
    else:
        print(f"❌ Weight file not found: {MODEL_PATH}")
        return

    # Transforms (No resizing here, handled dynamically in process_image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get list of images
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_ext)]
    
    if not image_files:
        print(f"⚠️ No images in '{INPUT_DIR}'")
        return

    print(f"📂 Found {len(image_files)} images. Processing...")
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, img_name)
        save_name = os.path.splitext(img_name)[0] + "_vis.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        try:
            mask, orig_pil = process_image(img_path, model, device, transform)
            save_blend_result(mask, orig_pil, save_path)
        except Exception as e:
            print(f"⚠️ Error on {img_name}: {e}")
            # Print traceback for debugging (especially for dimension mismatches)
            import traceback
            traceback.print_exc()

    print(f"\n✅ Done! Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()