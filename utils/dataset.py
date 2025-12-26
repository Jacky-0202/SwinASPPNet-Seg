# utils/dataset.py

import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F # 👈 新增這行，供 sliding_window 使用

class BinarySegDataset(Dataset):
    def __init__(self, root_dir, img_folder='im', mask_folder='gt', mode='train', img_size=320):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.image_dir = os.path.join(root_dir, img_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"❌ Directories not found. Please check paths:\nImg: {self.image_dir}\nMask: {self.mask_dir}")

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_list = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_ext)])
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize((self.img_size, self.img_size))
        self.resize_mask = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        file_prefix = os.path.splitext(img_name)[0]
        mask_name = file_prefix + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Mask not found for image: {mask_path}")
        
        if self.mode == 'train':
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.randint(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        image = self.resize(image)
        image = TF.to_tensor(image)
        image = self.norm(image)
        mask = self.resize_mask(mask)
        mask = TF.to_tensor(mask)
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        return image, mask

class MultiClassSegDataset(Dataset):
    def __init__(self, root_dir, img_folder='images', mask_folder='annotations', mode='train', img_size=512):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.image_dir = os.path.join(root_dir, img_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"❌ Directories not found:\nImg: {self.image_dir}\nMask: {self.mask_dir}")

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        available_masks = set(os.path.splitext(f)[0] for f in os.listdir(self.mask_dir) if not f.startswith('.'))
        
        self.image_list = []
        for f in sorted(os.listdir(self.image_dir)):
            if f.lower().endswith(valid_ext):
                file_id = os.path.splitext(f)[0]
                if file_id in available_masks:
                    self.image_list.append(f)
        
        print(f"Dataset ({mode}): Found {len(self.image_list)} valid pairs.")
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        file_id = os.path.splitext(img_name)[0]
        mask_name = file_id + '.png'
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path) 
        except Exception as e:
            raise e
        
        if self.mode == 'train':
            if random.random() > 0.3:
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(0.75, 1.33))
                image = TF.resized_crop(image, i, j, h, w, (self.img_size, self.img_size), Image.BILINEAR)
                mask = TF.resized_crop(mask, i, j, h, w, (self.img_size, self.img_size), Image.NEAREST)
            else:
                image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
                mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.2:
                jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                image = jitter(image)
        else:
            image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)

        image = TF.to_tensor(image)
        image = self.norm(image)
        mask_np = np.array(mask)
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]
        mask_tensor = torch.from_numpy(mask_np.copy()).long() 
        return image, mask_tensor

class CityscapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mode='train', img_size=512):
        """
        Cityscapes Dataset with Multi-scale Training Augmentation.
        
        Args:
            img_dir (str): Path to image directory.
            mask_dir (str): Path to mask directory.
            mode (str): 'train' or 'val'.
            img_size (int): The crop size for training (e.g., 512 or 768).
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.img_size = img_size
        
        # Supported image extensions
        valid_ext = ('.png', '.jpg', '.jpeg')
        
        # Load file list
        self.items = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_ext)])
        
        if len(self.items) == 0:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
            
        # Standard ImageNet Normalization
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Define scale range for Multi-scale Training (0.5x to 2.0x)
        # This allows the model to learn objects at various distances.
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # 1. Load Image and Mask
        filename = self.items[idx]
        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) # Mask is usually single channel (L or P mode)

        if self.mode == 'train':
            # --- Training Augmentation Pipeline ---
            
            # A. Random Scale (Multi-scale Training) -------------------------
            # Randomly select a scale factor
            scale = random.choice(self.scales)
            
            # Calculate new dimensions
            w, h = image.size
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize image (Bilinear) and mask (Nearest Neighbor to preserve classes)
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

            # B. Padding (Handle cases where resized image < crop size) ------
            if new_w < self.img_size or new_h < self.img_size:
                pad_w = max(0, self.img_size - new_w)
                pad_h = max(0, self.img_size - new_h)
                
                # Pad Image with black (0) and Mask with Ignore Index (255)
                image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
                mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=255)

            # C. Random Crop -------------------------------------------------
            # Get random crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.img_size, self.img_size)
            )
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # D. Random Horizontal Flip --------------------------------------
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # E. Color Jitter (Image only) -----------------------------------
            # Adjust brightness, contrast, saturation slightly to handle lighting variations
            if random.random() > 0.5:
                image = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)(image)
        
        # --- Validation / Inference Pipeline ---
        # For validation, we typically return the full original image (or handled by train.py resizing)
        # So no cropping/resizing is done here for mode='val' to keep it raw.

        # 2. Convert to Tensor & Normalize
        image = TF.to_tensor(image)
        image = self.norm(image)
        
        # 3. Process Mask (Ensure it's Long Tensor for CrossEntropyLoss)
        mask_np = np.array(mask)
        # Handle cases where mask might have 3 channels (RGB) instead of 1
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0] # Take only the first channel
            
        mask_tensor = torch.from_numpy(mask_np.copy()).long() 
        
        return image, mask_tensor