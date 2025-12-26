# preprocess_cityscapes.py

import os
import shutil
from tqdm import tqdm

# --- Configuration ---
SOURCE_ROOT = '/home/tec/Desktop/Project/Datasets/Cityscapes'
TARGET_ROOT = '/home/tec/Desktop/Project/Datasets/Cityscapes_Flat'

# Define suffixes to strip away
IMG_SUFFIX = '_leftImg8bit.png' 
MASK_SUFFIX = '_gtFine_labelTrainIds.png'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_split(split_name):
    print(f"\n--- Processing split: {split_name} ---")
    
    src_img_dir = os.path.join(SOURCE_ROOT, 'images', split_name)
    src_mask_dir = os.path.join(SOURCE_ROOT, 'gtFine', split_name)
    
    tgt_img_dir = os.path.join(TARGET_ROOT, 'images', split_name)
    tgt_mask_dir = os.path.join(TARGET_ROOT, 'masks', split_name)
    
    create_dir(tgt_img_dir)
    create_dir(tgt_mask_dir)
    
    cities = os.listdir(src_img_dir)
    count = 0
    
    for city in tqdm(cities, desc=f"Processing {split_name} cities"):
        city_img_path = os.path.join(src_img_dir, city)
        city_mask_path = os.path.join(src_mask_dir, city)
        
        if not os.path.isdir(city_img_path):
            continue
            
        files = os.listdir(city_img_path)
        
        for file_name in files:
            if file_name.endswith(IMG_SUFFIX):
                # 1. Get the core ID (e.g., "aachen_000000_000019")
                # simply replace the suffix with empty string
                core_id = file_name.replace(IMG_SUFFIX, '')
                
                # 2. Define simple new name
                simple_name = core_id + '.png'
                
                # 3. Source Paths
                full_img_src = os.path.join(city_img_path, file_name)
                # Construct mask source name
                mask_src_name = core_id + MASK_SUFFIX
                full_mask_src = os.path.join(city_mask_path, mask_src_name)
                
                # 4. Check and Copy (with Renaming)
                if os.path.exists(full_mask_src):
                    # Copy Image -> new_folder/aachen_000000_000019.png
                    shutil.copy2(full_img_src, os.path.join(tgt_img_dir, simple_name))
                    
                    # Copy Mask  -> new_folder/aachen_000000_000019.png
                    # (Identical filename, but in different folder)
                    shutil.copy2(full_mask_src, os.path.join(tgt_mask_dir, simple_name))
                    
                    count += 1

    print(f"✅ Finished {split_name}: {count} pairs processed.")

def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"❌ Error: Source root not found at {SOURCE_ROOT}")
        return

    # Safety: Remove old target directory if exists to avoid mixing files
    if os.path.exists(TARGET_ROOT):
        print(f"🧹 Cleaning up old target directory: {TARGET_ROOT}")
        shutil.rmtree(TARGET_ROOT)

    print(f"📂 Source: {SOURCE_ROOT}")
    print(f"📂 Target: {TARGET_ROOT}")
    
    process_split('train')
    process_split('val')
    
    print("\n🎉 Preprocessing Complete! Filenames are now simplified.")

if __name__ == "__main__":
    main()