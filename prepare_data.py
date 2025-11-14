import os
import shutil
from PIL import Image
import random
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure for the dataset."""
    base_dirs = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    for base_dir in base_dirs:
        for class_name in classes:
            os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

def resize_image(image_path, target_size=(224, 224)):
    """Resize an image to the target size."""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img

def split_and_prepare_data(source_dir, train_ratio=0.7, val_ratio=0.15):
    """Split the dataset into train, validation, and test sets."""
    for class_name in ['NORMAL', 'PNEUMONIA']:
        source_path = os.path.join(source_dir, class_name)
        if not os.path.exists(source_path):
            print(f"Warning: {source_path} does not exist")
            continue
            
        # Get all image files
        image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(image_files)
        
        # Calculate split indices
        total_files = len(image_files)
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)

        # Split files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]

         # Process and save files
        for files, target_dir in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
            for file in files:
                source_file = os.path.join(source_path, file)
                target_file = os.path.join(target_dir, class_name, file)
                
                # Resize and save image
                resized_img = resize_image(source_file)
                resized_img.save(target_file)

def main():
    create_directory_structure()

    source_dir = 'train'
    split_and_prepare_data(source_dir)

    print("Data preparation completed!")

if __name__ == "__main__":
    main()