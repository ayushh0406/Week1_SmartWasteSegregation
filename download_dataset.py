"""
Download and prepare waste classification dataset
Downloads different waste images for better training
"""

import os
import urllib.request
from pathlib import Path
import shutil

def download_sample_images():
    """Download different waste images from internet"""
    
    print("� Downloading sample waste images...")
    
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    # Image URLs - different waste images
    image_urls = {
        'Recyclable': [
            'https://images.unsplash.com/photo-1604187351574-c75ca79f5807?w=400',  # plastic bottle
            'https://images.unsplash.com/photo-1607968565043-36af90dde238?w=400',  # cardboard
            'https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=400',  # paper
            'https://images.unsplash.com/photo-1610557892470-55d9e80c0bce?w=400',  # glass
            'https://images.unsplash.com/photo-1621951753163-10684d26117e?w=400',  # metal can
        ],
        'Organic': [
            'https://images.unsplash.com/photo-1610348725531-843dff563e2c?w=400',  # food waste
            'https://images.unsplash.com/photo-1628773822503-930a7eaecf80?w=400',  # vegetables
            'https://images.unsplash.com/photo-1542838132-92c53300491e?w=400',  # leaves
            'https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=400',  # organic matter
        ],
        'Non-Recyclable': [
            'https://images.unsplash.com/photo-1530587191325-3db32d826c18?w=400',  # trash
            'https://images.unsplash.com/photo-1609476784169-7a3e3e0a5f34?w=400',  # waste
            'https://images.unsplash.com/photo-1611284446314-60a58ac0deb9?w=400',  # mixed waste
        ]
    }
    
    for class_name, urls in image_urls.items():
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        print(f"\n📂 Downloading {class_name} images...")
        
        for idx, url in enumerate(urls, 1):
            try:
                img_path = class_dir / f'{class_name.lower()}_{idx}.jpg'
                if not img_path.exists():
                    print(f"   Downloading image {idx}...")
                    urllib.request.urlretrieve(url, img_path)
                    print(f"   ✅ Saved: {img_path.name}")
            except Exception as e:
                print(f"   ⚠️  Failed to download image {idx}: {e}")
    
    return dataset_dir

def create_sample_dataset():
    """Copy from dataset_sample without excessive augmentation"""
    
    print("\n📂 Preparing training dataset...")
    
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    sample_dir = Path('dataset_sample')
    
    if sample_dir.exists():
        for class_name in ['Recyclable', 'Organic', 'Non-Recyclable']:
            src_class_dir = sample_dir / class_name
            dst_class_dir = dataset_dir / class_name
            dst_class_dir.mkdir(exist_ok=True)
            
            if src_class_dir.exists():
                print(f"  Processing {class_name}...")
                
                # Just copy original images - no augmentation
                for img_file in src_class_dir.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(img_file, dst_class_dir / img_file.name)
    
    # Count images
    total = 0
    print("\n📊 Dataset Summary:")
    for class_name in ['Recyclable', 'Organic', 'Non-Recyclable']:
        class_dir = dataset_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob('*.*')))
            total += count
            print(f"  {class_name}: {count} images")
    
    print(f"\n  Total: {total} images")
    
    if total < 5:
        print("\n⚠️  Warning: Very few images! For better results:")
        print("  - Add more images to dataset_sample/ folder")
        print("  - Or download Kaggle dataset")
    
    return dataset_dir

if __name__ == "__main__":
    print("="*60)
    print("WASTE CLASSIFICATION DATASET SETUP")
    print("="*60)
    
    # Try to download images (optional)
    try:
        download_sample_images()
    except Exception as e:
        print(f"\nCould not download images: {e}")
        print("Using local dataset_sample/ instead")
    
    # Prepare dataset
    dataset_dir = create_sample_dataset()
    
    print("\n"+"="*60)
    print("DATASET READY")
    print("="*60)
    print(f"\nLocation: {dataset_dir.absolute()}")
    print("\nNext steps:")
    print("  1. python train_simple_model.py")
    print("  2. streamlit run app.py")
    print("\nFor better results, add more images to dataset/")
    print("="*60)
