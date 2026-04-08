import os
import shutil
from pathlib import Path

def merge_seadronesee_to_c2a():
    """
    Merge SeaDroneSee dataset into C2A dataset
    Copy images and labels sequentially to maintain index matching
    """
    
    # Source paths (SeaDroneSee)
    seadronesee_base = "/home/pragesh-shrestha/Desktop/binayak_sir/SeaDroneSee v2.v1i.yolo26"
    seadronesee_train_images = os.path.join(seadronesee_base, "train", "images")
    seadronesee_train_labels = os.path.join(seadronesee_base, "train", "labels")
    seadronesee_test_images = os.path.join(seadronesee_base, "test", "images")
    seadronesee_test_labels = os.path.join(seadronesee_base, "test", "labels")
    
    # Target paths (C2A)
    c2a_base = "/home/pragesh-shrestha/Desktop/binayak_sir/archive/C2A_Dataset/new_dataset3"
    c2a_train_images = os.path.join(c2a_base, "train", "images")
    c2a_train_labels = os.path.join(c2a_base, "train", "labels")
    c2a_test_images = os.path.join(c2a_base, "test", "images")
    c2a_test_labels = os.path.join(c2a_base, "test", "labels")
    
    # Get current C2A file counts
    c2a_train_img_count = len([f for f in os.listdir(c2a_train_images) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(c2a_train_images) else 0
    c2a_train_label_count = len([f for f in os.listdir(c2a_train_labels) if f.endswith('.txt')]) if os.path.exists(c2a_train_labels) else 0
    c2a_test_img_count = len([f for f in os.listdir(c2a_test_images) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(c2a_test_images) else 0
    c2a_test_label_count = len([f for f in os.listdir(c2a_test_labels) if f.endswith('.txt')]) if os.path.exists(c2a_test_labels) else 0
    
    print(f"Current C2A dataset:")
    print(f"  Train: {c2a_train_img_count} images, {c2a_train_label_count} labels")
    print(f"  Test: {c2a_test_img_count} images, {c2a_test_label_count} labels")
    
    # Merge train dataset
    print("\nMerging SeaDroneSee train dataset...")
    train_copied = copy_dataset_files(
        seadronesee_train_images, 
        seadronesee_train_labels,
        c2a_train_images,
        c2a_train_labels,
        start_index=c2a_train_img_count + 1
    )
    
    # Merge test dataset
    print("\nMerging SeaDroneSee test dataset...")
    test_copied = copy_dataset_files(
        seadronesee_test_images,
        seadronesee_test_labels, 
        c2a_test_images,
        c2a_test_labels,
        start_index=c2a_test_img_count + 1
    )
    
    print(f"\nMerge completed!")
    print(f"Train: copied {train_copied} images and labels")
    print(f"Test: copied {test_copied} images and labels")
    
    # Show final counts
    final_train_img = len([f for f in os.listdir(c2a_train_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
    final_train_label = len([f for f in os.listdir(c2a_train_labels) if f.endswith('.txt')])
    final_test_img = len([f for f in os.listdir(c2a_test_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
    final_test_label = len([f for f in os.listdir(c2a_test_labels) if f.endswith('.txt')])
    
    print(f"\nFinal C2A dataset:")
    print(f"  Train: {final_train_img} images, {final_train_label} labels")
    print(f"  Test: {final_test_img} images, {final_test_label} labels")

def copy_dataset_files(src_images, src_labels, dst_images, dst_labels, start_index):
    """
    Copy images and labels with sequential numbering
    """
    # Create destination directories if they don't exist
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    
    # Get all label files (they correspond to images)
    label_files = [f for f in os.listdir(src_labels) if f.endswith('.txt')]
    
    copied_count = 0
    
    for i, label_file in enumerate(sorted(label_files)):
        # Find corresponding image file
        label_name = Path(label_file).stem
        image_file = None
        
        # Look for image with same name (different extensions)
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_image = os.path.join(src_images, label_name + ext)
            if os.path.exists(potential_image):
                image_file = potential_image
                break
        
        if image_file is None:
            print(f"Warning: No image found for label {label_file}")
            continue
        
        # New sequential names
        new_index = start_index + i
        new_label_name = f"merged_{new_index:06d}.txt"
        new_image_name = f"merged_{new_index:06d}{Path(image_file).suffix}"
        
        # Copy files
        try:
            shutil.copy2(
                os.path.join(src_labels, label_file),
                os.path.join(dst_labels, new_label_name)
            )
            shutil.copy2(
                image_file,
                os.path.join(dst_images, new_image_name)
            )
            copied_count += 1
        except Exception as e:
            print(f"Error copying {label_file}: {e}")
    
    return copied_count

if __name__ == "__main__":
    merge_seadronesee_to_c2a()
