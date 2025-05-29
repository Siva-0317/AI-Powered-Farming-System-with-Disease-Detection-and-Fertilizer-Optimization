import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# Paths
data_dir = "D:/smart_irrigation/PlantVillage_jpg"
split_dir = "./PlantVillage_split"

# Remove existing split directory if it exists (to start fresh)
if os.path.exists(split_dir):
    shutil.rmtree(split_dir)

# Create directories for train, val, test
for split in ['train', 'val', 'test']:
    split_path = os.path.join(split_dir, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    # Create class subfolders in each split
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(split_path, class_name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)

# Collect all image paths and labels
image_paths = []
labels = []
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for img_file in os.listdir(class_path):
            if img_file.endswith('.jpg'):
                image_paths.append(os.path.join(class_path, img_file))
                labels.append(class_name)

# Filter out classes with fewer than 2 images
class_counts = Counter(labels)
valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
filtered_image_paths = []
filtered_labels = []
for img_path, label in zip(image_paths, labels):
    if label in valid_classes:
        filtered_image_paths.append(img_path)
        filtered_labels.append(label)

# Print class distribution after filtering
print("Class distribution after filtering:")
for class_name, count in Counter(filtered_labels).items():
    print(f"{class_name}: {count} images")
print(f"Total images after filtering: {len(filtered_image_paths)}")

# Split the dataset (70% train, 15% val, 15% test)
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    filtered_image_paths, filtered_labels, test_size=0.3, stratify=filtered_labels, random_state=42
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

# Function to copy images to the appropriate split and class subfolder
def copy_images(paths, labels, split):
    for img_path, label in zip(paths, labels):
        # Destination directory: split/class_name/
        dest_dir = os.path.join(split_dir, split, label)
        # Get the filename from the image path
        img_filename = os.path.basename(img_path)
        # Destination path for the image
        dest_path = os.path.join(dest_dir, img_filename)
        # Copy the image
        shutil.copy(img_path, dest_path)

# Copy images to train, val, test directories
copy_images(train_paths, train_labels, 'train')
copy_images(val_paths, val_labels, 'val')
copy_images(test_paths, test_labels, 'test')

# Verify the split
for split in ['train', 'val', 'test']:
    split_path = os.path.join(split_dir, split)
    total_images = 0
    print(f"\n{split.capitalize()} split:")
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        num_images = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
        total_images += num_images
        print(f"{class_name}: {num_images} images")
    print(f"Total {split} images: {total_images}")