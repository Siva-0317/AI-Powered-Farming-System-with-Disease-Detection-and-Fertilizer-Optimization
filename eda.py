import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set the path to the dataset
data_dir = "./PlantVillage_jpg"  # Update this to your dataset path

# Get the list of all class folders
class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# Count the number of classes
num_classes = len(class_folders)
print(f"Total number of classes: {num_classes}")
print("Classes:", class_folders)

# Count images per class
image_counts = {}
for class_name in class_folders:
    class_path = os.path.join(data_dir, class_name)
    num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    image_counts[class_name] = num_images

# Convert to DataFrame for easier analysis
df = pd.DataFrame(list(image_counts.items()), columns=['Class', 'Image_Count'])
print("\nDataset Summary:")
print(df)

# Basic statistics
print("\nBasic Statistics:")
print(df['Image_Count'].describe())

# Bar plot of image counts per class
plt.figure(figsize=(12, 6))
sns.barplot(x='Class', y='Image_Count', data=df)
plt.xticks(rotation=90)
plt.title("Number of Images per Class")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.show()

# Pie chart for proportion
plt.figure(figsize=(10, 10))
plt.pie(df['Image_Count'], labels=df['Class'], autopct='%1.1f%%', startangle=90)
plt.title("Proportion of Images per Class")
plt.axis('equal')
plt.show()

# Function to get image properties
def get_image_properties(class_name, image_path):
    img = Image.open(image_path)
    return {
        'Class': class_name,
        'Width': img.width,
        'Height': img.height,
        'Format': img.format
    }

# Collect properties for a sample of images
image_properties = []
for class_name in class_folders[:5]:  # Sample first 5 classes
    class_path = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]  # Sample 5 images per class
    for img_file in images:
        img_path = os.path.join(class_path, img_file)
        props = get_image_properties(class_name, img_path)
        image_properties.append(props)

# Convert to DataFrame
img_df = pd.DataFrame(image_properties)
print("\nImage Properties Sample:")
print(img_df)

# Visualize image dimensions
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Width', y='Height', hue='Class', data=img_df)
plt.title("Image Dimensions by Class")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.show()

# Display sample images
plt.figure(figsize=(15, 5))
for i, class_name in enumerate(class_folders[:5]):  # Show 1 image from first 5 classes
    class_path = os.path.join(data_dir, class_name)
    img_file = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))][0]
    img_path = os.path.join(class_path, img_file)
    img = Image.open(img_path)
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(class_name)
    plt.axis('off')
plt.tight_layout()
plt.show()