import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Define paths
model_path = "D:/smart_irrigation/resnet50_plantvillage.pth"
data_dir = "./PlantVillage_split"
sample_image_path = "C:/Users/sivab/Downloads/potatolate.jpeg"  # Path to your .jpeg image
converted_image_path = "C:/Users/sivab/Downloads/potatolate.jpg"  # Path to save the .jpg image

# Step 1: Convert .jpeg to .jpg
def convert_jpeg_to_jpg(jpeg_path, jpg_path):
    try:
        # Load the .jpeg image
        image = Image.open(jpeg_path).convert("RGB")
        # Save as .jpg
        image.save(jpg_path, "JPEG")
        print(f"Successfully converted {jpeg_path} to {jpg_path}")
    except Exception as e:
        print(f"Error converting image: {e}")
        return False
    return True

# Step 2: Load class names from the training dataset
train_dir = os.path.join(data_dir, "train")
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Step 3: Load the pretrained ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Replace the final fully connected layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

# Step 4: Define the image transformations (same as used during training/testing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 5: Function to predict on a single image
def predict_image(image_path, model, transform, class_names, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item() * 100  # Convert to percentage

    return predicted_class, confidence

# Main execution block
if __name__ == '__main__':
    # Ensure the sample_images directory exists
    sample_images_dir = os.path.dirname(sample_image_path)
    if not os.path.exists(sample_images_dir):
        os.makedirs(sample_images_dir)
        print(f"Created directory: {sample_images_dir}")

    # Convert the image
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at {sample_image_path}. Please check the path.")
    else:
        success = convert_jpeg_to_jpg(sample_image_path, converted_image_path)
        if success:
            # Test the model on the converted image
            print("\nTesting on the converted image:")
            predicted_class, confidence = predict_image(converted_image_path, model, transform, class_names, device)
            print(f"Image: {os.path.basename(converted_image_path)}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")
        else:
            print("Failed to convert the image. Cannot proceed with prediction.")

    print("\nScript completed.")