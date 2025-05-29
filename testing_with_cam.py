import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
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

# Load class names from the training dataset
train_dir = os.path.join(data_dir, "train")
class_names = sorted(os.listdir(train_dir))
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# Load the pretrained ResNet-50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Replace the final fully connected layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load the saved model weights
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

# Define the image transformations (same as used during training/testing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to preprocess and predict on a single frame
def predict_frame(frame, model, transform, class_names, device):
    # Convert the frame (numpy array in BGR format) to a PIL Image (RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Apply transformations
    image_tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item() * 100  # Convert to percentage

    return predicted_class, confidence

# Main function for real-time detection
def real_time_detection():
    # Open the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time detection. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Make prediction on the frame
        predicted_class, confidence = predict_frame(frame, model, transform, class_names, device)

        # Overlay the prediction on the frame
        text = f"Class: {predicted_class} ({confidence:.2f}%)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Leaf Disease Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Real-time detection stopped.")

# Main execution block
if __name__ == '__main__':
    real_time_detection()
    print("Script completed.")