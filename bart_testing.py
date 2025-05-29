import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load fine-tuned model and tokenizer
model_path = "./trained_fertilizer_bart_model"  # Update if saved elsewhere
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

def predict_fertilizer(input_text):
    """Generates fertilizer recommendation based on input text."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=128, num_return_sequences=1)
    
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return prediction

def get_user_input():
    """Collects input factors from the user during runtime."""
    print("Enter the following factors for fertilizer recommendation:")
    
    temperature = input("Temperature (e.g., 30): ")
    moisture = input("Moisture (e.g., 60): ")
    rainfall = input("Rainfall (e.g., 120): ")
    ph = input("PH (e.g., 6.5): ")
    nitrogen = input("Nitrogen (e.g., 50): ")
    phosphorous = input("Phosphorous (e.g., 40): ")
    potassium = input("Potassium (e.g., 30): ")
    carbon = input("Carbon (e.g., High, Medium, Low): ")
    soil_type = input("Soil Type (e.g., Loamy, Sandy, Clay): ")
    crop = input("Crop (e.g., Wheat, Rice): ")
    land_area = input("Land Area (e.g., 5): ")

    # Format the input into the expected text string
    input_text = (
        f"Temperature: {temperature}, Moisture: {moisture}, Rainfall: {rainfall}, PH: {ph}, "
        f"Nitrogen: {nitrogen}, Phosphorous: {phosphorous}, Potassium: {potassium}, "
        f"Carbon: {carbon}, Soil Type: {soil_type}, Crop: {crop}, Land Area: {land_area}"
    )
    return input_text

# Main execution
if __name__ == "__main__":
    # Get input from user
    input_text = get_user_input()
    
    # Generate prediction
    prediction = predict_fertilizer(input_text)
    print("\nPredicted Fertilizer Recommendation:", prediction)