import torch
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
file_path = "updated_fertilizer_recommendation_dataset.csv"  # Update with correct path
df = pd.read_csv(file_path)

# Prepare text inputs and outputs
def create_text_input(row):
    return (
        f"Temperature: {row['Temperature']}, "
        f"Moisture: {row['Moisture']}, "
        f"Rainfall: {row['Rainfall']}, "
        f"PH: {row['PH']}, "
        f"Nitrogen: {row['Nitrogen']}, "
        f"Phosphorous: {row['Phosphorous']}, "
        f"Potassium: {row['Potassium']}, "
        f"Carbon: {row['Carbon']}, "
        f"Soil Type: {row['Soil']}, "
        f"Crop: {row['Crop']}, "
        f"Land Area: {row['Land Area Used (ha)']}"
    )

def create_text_output(row):
    return (
        f"Fertilizer: {row['Fertilizer']}. "
        f"Remark: {row['Remark']}. "
        f"Amount: {round(row['Fertilizer Amount (kg/ha)'], 2)} "
    )

df["Input_Text"] = df.apply(create_text_input, axis=1)
df["Output_Text"] = df.apply(create_text_output, axis=1)

# Convert to Hugging Face dataset
train_dataset = Dataset.from_pandas(df[["Input_Text", "Output_Text"]])

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

# Tokenize dataset
def tokenize_function(examples):
    inputs = tokenizer(examples["Input_Text"], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(examples["Output_Text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fertilizer_bart_model",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    push_to_hub=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train model
trainer.train()

# Save trained model
model.save_pretrained("./trained_fertilizer_bart_model")
tokenizer.save_pretrained("./trained_fertilizer_bart_model")
print("Model training complete. Saved in 'trained_fertilizer_bart_model'.")