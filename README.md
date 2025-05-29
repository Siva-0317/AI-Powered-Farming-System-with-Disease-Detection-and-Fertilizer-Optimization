🌾 AI-Powered Smart Farming System
Automated Irrigation • Disease Detection • Fertilizer Optimization

🔍 Overview
This project integrates IoT and AI to modernize traditional farming practices. 
It features a smart irrigation control mechanism, crop disease detection using image processing, and a fertilizer recommendation system — 
all designed to improve agricultural efficiency and sustainability.

1. Crop Disease Detection (ResNet-50)
A fine-tuned ResNet-50 CNN model classifies crop diseases using leaf images.

Real-time image capture from USB Camera or Raspberry Pi.

Deployed on Django backend with REST API for prediction.

2. Fertilizer Recommendation (NLP + Tabular)
XGBoost model predicts required fertilizer quantity based on crop, soil, and weather.

A fine-tuned BERT model recommends preventive measures.

A BART text generation model provides advisory instructions in natural language.

💻 Tech Stack
Frontend/UI: Streamlit (optional), OLED/LCD display (Raspberry Pi)

Backend: Django REST Framework

Models: ResNet-50 (PyTorch), BERT/BART (Transformers), XGBoost, FFNN

IoT: Arduino Uno R4 WiFi, Raspberry Pi 4, USB Camera, DHT11/Soil Moisture Sensor

Database: MongoDB (cloud-hosted)

APIs: OpenWeatherMap, Gemini (for generative recommendations)

📢 Team & Acknowledgments
This project was built as part of KL University's Hack with IoT National Hackathon by:

Sivakumar Balaji

Aswath S

Harshitha KG

Deepan P

Special thanks to the mentors and judges for their valuable feedback.

