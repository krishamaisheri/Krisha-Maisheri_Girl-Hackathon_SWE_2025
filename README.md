# Krisha-Maisheri\_Girl-Hackathon\_SWE\_2025

# Medical Analysis System

## Overview

This project is a comprehensive medical analysis system designed to assist in diagnosing various diseases, including pneumonia and brain tumors. By analyzing patients' medical reports and test results, the system aims to provide early detection and insights. It consists of a Flask-based backend, multiple machine learning models, and a database management system for handling patient records efficiently.

## Features

- **Disease Diagnosis API**: Leverages OpenAI, FAISS, and sentence transformers to analyze medical data and provide potential diagnoses.
- **Pneumonia Detection**: A Jupyter notebook implementing machine learning models to detect pneumonia from medical images.
- **Brain Tumor Detection**: A deep learning model for detecting brain tumors from MRI scans.
- **Medical Record Management**: A Flask-based system to store and manage patient records, including medical reports and diagnostic images.
- **Secure File Upload**: Allows users to upload medical images and reports for automated analysis.
- **Automated Report Analysis via WhatsApp**: Patients who prefer physical consultation but can't come meet the doctor physically, can send their reports via WhatsApp. The system will automatically analyze the reports using AI models and generate an initial response. The doctor can review, approve, or edit the response before it is sent to the patient.

## Project Structure

```
├── symptoms_backend.py   # Backend API for disease diagnosis
├── pneumonia.ipynb       # Jupyter notebook for pneumonia detection
├── brain_tumor_model.py  # Machine learning model for brain tumor detection
├── whatsapp_bot.py       # WhatsApp integration for automated report analysis
├── main.py               # Main backend for medical analysis system
├── requirements.txt      # Dependencies required for the project
├── uploads/              # Directory for storing uploaded images and reports
├── templates/            # HTML templates for frontend (if applicable)
└── static/               # Static files (CSS, JavaScript, images)
```

## Installation

### Prerequisites

- Python 3.x
- Flask
- TensorFlow & PyTorch
- OpenAI API Key
- FAISS & Sentence Transformers
- Twilio API (for WhatsApp integration)

### Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/krishamaisheri/Krisha-Maisheri_Girl-Hackathon_SWE_2025.git
   cd medical-analysis-system
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up the database:
   ```sh
   python -c "from main import db; db.create_all()"
   ```
4. Configure Twilio API for WhatsApp integration in `whatsapp_bot.py`.
5. Run the Flask application:
   ```sh
   python main.py
   python symptoms_backend.py
   ```

## Usage

- Access the disease diagnosis API at `http://localhost:5000/api/diseases`
- Utilize the pneumonia detection notebook to analyze medical images
- Upload medical reports and patient records via the web interface for automated analysis
- Send medical reports via WhatsApp for AI-based analysis and receive a response after doctor approval

## API Endpoints

| Method | Endpoint           | Description                               |
| ------ | ------------------ | ----------------------------------------- |
| POST   | `/api/diseases`    | Analyze medical data for diagnosis        |
| GET    | `/api/patients`    | Retrieve all patient records              |
| POST   | `/api/upload`      | Upload medical images or reports          |
| GET    | `/api/brain_tumor` | Detect brain tumors from MRI scans        |
| POST   | `/api/whatsapp`    | Process medical reports sent via WhatsApp |

## Contributors

- **Krisha Maisheri** - Developer & Project Lead

