# Krisha-Maisheri_Girl-Hackathon_SWE_2025

# Medical Analysis System

## Overview
This project is a medical analysis system designed to assist in symptom diagnosis and pneumonia detection using machine learning. The system consists of a Flask-based backend, a pneumonia analysis model, and database management for patient records.

## Features
- **Symptom Analysis API**: Uses OpenAI, FAISS, and sentence transformers to analyze medical symptoms.
- **Pneumonia Detection**: A Jupyter notebook for pneumonia detection using machine learning models.
- **Medical Record Management**: A Flask-based system to handle patient records, medical reports, and image processing.

## Project Structure
```
├── symptoms_backend.py   # Backend API for symptom analysis
├── pneumonia.ipynb       # Jupyter notebook for pneumonia detection
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

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/medical-analysis-system.git
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
4. Run the Flask application:
   ```sh
   python main.py
   ```

## Usage
- Access the symptom analysis API at `http://localhost:5000/api/symptoms`
- Use the pneumonia detection notebook to analyze medical images
- Upload medical reports and patient records via the web interface

## API Endpoints
| Method | Endpoint               | Description                        |
|--------|------------------------|------------------------------------|
| POST   | `/api/symptoms`        | Analyze symptoms for diagnosis    |
| GET    | `/api/patients`        | Retrieve all patient records      |
| POST   | `/api/upload`          | Upload medical images or reports  |

## Contributors
- **Your Name** - Krisha Maisheri

## License
This project is licensed under the MIT License.

