from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json
import uuid
import torch
import torchvision.transforms as transforms
import tensorflow as tf
import random
import numpy as np
from PIL import Image
import pytesseract
import cv2
import base64
import pdfplumber
import openai

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'medical-analysis-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Set OpenAI API key - replace with your actual key
openai.api_key = "your_openai_api_key"

# Ensure upload directories exist
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'reports'), exist_ok=True)

# Custom nl2br filter
def nl2br(value):
    """Converts newlines in text to HTML <br> tags."""
    return value.replace('\n', '<br>') if value else ''

# Register the filter with Jinja2
app.jinja_env.filters['nl2br'] = nl2br

# Initialize database
db = SQLAlchemy(app)

# Define database models
class Patient(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    medical_history = db.Column(db.Text, nullable=True)
    
    # Define relationships
    analyses = db.relationship('Analysis', backref='patient', lazy=True, cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': f"{self.first_name} {self.last_name}",
            'date_of_birth': self.date_of_birth.strftime('%Y-%m-%d') if self.date_of_birth else None,
            'gender': self.gender,
            'email': self.email,
            'phone': self.phone,
            'address': self.address,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'medical_history': self.medical_history
        }

class Analysis(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(36), db.ForeignKey('patient.id'), nullable=False)
    image_type = db.Column(db.String(20), nullable=True)  # 'brain' or 'chest' or None
    image_path = db.Column(db.String(255), nullable=True)
    report_path = db.Column(db.String(255), nullable=True)
    prediction = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    report_text = db.Column(db.Text, nullable=True)
    analysis_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'image_type': self.image_type,
            'image_path': self.image_path,
            'report_path': self.report_path,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'report_text': self.report_text,
            'analysis_text': self.analysis_text,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

# Create all tables
with app.app_context():
    db.create_all()

# Medical Analysis Agent class (updated with real model implementation)
class MedicalAnalysisAgent:
    def __init__(self):
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.brain_model = "/Users/krishilparikh/Desktop/diagnostic-assistant-/brain_tumor.pth"
        self.chest_model = "/Users/krishilparikh/Desktop/diagnostic-assistant-/pneumonia_detection_cnn_lstm_model.h5"
        
        # Try loading models, set to testing mode if models not available
        self.testing_mode = False
        try:
            # Update these paths to where your models are actually stored
            brain_model_path = os.path.join(os.getcwd(), 'models', '/Users/krishilparikh/Desktop/diagnostic-assistant-/brain_tumor.pth')
            self.brain_model = torch.load(brain_model_path, map_location=self.device)
            self.brain_model.eval()
        except Exception as e:
            print(f"\nWarning: Brain tumor model not found: {str(e)}")
            print("Running in testing mode with simulated predictions.")
            self.testing_mode = True
        
        try:
            chest_model_path = os.path.join(os.getcwd(), 'models', '/Users/krishilparikh/Desktop/diagnostic-assistant-/pneumonia_detection_cnn_lstm_model.h')
            self.chest_model = tf.keras.models.load_model(chest_model_path)
        except Exception as e:
            if not self.testing_mode:
                print(f"\nWarning: Chest X-ray model not found: {str(e)}")
                print("Running in testing mode with simulated predictions.")
                self.testing_mode = True
        
        # Define class labels
        self.brain_classes = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        self.chest_classes = ['NORMAL', 'PNEUMONIA']
        
        # Image preprocessing parameters
        self.brain_img_size = (224, 224)
        self.chest_img_size = (150, 150)
    
    def preprocess_image(self, image_path, image_type):
        """Preprocess the image based on type (brain/chest)"""
        try:
            # Open and convert image to RGB to ensure 3 channels
            image = Image.open(image_path).convert('RGB')
            
            if image_type.lower() == 'brain':
                # Brain image preprocessing for 3-channel input
                transform = transforms.Compose([
                    transforms.Resize(self.brain_img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                processed_image = transform(image).unsqueeze(0)
                return processed_image.to(self.device)
            
            else:  # Chest image
                # Convert to grayscale for chest X-rays
                image_gray = image.convert('L')
                image = image_gray.resize(self.chest_img_size)
                
                # Convert to numpy array and normalize
                image_array = np.array(image)
                image_array = image_array / 255.0
                
                # Add channel and batch dimensions
                image_array = np.expand_dims(image_array, axis=-1)
                image_array = np.expand_dims(image_array, axis=0)
                return image_array
                
        except Exception as e:
            print(f"Detailed preprocessing error: {str(e)}")
            print(f"Image path: {image_path}")
            print(f"Image type: {image_type}")
            raise Exception(f"Error preprocessing image: {str(e)}")

    def get_prediction(self, image_path, image_type):
        """Get prediction from the image"""
        if not image_path:
            return None, None
            
        try:
            # In testing mode, return simulated predictions
            if self.testing_mode:
                if image_type.lower() == 'brain':
                    prediction = random.choice(self.brain_classes)
                else:
                    prediction = random.choice(self.chest_classes)
                confidence = random.uniform(0.70, 0.99)
                return prediction, confidence
            
            # Process actual image and get prediction
            image = self.preprocess_image(image_path, image_type)
            
            if image_type.lower() == 'brain':
                if self.brain_model is None:
                    raise ValueError("Brain tumor model not available")
                
                with torch.no_grad():
                    outputs = self.brain_model(image)
                    _, predicted = torch.max(outputs, 1)
                    prediction = self.brain_classes[predicted.item()]
                    confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
            else:
                if self.chest_model is None:
                    raise ValueError("Chest X-ray model not available")
                
                prediction = self.chest_model.predict(image)
                predicted_class = int(prediction[0][0] > 0.5)
                confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]
                prediction = self.chest_classes[predicted_class]
                
            return prediction, confidence
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None

    def extract_text_from_report(self, report_path):
        """Extract text from medical report if provided"""
        if not report_path:
            return None
            
        try:
            # Check file extension
            file_extension = os.path.splitext(report_path)[1].lower()
            
            if file_extension == '.pdf':
                # Use pdfplumber for PDF text extraction
                with pdfplumber.open(report_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    return text.strip() if text else None
                    
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                # Use OCR for image-based reports
                image = cv2.imread(report_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                text = pytesseract.image_to_string(threshold)
                return text.strip() if text else None
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"Error in text extraction: {str(e)}")
            return None

    def generate_analysis(self, prediction, confidence, report_text, previous_records, image_type):
        """Generate analysis based on available data - using GPT if enabled, fallback to template"""
        try:
            # Check if OpenAI API key is valid
            if openai.api_key and not openai.api_key.startswith("your-"):
                return self.generate_gpt_response(prediction, confidence, report_text, previous_records, image_type)
            else:
                return self.generate_template_analysis(prediction, confidence, report_text, previous_records, image_type)
        except Exception as e:
            print(f"Error generating GPT analysis: {str(e)}")
            return self.generate_template_analysis(prediction, confidence, report_text, previous_records, image_type)

    def generate_gpt_response(self, prediction, confidence, report_text, previous_records, image_type):
        """Generate response using GPT-3.5 Turbo based on available data"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""You are a medical analysis assistant. Provide detailed, professional analysis 
        while maintaining a compassionate tone based on the available information.

        Date: {current_date}\n"""
        
        if prediction and confidence:
            prompt += f"""
            Image Type: {image_type.upper()} Scan
            AI Analysis Results:
            - Predicted Condition: {prediction}
            - Confidence Level: {confidence:.2f}%\n"""
        
        if report_text:
            # Clean up PDF text formatting
            cleaned_report = report_text.replace('\n\n', '\n').strip()
            prompt += f"""
            Current Medical Report Extract:
            {cleaned_report}\n"""
        
        prompt += f"""
        Patient History:
        {previous_records}
        
        Please provide:
        1. Analysis of the available findings
        2. Correlation with patient history
        3. Recommendations for next steps
        4. Any potential concerns or areas requiring attention
        5. General lifestyle and preventive recommendations
        
        Note: This analysis is based on {
            'both image analysis and medical report' if prediction and report_text
            else 'image analysis only' if prediction
            else 'medical report only' if report_text
            else 'patient history only'
        }."""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message['content']

        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return self.generate_template_analysis(prediction, confidence, report_text, previous_records, image_type)

    def generate_template_analysis(self, prediction, confidence, report_text, previous_records, image_type):
        """Generate analysis using templates as fallback when GPT is not available"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create base analysis text
        analysis = f"""Medical Analysis Report\n\nDate: {current_date}\n\n"""
        
        if prediction and confidence:
            analysis += f"""Image Analysis Results:\n- Image Type: {image_type.upper()} Scan\n- Predicted Condition: {prediction}\n- Confidence Level: {confidence:.2f}%\n\n"""
        
        if report_text:
            analysis += f"""Medical Report Extract:\n{report_text}\n\n"""
        
        analysis += f"""Comprehensive Analysis:\n
Based on {'both image analysis and medical report' if prediction and report_text 
    else 'image analysis only' if prediction 
    else 'medical report only' if report_text 
    else 'patient history only'}, the patient """
        
        # Generate different analysis based on prediction
        if prediction:
            if prediction.lower() == 'no tumor' or prediction.lower() == 'normal':
                analysis += f"""shows no significant abnormalities. The {image_type} scan appears to be within normal parameters with {confidence:.2f}% confidence.
Recommendations:
1. Continue routine check-ups as scheduled
2. Maintain a healthy lifestyle with regular exercise
3. Follow a balanced diet rich in antioxidants
4. Stay hydrated and avoid excessive stress
Follow-up:
No immediate follow-up required. Schedule next routine examination in 12 months.
"""
            else:
                analysis += f"""exhibits signs consistent with {prediction} with {confidence:.2f}% confidence. This warrants further investigation and potentially a specialist consultation.
Recommendations:
1. Schedule a follow-up examination within 2-4 weeks
2. Consult with a specialist in {image_type} conditions
3. Additional testing may be required to confirm diagnosis
4. Monitor for any changes in symptoms
Follow-up:
Urgent follow-up recommended. Please schedule an appointment with the appropriate specialist within 7-10 days.
"""
        return analysis

    def process_case(self, image_path, report_path, previous_records, image_type=None):
        """Process a medical case with optional inputs"""
        try:
            prediction = None
            confidence = None
            report_text = None
            
            if image_path:
                if not image_type:
                    raise ValueError("Image type must be specified when providing an image")
                prediction, confidence = self.get_prediction(image_path, image_type)
            
            if report_path:
                report_text = self.extract_text_from_report(report_path)
            
            analysis = self.generate_analysis(
                prediction, 
                confidence * 100 if confidence else None,
                report_text,
                previous_records,
                image_type if image_type else "N/A"
            )
            
            return {
                'prediction': prediction,
                'confidence': confidence * 100 if confidence else None,
                'report_text': report_text,
                'analysis_text': analysis,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

# Initialize the analysis agent
agent = MedicalAnalysisAgent()

# Helper functions
def allowed_file(filename, types):
    allowed_extensions = {
        'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
        'report': {'pdf', 'png', 'jpg', 'jpeg', 'bmp'}
    }
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions[types]

def save_uploaded_file(file, file_type, patient_id):
    if file and allowed_file(file.filename, file_type):
        # Secure the filename and generate a unique name
        filename = secure_filename(file.filename)
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
        
        # Determine the upload folder
        folder = 'images' if file_type == 'image' else 'reports'
        patient_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder, patient_id)
        os.makedirs(patient_folder, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(patient_folder, unique_filename)
        file.save(file_path)
        
        # Return relative path for database storage
        return os.path.join(folder, patient_id, unique_filename)
    return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patients')
def patient_list():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    return render_template('patients.html', patients=patients)

@app.route('/api/patients', methods=['GET'])
def get_patients():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    return jsonify([patient.to_dict() for patient in patients])

@app.route('/api/patients', methods=['POST'])
def add_patient():
    data = request.form.to_dict()
    try:
        # Parse date of birth
        dob = datetime.strptime(data['date_of_birth'], '%Y-%m-%d').date() if data.get('date_of_birth') else None
        
        # Create new patient
        patient = Patient(
            first_name=data['first_name'],
            last_name=data['last_name'],
            date_of_birth=dob,
            gender=data['gender'],
            email=data.get('email'),
            phone=data.get('phone'),
            address=data.get('address'),
            medical_history=data.get('medical_history')
        )
        db.session.add(patient)
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Patient added successfully',
            'patient': patient.to_dict()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/patient/<patient_id>')
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    analyses = Analysis.query.filter_by(patient_id=patient_id).order_by(Analysis.created_at.desc()).all()
    return render_template('patient_detail.html', patient=patient, analyses=analyses)

@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return jsonify(patient.to_dict())

@app.route('/api/patients/<patient_id>/analyses', methods=['GET'])
def get_patient_analyses(patient_id):
    analyses = Analysis.query.filter_by(patient_id=patient_id).order_by(Analysis.created_at.desc()).all()
    return jsonify([analysis.to_dict() for analysis in analyses])

@app.route('/api/patients/<patient_id>/analyses', methods=['POST'])
def add_analysis(patient_id):
    # Ensure patient exists
    patient = Patient.query.get_or_404(patient_id)
    
    # Get form data
    image_type = request.form.get('image_type')
    image_file = request.files.get('image_file')
    report_file = request.files.get('report_file')
    
    # Save uploaded files
    image_path = save_uploaded_file(image_file, 'image', patient_id) if image_file else None
    report_path = save_uploaded_file(report_file, 'report', patient_id) if report_file else None
    
    # Process the case
    result = agent.process_case(
        image_path=os.path.join(app.config['UPLOAD_FOLDER'], image_path) if image_path else None,
        report_path=os.path.join(app.config['UPLOAD_FOLDER'], report_path) if report_path else None,
        previous_records=patient.medical_history,
        image_type=image_type
    )
    
    if result['status'] == 'success':
        # Create new analysis record
        analysis = Analysis(
            patient_id=patient_id,
            image_type=image_type,
            image_path=image_path,
            report_path=report_path,
            prediction=result['prediction'],
            confidence=result['confidence'],
            report_text=result['report_text'],
            analysis_text=result['analysis_text']
        )
        db.session.add(analysis)
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Analysis completed successfully',
            'analysis': analysis.to_dict()
        })
    else:
        return jsonify({
            'status': 'error',
            'message': result.get('error', 'Unknown error processing case')
        }), 400

@app.route('/analysis/<analysis_id>')
def analysis_detail(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    patient = Patient.query.get_or_404(analysis.patient_id)
    return render_template('analysis_detail.html', analysis=analysis, patient=patient)

@app.route('/api/analyses/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    return jsonify(analysis.to_dict())

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/new_patient')
def new_patient_form():
    return render_template('new_patient.html')

@app.route('/new_analysis/<patient_id>')
def new_analysis_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template('new_analysis.html', patient=patient)

@app.route('/api/statistics')
def get_statistics():
    """Endpoint to fetch general statistics for the dashboard."""
    try:
        # Get total number of patients
        total_patients = Patient.query.count()

        # Get total number of analyses
        total_analyses = Analysis.query.count()

        # Get total number of brain and chest scans
        brain_scans = Analysis.query.filter_by(image_type='brain').count()
        chest_scans = Analysis.query.filter_by(image_type='chest').count()

        return jsonify({
            'status': 'success',
            'patients': total_patients,
            'analyses': total_analyses,
            'brainScans': brain_scans,
            'chestScans': chest_scans
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/statistics/charts')
def get_statistics_charts():
    """Endpoint to fetch data for charts on the dashboard."""
    try:
        # Get analysis distribution data
        brain_scans = Analysis.query.filter_by(image_type='brain').count()
        chest_scans = Analysis.query.filter_by(image_type='chest').count()
        other_scans = Analysis.query.filter(Analysis.image_type.is_(None)).count()

        # Get confidence distribution data
        confidence_ranges = {
            '90-100': Analysis.query.filter(Analysis.confidence >= 90.0).count(),
            '80-89': Analysis.query.filter(Analysis.confidence >= 80.0, Analysis.confidence < 90.0).count(),
            '70-79': Analysis.query.filter(Analysis.confidence >= 70.0, Analysis.confidence < 80.0).count(),
            '60-69': Analysis.query.filter(Analysis.confidence >= 60.0, Analysis.confidence < 70.0).count(),
            'below60': Analysis.query.filter(Analysis.confidence < 60.0).count()
        }

        return jsonify({
            'status': 'success',
            'analysisDistribution': {
                'brainScans': brain_scans,
                'chestScans': chest_scans,
                'other': other_scans
            },
            'confidenceDistribution': confidence_ranges
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)