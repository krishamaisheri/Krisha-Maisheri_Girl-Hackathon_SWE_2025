from flask import Flask, request, jsonify, render_template, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import uuid
import pandas as pd
import openai
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Use the same database configuration as main backend
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///medical_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24)

# Initialize database with existing tables
db = SQLAlchemy(app)

# Existing Patient model (must match exactly with main backend)
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
    diagnoses = db.relationship('SymptomDiagnosis', backref='patient', lazy=True, cascade="all, delete-orphan")
    
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

# Analysis model from main backend to prevent conflicts
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

# New Diagnosis Feature Models
class Disease(db.Model):
    __tablename__ = 'disease'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), unique=True, nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    is_approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Precaution(db.Model):
    __tablename__ = 'precaution'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    disease_id = db.Column(db.String(36), db.ForeignKey('disease.id'), nullable=False)
    precautions = db.Column(db.Text, nullable=False)  # JSON string
    doctor_notes = db.Column(db.Text)
    approved_by = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SymptomDiagnosis(db.Model):
    __tablename__ = 'symptom_diagnosis'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = db.Column(db.String(36), db.ForeignKey('patient.id'), nullable=False)
    disease_id = db.Column(db.String(36), db.ForeignKey('disease.id'))
    symptoms = db.Column(db.Text, nullable=False)
    doctor_notes = db.Column(db.Text)
    is_approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables (only new ones will be created)
with app.app_context():
    db.create_all()

# AI Components
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = None
disease_embeddings = None
disease_list = []

def refresh_ai_components():
    global faiss_index, disease_embeddings, disease_list
    # Ensure this database query happens within app context
    with app.app_context():
        diseases = Disease.query.filter_by(is_approved=True).all()
        disease_list = [(d.id, d.name, d.symptoms) for d in diseases]
        
        if disease_list:
            symptoms_list = [d[2] for d in disease_list]
            disease_embeddings = embedding_model.encode(symptoms_list, convert_to_numpy=True)
            faiss_index = faiss.IndexFlatL2(disease_embeddings.shape[1])
            faiss_index.add(disease_embeddings.astype(np.float32))

# Initialize AI components within app context
with app.app_context():
    refresh_ai_components()
# Import disease dataset if the disease table is empty
def import_disease_dataset():
    if Disease.query.count() == 0:
        try:
            # Try to load from CSV files
            try:
                disease_symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")
                disease_precaution_df = pd.read_csv("Disease precaution.csv")
            except FileNotFoundError:
                print("Warning: CSV files not found. Cannot import disease data.")
                return
            
            # Process disease symptoms
            unique_diseases = disease_symptom_df["Disease"].unique()
            for disease_name in unique_diseases:
                # Get all rows for this disease
                disease_rows = disease_symptom_df[disease_symptom_df["Disease"] == disease_name]
                
                # Collect all symptoms for this disease
                all_symptoms = []
                for _, row in disease_rows.iterrows():
                    symptom_columns = [col for col in disease_symptom_df.columns if col.startswith("Symptom")]
                    for col in symptom_columns:
                        if pd.notna(row[col]) and row[col]:
                            all_symptoms.append(row[col])
                
                # Create new disease record
                symptoms_text = ", ".join(all_symptoms)
                new_disease = Disease(
                    name=disease_name,
                    symptoms=symptoms_text,
                    is_approved=True
                )
                db.session.add(new_disease)
                db.session.flush()  # Get the ID
                
                # Add precautions if available
                precaution_row = disease_precaution_df[disease_precaution_df["Disease"] == disease_name]
                if not precaution_row.empty:
                    precaution_list = []
                    for i in range(1, 5):
                        col_name = f"Precaution_{i}"
                        if col_name in precaution_row.columns and pd.notna(precaution_row.iloc[0][col_name]):
                            precaution_list.append(precaution_row.iloc[0][col_name])
                    
                    if precaution_list:
                        new_precaution = Precaution(
                            disease_id=new_disease.id,
                            precautions=json.dumps(precaution_list),
                            doctor_notes="Imported from dataset",
                            approved_by="System"
                        )
                        db.session.add(new_precaution)
            
            db.session.commit()
            print(f"Successfully imported {len(unique_diseases)} diseases with their symptoms and precautions.")
            
            # Refresh AI components with the new data
            refresh_ai_components()
        except Exception as e:
            db.session.rollback()
            print(f"Error importing disease dataset: {str(e)}")

# Try to import disease dataset when starting
with app.app_context():
    import_disease_dataset()

# OpenAI Configuration - Use environment variable if available
openai_api_key = os.environ.get('OPENAI_API_KEY', "your_openai_api_key")
openai.api_key = openai_api_key

@app.route('/diagnosis/suggest', methods=['POST'])
def suggest_diagnosis():
    data = request.get_json()
    patient_id = data.get('patient_id')
    symptoms = data.get('symptoms', '')
    
    if not patient_id or not symptoms:
        return jsonify({'error': 'Missing patient ID or symptoms'}), 400
    
    # Check patient exists
    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Find matching disease
    suggestions = []
    
    # Make sure we have disease data and AI components are ready
    if faiss_index is None or len(disease_list) == 0:
        refresh_ai_components()
        if faiss_index is None or len(disease_list) == 0:
            return jsonify({'error': 'No disease data available. Please import disease dataset.'}), 500
    
    # Find matching diseases
    input_embedding = embedding_model.encode([symptoms], convert_to_numpy=True)
    distances, indices = faiss_index.search(input_embedding.astype(np.float32), k=3)
    
    for i in range(len(indices[0])):
        if indices[0][i] < len(disease_list) and distances[0][i] < 1.5:  # Adjusted threshold for better matches
            disease_id, name, _ = disease_list[indices[0][i]]
            precautions = Precaution.query.filter_by(disease_id=disease_id).first()
            suggestions.append({
                'disease_id': disease_id,
                'name': name,
                'confidence': float(1 - distances[0][i] / 2),  # Normalize confidence score
                'precautions': json.loads(precautions.precautions) if precautions and precautions.precautions else []
            })
    
    # Generate AI analysis
    ai_notes = generate_ai_analysis(symptoms, suggestions)
    
    # Create diagnosis record
    new_diagnosis = SymptomDiagnosis(
        patient_id=patient_id,
        symptoms=symptoms,
        doctor_notes=ai_notes
    )
    if suggestions:
        new_diagnosis.disease_id = suggestions[0]['disease_id']
    
    db.session.add(new_diagnosis)
    db.session.commit()
    
    return jsonify({
        'diagnosis_id': new_diagnosis.id,
        'suggestions': suggestions,
        'ai_notes': ai_notes
    })

@app.route('/diagnosis/confirm/<diagnosis_id>', methods=['PUT'])
def confirm_diagnosis(diagnosis_id):
    data = request.get_json()
    diagnosis = SymptomDiagnosis.query.get_or_404(diagnosis_id)
    
    # Update diagnosis
    diagnosis.is_approved = True
    diagnosis.doctor_notes = data.get('doctor_notes', diagnosis.doctor_notes)
    
    # Handle new disease creation
    if 'new_disease' in data:
        disease_data = data['new_disease']
        new_disease = Disease(
            name=disease_data['name'],
            symptoms=diagnosis.symptoms,
            is_approved=True
        )
        db.session.add(new_disease)
        db.session.flush()  # Get the new ID
        
        new_precaution = Precaution(
            disease_id=new_disease.id,
            precautions=json.dumps(disease_data['precautions']),
            doctor_notes=disease_data.get('notes', ''),
            approved_by=data.get('doctor_name', 'Unknown')
        )
        db.session.add(new_precaution)
        
        diagnosis.disease_id = new_disease.id
    
    db.session.commit()
    refresh_ai_components()
    
    return jsonify({
        'status': 'confirmed',
        'diagnosis_id': diagnosis.id,
        'disease_id': diagnosis.disease_id
    })

def generate_ai_analysis(symptoms, suggestions):
    disease_names = [s['name'] for s in suggestions]
    disease_text = ", ".join(disease_names) if disease_names else "No specific match found"
    
    prompt = f"""Analyze these symptoms: {symptoms}
    Top suggestions: {disease_text}
    Provide differential diagnosis considering:
    1. Likelihood of each suggested condition
    2. Possible missing symptoms to ask about
    3. Recommended initial tests"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical diagnosis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

# Route to get a list of all diseases
@app.route('/diagnosis/diseases', methods=['GET'])
def get_diseases():
    diseases = Disease.query.filter_by(is_approved=True).all()
    return jsonify([{
        'id': d.id,
        'name': d.name,
        'symptoms': d.symptoms
    } for d in diseases])

# Route to get a patient's diagnosis history
@app.route('/diagnosis/history/<patient_id>', methods=['GET'])
def get_diagnosis_history(patient_id):
    diagnoses = SymptomDiagnosis.query.filter_by(patient_id=patient_id).order_by(SymptomDiagnosis.created_at.desc()).all()
    result = []
    
    for diagnosis in diagnoses:
        disease = Disease.query.get(diagnosis.disease_id) if diagnosis.disease_id else None
        result.append({
            'id': diagnosis.id,
            'symptoms': diagnosis.symptoms,
            'disease': disease.name if disease else None,
            'doctor_notes': diagnosis.doctor_notes,
            'is_approved': diagnosis.is_approved,
            'created_at': diagnosis.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(result)

# Symptom Diagnosis Routes


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patients', methods=['GET'])
def patient_list():
    patients = Patient.query.order_by(Patient.last_name).all()
    return render_template('patient.html', patients=patients)

# View Routes - HTML Templates
# Change this route definition
@app.route('/patient/<string:patient_id>/symptom-diagnosis', methods=['GET'])
def symptom_diagnosis(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template('diagnosis.html', patient=patient)
@app.route('/patient/<string:patient_id>', methods=['GET'])
def patient_detail(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    analyses = Analysis.query.filter_by(patient_id=patient_id).order_by(Analysis.created_at.desc()).all()
    return render_template('patient_detail.html', patient=patient, analyses=analyses)

@app.route('/analysis/<int:analysis_id>', methods=['GET'])
def analysis_detail(analysis_id):
    # Get analysis details
    analysis = Analysis.query.get_or_404(analysis_id)
    return render_template('analysis_detail.html', analysis=analysis)

@app.route('/new_analysis/<string:patient_id>', methods=['GET'])
def new_analysis_form(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template('new_analysis.html', patient=patient)

@app.route('/api/patients/<string:patient_id>/analyses', methods=['GET'])
def get_patient_analyses(patient_id):
    analyses = Analysis.query.filter_by(patient_id=patient_id).order_by(Analysis.created_at.desc()).all()
    return jsonify([{
        'id': a.id,
        'image_type': a.image_type,
        'prediction': a.prediction,
        'confidence': a.confidence,
        'created_at': a.created_at.strftime('%Y-%m-%d %H:%M:%S')
    } for a in analyses])

# Additional Utilities/Views

# Patient detail page
 

# Route to refresh AI components
@app.route('/admin/refresh-ai', methods=['POST'])
def admin_refresh_ai():
    if 'admin' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    refresh_ai_components()
    return jsonify({'status': 'success', 'message': 'AI components refreshed successfully'})


if __name__ == '__main__':
    app.run(port=5001, debug=True)