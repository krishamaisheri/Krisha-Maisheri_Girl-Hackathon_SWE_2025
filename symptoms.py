import pandas as pd
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sklearn.metrics.pairwise import cosine_similarity

# SQLAlchemy setup
Base = declarative_base()

# Define the Disease table
class Disease(Base):
    __tablename__ = "diseases"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    symptoms = Column(Text, nullable=False)

# Define the Precaution table
class Precaution(Base):
    __tablename__ = "precautions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    disease = Column(String, nullable=False)
    precaution_1 = Column(String)
    precaution_2 = Column(String)
    precaution_3 = Column(String)
    precaution_4 = Column(String)

# Create the database engine and session
engine = create_engine("sqlite:///medical_db.sqlite")
Session = sessionmaker(bind=engine)
session = Session()

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Load the datasets
try:
    # Load symptoms dataset
    disease_symptom_df = pd.read_csv("DiseaseAndSymptoms.csv")  # Ensure this file exists
    # Load precautions dataset
    disease_precaution_df = pd.read_csv("Disease precaution.csv")  # Ensure this file exists
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the CSV files are in the correct directory.")
    exit()

# Debugging: Print column names to verify
print("Columns in disease_symptom_df:", disease_symptom_df.columns.tolist())
print("Columns in disease_precaution_df:", disease_precaution_df.columns.tolist())

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Store data in the database
def store_data():
    # Clear existing data to avoid conflicts
    session.query(Disease).delete()
    session.query(Precaution).delete()
    session.commit()

    # Store disease symptoms
    symptom_dict = {}  # Dictionary to combine symptoms for duplicate diseases
    for _, row in disease_symptom_df.iterrows():
        disease_name = row["Disease"]
        # Dynamically generate symptom columns based on the actual column names
        symptom_columns = [col for col in disease_symptom_df.columns if col.startswith("Symptom")]
        symptoms = ", ".join([str(row[col]) for col in symptom_columns if pd.notna(row[col])])
        if not symptoms:  # Handle empty symptoms
            symptoms = "No symptoms available"
        
        # Combine symptoms for duplicate diseases
        if disease_name in symptom_dict:
            symptom_dict[disease_name] += f"; {symptoms}"
        else:
            symptom_dict[disease_name] = symptoms
    
    # Insert combined symptoms into the database
    for disease_name, symptoms in symptom_dict.items():
        disease = Disease(name=disease_name, symptoms=symptoms)
        session.add(disease)
    
    # Store disease precautions
    for _, row in disease_precaution_df.iterrows():
        precaution = Precaution(
            disease=row["Disease"],
            precaution_1=row.get("Precaution_1", ""),
            precaution_2=row.get("Precaution_2", ""),
            precaution_3=row.get("Precaution_3", ""),
            precaution_4=row.get("Precaution_4", "")
        )
        session.add(precaution)
    
    session.commit()

# Retrieve diseases and generate embeddings
def load_diseases():
    diseases = session.query(Disease).all()
    return [(disease.name, disease.symptoms) for disease in diseases]

# Initialize the database and store data
store_data()

# Load diseases and generate embeddings
diseases = load_diseases()
disease_names = [d[0] for d in diseases]
disease_symptom_texts = [d[1] for d in diseases]
embeddings = embedding_model.encode(disease_symptom_texts, convert_to_numpy=True)

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Find disease based on input symptoms
def find_disease(input_symptoms):
    input_embedding = embedding_model.encode([input_symptoms], convert_to_numpy=True)
    distances, indices = index.search(input_embedding, k=1)
    
    # Check similarity score
    similarity_threshold = 0.7  # Adjust this threshold as needed
    if distances[0][0] > similarity_threshold:
        return "No matching disease found."
    
    return disease_names[indices[0][0]]

# Get precautions for a disease
def get_precautions(disease_name):
    if disease_name == "No matching disease found.":
        return ["No specific precautions available."]
    
    precaution = session.query(Precaution).filter(Precaution.disease == disease_name).first()
    if precaution:
        return [precaution.precaution_1, precaution.precaution_2, precaution.precaution_3, precaution.precaution_4]
    return ["No specific precautions found."]

# OpenAI API Key
openai.api_key = "your_openai_api_key"

# Generate a response using OpenAI
def generate_response(disease, precautions, input_symptoms):
    prompt = f"""
    You are a diagnozing assistent for a Doctor, your responses are given to the Doctor to help diagnose the patient.
    A doctor has provided the following symptoms: {input_symptoms}.
    Based on analysis, the predicted disease is **{disease}**.
    Check if the predicted disease and the symptops match. If not, give response accordingly but don't mention that the prediction is wrong,
    instead either ask for more details or go with the user's response and predict the disease on your own.
    Here are some precautions to follow:
    - """ + "\n    - ".join(precautions)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a medical assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Example Usage
if __name__ == "__main__":
    input_symptoms = "fever, cough, sore throat"
    predicted_disease = find_disease(input_symptoms)
    precautions = get_precautions(predicted_disease)
    response = generate_response(predicted_disease, precautions, input_symptoms)
    print(response)