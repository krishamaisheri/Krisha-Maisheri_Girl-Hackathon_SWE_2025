# Updated MedicalAnalysisAgent class for Flask integration
import torch
import torchvision.transforms as transforms
import tensorflow as tf
from PIL import Image
import pytesseract
import numpy as np
import cv2
import os
import openai
from datetime import datetime
import random
import pdfplumber
import base64
from werkzeug.utils import secure_filename

class MedicalAnalysisAgent:
    def __init__(self, api_key=None):
        # Initialize OpenAI if API key is provided
        self.use_gpt = False
        if api_key:
            openai.api_key = api_key
            self.use_gpt = True
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.brain_model = None
        self.chest_model = None
        
        # Try loading models, set to testing mode if models not available
        self.testing_mode = False
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'brain_tumor_model.pth')
            self.brain_model = torch.load(model_path, map_location=self.device)
            self.brain_model.eval()
        except Exception as e:
            print(f"\nWarning: Brain tumor model not found: {str(e)}")
            print("Running in testing mode with simulated predictions.")
            self.testing_mode = True
        
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'pneumonia_detection_model.h5')
            self.chest_model = tf.keras.models.load_model(model_path)
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
        """Get prediction if image is provided"""
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
        """Generate medical analysis based on available data"""
        # Use GPT if available, otherwise fall back to template-based analysis
        if self.use_gpt:
            return self.generate_gpt_analysis(prediction, confidence, report_text, previous_records, image_type)
        else:
            return self.generate_template_analysis(prediction, confidence, report_text, previous_records, image_type)

    def generate_gpt_analysis(self, prediction, confidence, report_text, previous_records, image_type):
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
            print(f"Error in GPT analysis: {str(e)}")
            # Fall back to template analysis if GPT fails
            return self.generate_template_analysis(prediction, confidence, report_text, previous_records, image_type)

    def generate_template_analysis(self, prediction, confidence, report_text, previous_records, image_type):
        """Generate medical analysis using templates (fallback when GPT is unavailable)"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
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