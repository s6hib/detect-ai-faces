import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
def load_model():
    model = models.resnet50(pretrained=False)
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )
    
    model.load_state_dict(torch.load('models/face_classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Initialize model and transforms
model = load_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    """Predict if an image is AI-generated or real."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
    return {
        'prediction': 'AI-Generated' if prediction.item() == 1 else 'Real',
        'confidence': float(confidence.item()) * 100
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG)'})
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)