# AI Face Detection Project

This project showcases a custom-built deep learning model that classifies images as either real human faces or AI-generated faces. I developed this model using transfer learning with [ResNet-50](https://blog.roboflow.com/what-is-resnet-50/) as a foundation, then customized and trained it to achieve exceptional accuracy in distinguishing between real and artificial images.

## Data Source

[detect-ai-generated-faces-high-quality-dataset](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset)

## Project Structure

```
detect-ai-faces/
├── data/
│   ├── real/         # Real face images
│   └── ai/           # AI-generated face images
├── main/
│   ├── train.py      # Training script
│   └── evaluate.py   # Evaluation script
├── models/           # Saved model weights
│   └── face_classifier.pth
├── templates/        # Frontend templates
│   └── index.html    # Web interface
├── app.py           # Flask application server
├── plots/           # Visualization outputs
│   ├── confusion_matrix.png
│   └── training_history.png
└── requirements.txt  # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/s6hib/detect-ai-faces.git
cd detect-ai-faces
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Frontend

To run the web interface locally:

1. Ensure your virtual environment is activated
2. Start the Flask development server:
```bash
python app.py
```
3. Open your browser and navigate to `http://localhost:5000`
4. Upload any face image to test if it's real or AI-generated

The web interface provides an intuitive way to interact with my model, offering real-time predictions with confidence scores for any uploaded image.

### Training

To train the model:

```bash
python main/train.py
```

This will:
- Load and preprocess the dataset
- Train the model using transfer learning
- Save the trained model to `models/face_classifier.pth`
- Generate training history plots

### Evaluation

To evaluate the trained model:

```bash
python main/evaluate.py
```

This will:
- Load the trained model
- Evaluate it on the test set
- Generate a confusion matrix
- Print detailed metrics including accuracy, precision, recall, and F1-score
- Identify high-confidence mistakes

## Model Architecture

The model uses ResNet-50 pretrained on ImageNet as the backbone, with the following modifications:
- Frozen feature extraction layers
- Custom classification head with:
  - Linear layer (2048 → 256)
  - ReLU activation
  - Dropout (0.5)
  - Final linear layer (256 → 2)

### Data Preprocessing

Images are preprocessed using the following transformations:
- Resize to 224×224 pixels
- Random horizontal flip
- Random rotation (±10 degrees)
- Color jitter (brightness and contrast)
- Normalization using ImageNet statistics

## Dataset

The dataset consists of:
- Real face images: PNG format
- AI-generated face images: JPG format
- Balanced class distribution
- Train/validation split: 80%/20%

## Performance

The model achieves exceptional performance in distinguishing between real and AI-generated faces:

### Overall Metrics
- **Accuracy**: 99.78%
- **Visualization**: Performance metrics are visualized in:
  - `training_history.png`: Training and validation loss/accuracy curves
  - `confusion_matrix.png`: Confusion matrix visualization

### Detailed Classification Report
```
              precision    recall  f1-score   support

        Real       1.00      1.00      1.00      2202
AI-Generated       1.00      0.99      1.00      1001

    accuracy                           1.00      3203
   macro avg       1.00      1.00      1.00      3203
weighted avg       1.00      1.00      1.00      3203
```

### Key Insights
- Perfect precision (1.00) for both real and AI-generated face detection
- Near-perfect recall, with AI-generated faces having a slightly lower recall (0.99)
- Balanced performance across classes despite uneven support (2202 real vs 1001 AI-generated samples)

### Edge Cases
The model occasionally makes high-confidence mistakes:
- Example: AI-Generated face misclassified as Real with 99.96% confidence

This suggests that while the model's overall performance is exceptional, there are still some AI-generated images that can be incredibly convincing, even to our highly accurate classifier.

## Next Steps

The next phase of this project would be deploying the service to make it publicly accessible.
