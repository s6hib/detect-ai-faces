# AI Face Detection Project

This project implements a deep learning model to classify images as either real human faces or AI-generated faces. It uses transfer learning with a pre-trained ResNet-50 model to achieve high accuracy in distinguishing between real and artificial images.

## Project Structure

```
detect-ai-faces/
├── app/
│   ├── train.py      # Training script
│   └── evaluate.py   # Evaluation script
├── data/
│   ├── real/         # Real face images
│   └── ai/           # AI-generated face images
├── models/           # Saved model weights
└── requirements.txt  # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/detect-ai-faces.git
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

### Training

To train the model:

```bash
python app/train.py
```

This will:
- Load and preprocess the dataset
- Train the model using transfer learning
- Save the trained model to `models/face_classifier.pth`
- Generate training history plots

### Evaluation

To evaluate the trained model:

```bash
python app/evaluate.py
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

The model's performance metrics will be saved in:
- `training_history.png`: Training and validation loss/accuracy curves
- `confusion_matrix.png`: Confusion matrix visualization
- Console output: Detailed classification report

## License

[Your chosen license]
