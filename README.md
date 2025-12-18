# 🐶🐱 Dog vs Cat Image Classification using CNN (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images of **dogs and cats**. The model is trained from scratch and covers the complete deep learning workflow including data preprocessing, training, validation, and inference.

---

## 📌 Project Overview
- **Task**: Binary image classification (Dog vs Cat)
- **Framework**: PyTorch
- **Model**: Custom CNN
- **Training Platform**: Google Colab
- **Dataset**: Kaggle Dogs vs Cats

---

## 🧠 Model Architecture
- Convolutional layers with Batch Normalization and ReLU activation
- MaxPooling layers for spatial downsampling
- Fully connected layers with Dropout for regularization
- Output layer with 2 neurons (Dog, Cat)

---

## ⚙️ Training Configuration
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Input Image Size**: 128 × 128
- **Batch Size**: 32
- **Epochs**: 25
- **Normalization**: ImageNet mean and standard deviation

---

## 📂 Dataset
This project uses the **Dogs vs Cats dataset from Kaggle**.

🔗 https://www.kaggle.com/datasets/vishallazrus/cat-vs-dog-image-classification-making-prediction/discussion?sort=hotness

The dataset is **not included** in this repository due to size and licensing restrictions.

### Expected directory structure:

data/
├── train/
│ ├── cat/
│ └── dog/
└── val/
├── cat/
└── dog/


---

## 📁 Project Structure

dog-vs-cat-cnn/
├── model.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md


---

## 🚀 How to Run

 Install dependencies
```bash
pip install -r requirements.txt
```

python train.py

dogvscat.pth

class_to_idx.pth

from predict import predict_image

label, confidence = predict_image("sample.jpg")
print(label, confidence)
