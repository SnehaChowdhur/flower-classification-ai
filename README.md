# 🌸 Native Indian Flower Classification AI

### An End-to-End Deep Learning Application for Identifying Selected Native Indian Flowers

<p align="center">

**🌺 Identify • 🤖 Predict • 📊 Analyze**

</p>

<p align="center">

A complete AI-powered image classification system built with <strong>TensorFlow • MobileNetV2 • FastAPI • HTML • CSS • JavaScript</strong>

</p>

---

## 🌿 Overview

**Native Indian Flower Classification AI** is an end-to-end machine learning application designed to identify a **selected collection of 13 native Indian flower classes from images**.

The system combines a deep learning model with a FastAPI backend and an interactive web interface.

A user can simply upload a photograph of a flower, and the application:

```text
📷 Upload Image
      ↓
🌐 Web Frontend
      ↓
⚡ FastAPI Backend
      ↓
🧠 MobileNetV2 Model
      ↓
🔍 Image Classification
      ↓
🌸 Flower Name + Confidence
```

The project demonstrates how a trained machine learning model can be transformed into a **usable web application**, rather than remaining only as a standalone training script.

---

# ✨ Key Features

### 🤖 AI-Powered Classification

Uses **Transfer Learning with MobileNetV2** to classify flower images.

### 🌺 13 Selected Indian Flower Classes

The current model is trained specifically on a selected collection of **13 Indian flower classes**.

### 📷 Image Upload

Users can upload flower images directly through the web interface.

Supported formats:

```text
JPG
JPEG
PNG
```

### ⚡ FastAPI Backend

The trained model is exposed through a REST API using FastAPI.

### 🌐 Interactive Web Interface

The frontend provides:

* Image upload
* Image preview
* Prediction result
* Confidence score
* Loading state
* User-friendly interface

### 🔗 Complete ML Application Pipeline

The project connects:

**Machine Learning → Backend API → Frontend → User**

---

# 🌸 Supported Flower Classes

The current model recognizes these **13 selected flower classes**:

|  # | Flower            |
| -: | ----------------- |
| 01 | 🌼 Chandramallika |
| 02 | 🌸 Cosmos Phul    |
| 03 | 🌼 Gada           |
| 04 | 🌹 Golap          |
| 05 | 🌺 Jaba           |
| 06 | 🌸 Kagoj Phul     |
| 07 | 🌼 Noyontara      |
| 08 | 🌼 Radhachura     |
| 09 | 🌺 Rangan         |
| 10 | 🌸 Salvia         |
| 11 | 🌺 Sandhyamani    |
| 12 | 🌻 Surjomukhi     |
| 13 | 🌸 Zinnia         |

> **Important:** The model is designed for these selected classes. It is **not a general-purpose flower identifier** and should not be expected to correctly identify flowers outside these classes.

---

# 📊 Dataset

The model was trained using a dataset containing:

| Property          |     Value |
| ----------------- | --------: |
| Total Images      | **7,989** |
| Number of Classes |    **13** |
| Training Images   | **5,592** |
| Validation Images | **1,198** |
| Testing Images    | **1,199** |

The dataset was split using **stratified sampling**, helping preserve the relative distribution of flower classes across training, validation, and testing sets.

---

# 🧠 Machine Learning

## Transfer Learning with MobileNetV2

Instead of building and training a CNN completely from scratch, this project uses **MobileNetV2 pretrained on ImageNet** as the feature extraction backbone.

This provides a strong starting point for image recognition while keeping the model relatively lightweight.

### Model Architecture

```text
                    INPUT IMAGE
                         │
                         ▼
                ┌─────────────────┐
                │ Image Resize     │
                │ 224 × 224 × 3   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Normalization   │
                │ Pixel → 0–1     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Data            │
                │ Augmentation    │
                │                 │
                │ Flip            │
                │ Rotation        │
                │ Zoom            │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   MobileNetV2   │
                │ ImageNet Weights│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Global Average  │
                │ Pooling         │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Dropout 20%     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Dense Layer     │
                │ 13 Classes      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Softmax         │
                └────────┬────────┘
                         │
                         ▼
                 🌸 PREDICTION
```

---

# ⚙️ Model Configuration

| Parameter          | Configuration                   |
| ------------------ | ------------------------------- |
| Architecture       | MobileNetV2                     |
| Learning Method    | Transfer Learning               |
| Pretrained Weights | ImageNet                        |
| Input Size         | `224 × 224 × 3`                 |
| Output Classes     | `13`                            |
| Optimizer          | Adam                            |
| Learning Rate      | `0.001`                         |
| Loss               | Sparse Categorical Crossentropy |
| Batch Size         | `32`                            |
| Epochs             | `5`                             |
| Dropout            | `0.2`                           |
| Data Augmentation  | Flip + Rotation + Zoom          |

---

# 📈 Performance

After **5 training epochs**, the model achieved:

| Metric              |     Result |
| ------------------- | ---------: |
| Training Accuracy   | **97.19%** |
| Validation Accuracy | **96.66%** |
| Test Accuracy       | **95.25%** |
| Test Loss           | **0.1630** |

### 🎯 Test Accuracy

```text
███████████████████░ 95.25%
```

The reported test accuracy is based on the project's held-out test dataset.

> **Note:** Test-set accuracy does not guarantee the same performance on every real-world photograph. Image quality, lighting, background, camera angle, and flowers outside the supported classes can affect predictions.

---

# 🏗️ Application Architecture

```text
                       👤 USER
                          │
                          │
                    Upload Flower
                          │
                          ▼
                ┌──────────────────┐
                │   WEB FRONTEND   │
                │                  │
                │ HTML             │
                │ CSS              │
                │ JavaScript       │
                └────────┬─────────┘
                         │
                         │ HTTP POST
                         ▼
                ┌──────────────────┐
                │    FASTAPI       │
                │    BACKEND       │
                └────────┬─────────┘
                         │
                         │ Preprocess
                         ▼
                ┌──────────────────┐
                │   ML MODEL       │
                │   MobileNetV2    │
                └────────┬─────────┘
                         │
                         │ Prediction
                         ▼
                ┌──────────────────┐
                │   PREDICTION     │
                │                  │
                │ Flower Name      │
                │ Confidence       │
                └────────┬─────────┘
                         │
                         ▼
                       👤 USER
```

---

# 🔄 How the Application Works

### 1️⃣ Image Upload

The user selects a flower image from the website.

### 2️⃣ Frontend Request

JavaScript sends the image to the FastAPI server using a `POST` request.

```text
POST /predict
```

### 3️⃣ Backend Processing

FastAPI receives the uploaded image and prepares it for the model.

The image is:

* Read
* Converted to RGB
* Resized to `224 × 224`
* Converted to an array
* Normalized
* Expanded into a batch

### 4️⃣ Model Prediction

The processed image is passed to the trained MobileNetV2 classifier.

### 5️⃣ Classification

The model produces probabilities for all 13 supported flower classes.

The class with the highest probability becomes the predicted flower.

### 6️⃣ Result

The API returns the prediction to the frontend.

Example:

```json
{
  "flower": "Golap",
  "confidence": 95.42
}
```

### 7️⃣ User Interface

The website displays the predicted flower and confidence score.

---

# 🖥️ Web Application

## Main Interface

Add a screenshot of your website here:

```text
screenshots/homepage.png
```

Once you create the screenshot, replace this section with:

```markdown
![BloomAI Homepage](screenshots/homepage.png)
```

---

## 🔍 Prediction Result

Add a screenshot showing an actual prediction:

```text
screenshots/prediction.png
```

Then use:

```markdown
![Prediction Result](screenshots/prediction.png)
```

---

# 📁 Project Structure

```text
native-indian-flower-classification/
│
├── static/
│   ├── index.html          # Web page structure
│   ├── style.css           # Frontend styling
│   └── script.js           # Frontend logic & API communication
│
├── app.py                  # FastAPI backend
├── train_model.py          # Dataset preparation & model training
│
├── flower_classifier.keras # Trained TensorFlow model
├── class_names.npy         # Flower class mapping
│
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from Git
└── README.md               # Project documentation
```

---

# 🧩 Technologies

## 🧠 Machine Learning

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Pandas
* Scikit-learn

## ⚡ Backend

* FastAPI
* Uvicorn
* Python Multipart
* Pillow

## 🌐 Frontend

* HTML5
* CSS3
* JavaScript
* Fetch API

---

# 🚀 Installation

## Prerequisites

Make sure you have installed:

* Python 3.x
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/SnehaChowdhur/flower-classification-ai.git
```

Enter the project directory:

```bash
cd flower-classification-ai
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The server should start at:

```text
http://127.0.0.1:8000
```

Open this address in your browser.

---

# 🔌 API

## `POST /predict`

The main prediction endpoint accepts an image using multipart form data.

### Request

```text
POST /predict
Content-Type: multipart/form-data
```

### Example Response

```json
{
  "flower": "Chandramallika",
  "confidence": 97.31
}
```

---

# 🧪 Training the Model

To train the model yourself, create the following dataset structure:

```text
DATASET/
│
├── Chandramallika/
├── Cosmos Phul/
├── Gada/
├── Golap/
├── Jaba/
├── Kagoj Phul/
├── Noyontara/
├── Radhachura/
├── Rangan/
├── Salvia/
├── Sandhyamani/
├── Surjomukhi/
└── Zinnia/
```

Then run:

```bash
python train_model.py
```

The training script creates:

```text
flower_classifier.keras
class_names.npy
```

---

# 🛡️ Project Limitations

This system has been designed for **13 selected flower classes**.

It should not be treated as a universal botanical identification system.

Predictions can become less reliable when:

* The image is blurry.
* The flower occupies only a small portion of the image.
* Lighting is poor.
* The flower is partially hidden.
* The image contains multiple flowers.
* The flower is outside the 13 trained classes.
* The visual appearance differs significantly from the training data.

---

# 🔮 Future Improvements

The project can be extended in several directions:

* [ ] Add more native Indian flower species
* [ ] Fine-tune MobileNetV2 layers
* [ ] Improve dataset diversity
* [ ] Add confusion matrix visualization
* [ ] Add precision, recall and F1-score
* [ ] Add top-3 predictions
* [ ] Add prediction history
* [ ] Add drag-and-drop uploading
* [ ] Add camera capture
* [ ] Deploy FastAPI backend
* [ ] Deploy web frontend
* [ ] Add database integration
* [ ] Optimize model for mobile devices
* [ ] Build a Flutter mobile application
* [ ] Add an "unknown flower" detection mechanism

---

# 🎓 What This Project Demonstrates

This project demonstrates practical experience with the complete machine learning application lifecycle:

```text
Dataset
   ↓
Data Collection
   ↓
Data Cleaning
   ↓
Label Encoding
   ↓
Train / Validation / Test Split
   ↓
Image Preprocessing
   ↓
Data Augmentation
   ↓
Transfer Learning
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
FastAPI Backend
   ↓
REST API
   ↓
Frontend Integration
   ↓
User Prediction
```

Rather than stopping at model training, the project takes the additional step of integrating the trained model into a usable web application.

---

# 🌱 Why This Project?

India has an enormous diversity of flowers, many of which are deeply connected with local culture, traditions, gardens, festivals, and everyday life.

This project explores how computer vision can be used to recognize a **selected collection of Indian flowers** through photographs.

The long-term goal is to expand the system into a broader educational and plant-recognition platform.

---

# 👩‍💻 Author

## Sneha Chowdhury

**B.Tech — Computer Science & Engineering**

Interested in:

* 🤖 Artificial Intelligence
* 🧠 Machine Learning
* 📱 Application Development
* 🌐 Full-Stack Development

---

# ⭐ Project Highlights

<p align="center">

| 🌺 Classes | 📷 Images | 🎯 Accuracy |
| :--------: | :-------: | :---------: |
|   **13**   | **7,989** |  **95.25%** |

|     🧠 Model    |  ⚡ Backend  |   🌐 Frontend   |
| :-------------: | :---------: | :-------------: |
| **MobileNetV2** | **FastAPI** | **HTML/CSS/JS** |

</p>

---

## 🌸 From Flower Image to AI Prediction

```text
📷
Flower Photograph
       │
       ▼
🧠
Deep Learning Model
       │
       ▼
⚡
FastAPI
       │
       ▼
🌐
Web Interface
       │
       ▼
🌺
Predicted Flower
```

---

## ⭐ If you find this project interesting

Feel free to explore the repository, experiment with the model, and extend the system with additional Indian flower species.

---

<p align="center">

### 🌸 Built with Python, TensorFlow, FastAPI & curiosity.

**Native Indian Flower Classification AI**

</p>
