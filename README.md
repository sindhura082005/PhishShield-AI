# 🛡️ PhishShield AI  
### Real-Time Phishing URL Detection System (PyTorch + FastAPI)

🚀 **Live Demo:** 
📄 **API Docs:** http://127.0.0.1:8000/docs#/

PhishShield AI is an end-to-end machine learning and backend security system designed to detect phishing URLs in real-time.  
The system leverages engineered URL intelligence features and a deep neural network classifier to provide scalable REST API-based threat detection.

Built to simulate production-level cybersecurity infrastructure, the project demonstrates ML model development, performance optimization, and API deployment using FastAPI.

---

## 🚀 Key Engineering Highlights

- Designed and implemented a real-time phishing detection REST API
- Engineered 15+ lexical and reference-based URL security features
- Developed a Deep Neural Network classifier using PyTorch
- Applied class-weighted training to address data imbalance
- Optimized decision threshold to improve recall for cybersecurity use cases
- Built scalable backend inference pipeline using FastAPI
- Structured modular architecture for maintainability and extensibility

---

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| ROC-AUC | **0.8818** |
| Precision | **0.7797** |
| Recall | **0.8206** |
| F1 Score | **0.7997** |
| Optimized Threshold | **0.39** |

The model is optimized for high recall to reduce missed phishing attacks in cybersecurity scenarios.

---

## 🧠 Feature Engineering

The system extracts 15+ handcrafted lexical and reference-based features from URLs, including:

- HTTPS presence
- IP address detection in URL
- URL length
- Host length
- Path depth
- Number of dots in host
- Number of digits in URL
- Query parameter count
- Suspicious symbol detection (@, fragments)
- Reference phishing hit ratio
- Benign reference hits
- Host anomaly indicators

Features are normalized using `StandardScaler` before model inference.

---

## 🏗️ Model Architecture

Neural Network Architecture:

Input Layer  
→ 128 Units (ReLU + BatchNorm + Dropout)  
→ 64 Units (ReLU + BatchNorm + Dropout)  
→ 32 Units (ReLU)  
→ Output Layer (Binary Classification)

Training includes:

- Class-weighted Binary Cross Entropy Loss
- Early stopping
- Learning rate scheduling
- Threshold tuning for F1-score optimization

---

## ⚙️ Tech Stack

- Python
- PyTorch
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Uvicorn
- Joblib

---

## 📡 API Endpoints

### Health Check

```
GET /health
```

Returns server status.

---

### Predict URL

```
POST /predict
```

#### Example Request:

```json
{
  "url": "http://paypal-login-security-check.com"
}
```

#### Example Response:

```json
{
  "label": "phish",
  "score": 0.82,
  "details": { ... }
}
```
**Score Interpretation:**  
The `score` field represents the predicted phishing probability (after sigmoid activation).  
A decision threshold of **0.39** (optimized during training) is applied for classification:
- Score ≥ 0.39 → `"phish"`
- Score < 0.39 → `"benign"`

This threshold was selected to maximize F1-score and improve phishing recall.

---

## 🔧 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/sindhura082005/PhishShield-AI.git
cd PhishShield-AI
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

```bash
python -m src.serve.models.train_ref_tabmnet
```

Artifacts generated:

- model.pt
- scaler.pkl
- feature_keys.json

Saved in:
```
src/serve/artifacts/
```

---

## 🚀 Running the API Server

```bash
uvicorn src.serve.app:app --reload
```

Access interactive API documentation:

```
http://127.0.0.1:8000/docs
```

---

## 🔐 Security Focus

PhishShield AI prioritizes:

- High phishing detection recall
- Balanced precision-recall tradeoff
- Scalable inference pipeline
- Modular feature extraction
- Real-time API deployment

---

## 📈 Future Enhancements

- Transformer-based URL embeddings
- Docker containerization
- Cloud deployment (AWS / GCP)
- CI/CD integration
- Real-time threat intelligence API integration

---

## 👩‍💻 Author

Developed as an end-to-end Machine Learning and Backend Security project demonstrating:

- ML pipeline design
- Feature engineering for cybersecurity
- Model optimization & threshold tuning
- Production API deployment using FastAPI

---
