# 🚀 ML Inference Service

A production-oriented REST API for serving a PyTorch machine learning model — from trained artifacts to a publicly accessible cloud service.

This project demonstrates how to take a trained ML model beyond experimentation and turn it into a **reproducible, validated, containerized and deployable inference service**.

**PyTorch · FastAPI · Docker · GitHub Actions · Amazon ECR · AWS ECS/Fargate**

---

## 🎯 Overview

The service exposes a REST API for predicting **California housing prices** using a trained PyTorch regression model.

The main goal was not simply to train a model, but to build the engineering layer around it:

* 📦 Load a trained model and its preprocessing artifacts
* 🔍 Validate incoming data through Pydantic
* ⚙️ Reproduce the training preprocessing pipeline during inference
* 🌐 Serve predictions through FastAPI
* 🐳 Containerize the application with Docker
* 🔄 Automate the workflow with GitHub Actions
* ☁️ Build and publish the container image to Amazon ECR
* 🚢 Deploy the service to AWS ECS/Fargate
* 🔗 Expose the model through a public HTTPS API

The result is a complete ML inference pipeline that can be consumed through a standard HTTP endpoint.

---

## 🏗️ Architecture

### Application flow

```text
Client
  │
  │ HTTP Request
  ▼
FastAPI
  │
  ▼
Pydantic Validation
  │
  ▼
PredictionService
  │
  ├── Feature Mapping
  ├── Imputation
  ├── Feature Scaling
  ├── PyTorch Inference
  └── Target Inverse Scaling
  │
  ▼
PredictionSchema
  │
  ▼
JSON Response
```

### ☁️ Deployment flow

```text
GitHub
   │
   ▼
GitHub Actions
   │
   ├── Dependency installation
   ├── Application validation
   └── Docker build
        │
        ▼
   Amazon ECR
        │
        ▼
   AWS ECS/Fargate
        │
        ▼
   Public HTTPS API
```

This separation keeps the model-serving logic independent from HTTP concerns while allowing the same application to run locally, inside Docker, or in the cloud.

---

## 💡 Why this project?

A trained model is only one part of an ML system.

This project focuses on the transition from:

> **"I trained a model."**

to:

> **"I built a service that can actually serve that model."**

The project brings together concepts from **Machine Learning, Software Engineering and MLOps**, including:

* Model serialization and artifact management
* Data preprocessing
* API design
* Input/output validation
* Dependency injection
* Containerization
* CI/CD
* Cloud deployment
* ML inference architecture

---

## 🧠 Model

The model is a feed-forward neural network implemented with **PyTorch**.

```text
Input: 8 features
        │
        ▼
Linear(8 → 32)
        │
      ReLU
        │
        ▼
Linear(32 → 32)
        │
      ReLU
        │
        ▼
Linear(32 → 1)
        │
        ▼
Housing Price
```

### Input features

The API expects the following features:

| Feature              | Description              |
| -------------------- | ------------------------ |
| `longitude`          | Geographic longitude     |
| `latitude`           | Geographic latitude      |
| `housing_median_age` | Median age of houses     |
| `total_rooms`        | Total number of rooms    |
| `total_bedrooms`     | Total number of bedrooms |
| `population`         | Population               |
| `households`         | Number of households     |
| `median_income`      | Median income            |

### ⚙️ Preprocessing

The same preprocessing pipeline used during training is reproduced during inference:

```text
Input
  ↓
Median Imputation
  ↓
StandardScaler
  ↓
PyTorch Tensor
  ↓
Model Inference
  ↓
Inverse Target Scaling
  ↓
Prediction
```

The preprocessing objects are stored as separate artifacts so that inference uses the same transformations as training.

### 📦 Model artifacts

```text
artifacts/
├── housing_model.pt
├── imputer.joblib
├── scaler_x.joblib
├── scaler_y.joblib
└── metadata.json
```

The metadata file stores information about the trained model, preprocessing configuration, hyperparameters and evaluation metrics.

### 📊 Evaluation

The final model achieved approximately:

| Metric | Result |
| ------ | -----: |
| R²     |  ~0.78 |
| MAPE   |   ~20% |

The model is intended primarily as a demonstration of the **ML serving and deployment pipeline**, rather than as a production housing valuation system.

---

## 🌐 API

### `GET /health`

Health check endpoint used to verify that the service is running.

**Response:**

```json
{
  "status": "ok",
  "message": "The service is working fine."
}
```

### `POST /predict`

Receives housing features and returns the predicted house value.

**Request:**

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.3252
}
```

**Response:**

```json
{
  "prediction": 399200.4375
}
```

The API also provides automatically generated interactive documentation through FastAPI.

---

## 💻 Running locally

### 1. Clone the repository

```bash
git clone https://github.com/diego-marques-miranda/ml-inference-service.git
cd ml-inference-service
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

For development:

```bash
pip install -r requirements-dev.txt
```

### 4. Start the API

Run from the project root:

```bash
python -m uvicorn src.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 🐳 Running with Docker

The project includes a production-oriented Docker image containing the API, model and preprocessing artifacts.

### Build

```bash
docker build -t ml-inference-service .
```

### Run

```bash
docker run --name ml-inference-service -p 8000:8000 ml-inference-service
```

The API will then be available at:

```text
http://localhost:8000
```

---

## 🔄 CI/CD

The repository uses **GitHub Actions** to automate the application workflow.

The pipeline is divided into two stages:

```text
Push / Pull Request
        │
        ▼
      CI
        │
        ├── Install dependencies
        ├── Run project validation
        └── Build Docker image
        │
        ▼
     Push to main
        │
        ▼
     Deployment
        │
        ├── Authenticate with AWS using OIDC
        ├── Build Docker image
        ├── Tag image with commit SHA
        └── Push image to Amazon ECR
```

AWS authentication uses **GitHub Actions OIDC**, avoiding long-lived AWS access keys inside GitHub.

---

## ☁️ Cloud Deployment

The application is deployed using:

```text
Amazon ECR
      │
      ▼
AWS ECS
      │
      ▼
AWS Fargate
```

The container runs as an ECS service with:

* ⚡ 1 vCPU
* 🧠 2 GB memory
* 🔒 Public HTTPS endpoint
* ❤️ `/health` health check
* 📈 CPU-based autoscaling
* 📝 CloudWatch logging

The deployed service was validated through both `/health` and `/predict`, confirming that the same model artifacts used locally can be loaded and served in the cloud.

---

## 📁 Project Structure

```text
ml-inference-service/
│
├── artifacts/
│   ├── housing_model.pt
│   ├── imputer.joblib
│   ├── scaler_x.joblib
│   ├── scaler_y.joblib
│   └── metadata.json
│
├── src/
│   ├── models/
│   │   └── housing.py
│   │
│   ├── loaders/
│   │   └── artifacts.py
│   │
│   ├── services/
│   │   └── prediction.py
│   │
│   ├── routes/
│   │   └── prediction.py
│   │
│   ├── schemas/
│   │   └── prediction.py
│   │
│   └── main.py
│
├── tests/
│
├── train.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── requirements-train.txt
```

---

## 🛠️ Tech Stack

| Category           | Technology          |
| ------------------ | ------------------- |
| Language           | Python 3.13         |
| ML                 | PyTorch             |
| Data Processing    | NumPy, scikit-learn |
| API                | FastAPI             |
| Validation         | Pydantic            |
| Serialization      | Joblib, PyTorch     |
| Testing            | Pytest              |
| Containerization   | Docker              |
| CI/CD              | GitHub Actions      |
| Container Registry | Amazon ECR          |
| Cloud              | AWS ECS / Fargate   |
| Monitoring / Logs  | Amazon CloudWatch   |

---

## 🔑 Key Engineering Decisions

### Dependency injection

The `PredictionService` receives the model and preprocessing artifacts through its constructor instead of loading them on every request.

```text
Application startup
       │
       ├── Load model
       ├── Load imputer
       ├── Load X scaler
       └── Load y scaler
                │
                ▼
       PredictionService
                │
          Handles requests
```

This keeps artifact loading outside the request lifecycle and separates infrastructure concerns from prediction logic.

### Canonical feature ordering

The API receives named fields, but the model requires a specific feature order.

The service explicitly maps the validated request into the canonical training order before preprocessing.

This prevents a subtle but critical class of ML serving errors: **feeding valid values to the model in the wrong order.**

### Training/serving consistency

The model is not served independently from its preprocessing pipeline.

The trained artifacts include:

* model
* imputer
* feature scaler
* target scaler
* metadata

This makes the inference pipeline reproduce the transformations expected by the model.

---

## ⚠️ Limitations

This project is designed as an **ML engineering and deployment demonstration**.

Current limitations include:

* The model is trained on the California Housing dataset.
* The model is not intended for real-world housing valuation.
* The service currently exposes a single model.
* Authentication is not implemented.
* The API is designed around synchronous inference.
* The deployment uses a relatively simple ECS/Fargate configuration.

These limitations are intentional and keep the project focused on the core objective: demonstrating an end-to-end ML inference workflow.

---

## 🚀 What this project demonstrates

By completing this project, the following workflow is covered:

```text
Machine Learning
      ↓
Model Training
      ↓
Artifact Management
      ↓
Inference Pipeline
      ↓
REST API
      ↓
Containerization
      ↓
CI/CD
      ↓
Cloud Deployment
      ↓
Public ML Service
```

The project therefore serves as a practical example of how **Machine Learning and Software Engineering come together to build a deployable AI system.**

---

## 🔮 Future Improvements

Potential extensions include:

* API authentication
* Model versioning
* Monitoring and observability
* Model performance monitoring
* Automated deployment from GitHub Actions to ECS
* Support for multiple model versions
* More advanced inference and deployment strategies

---

## 👨‍💻 Author

**Diego Marques Miranda**

AI/ML & Data Science · Deep Learning · Data Engineering · Python

[GitHub](https://github.com/diego-marques-miranda)
