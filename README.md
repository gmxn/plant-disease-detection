# 🌿 Plant Disease Detection

An end-to-end AI/ML project that detects plant diseases from leaf images using deep learning.

## 📊 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 51.56% | 50.67% |
| Custom CNN | 33.20% | 31.91% |
| **MobileNetV2** ⭐ | **67.88%** | **67.01%** |

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `task2_Dataset_EDA.ipynb` | Dataset exploration & analysis |
| `task3_feature_engineering.ipynb` | Preprocessing & augmentation pipeline |
| `Task4_Model_Training.ipynb` | Training 3 models |
| `Task5_model_evaluation.ipynb` | Evaluation & comparison |
| `Task6_Hyperparameter_Tuning.ipynb` | Manual hyperparameter search |
| `Task8_documentation.ipynb` | Full project documentation |
| `main.py` | FastAPI deployment |
| `test_api.py` | API tests |

## 🗄️ Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- 54,305 images across 39 classes
- 14 crop species
- All images: 256×256 RGB JPEGs

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn fastapi uvicorn pillow numpy pandas matplotlib seaborn
```

### 2. Download dataset
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
```

### 3. Run the notebooks in order
- Task2 → Task3 → Task4 → Task5 → Task6 → Task8

### 4. Start the API
```bash
cd plant_disease_api
uvicorn main:app --reload
# Visit http://127.0.0.1:8000/docs
```

## 🛠️ Tech Stack

- **Python** 3.10+
- **TensorFlow / Keras** — CNN & MobileNetV2
- **Scikit-learn** — Logistic Regression & metrics
- **FastAPI** — REST API deployment
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — Visualization

## 📁 Models

- `cnn_best.keras` — Custom CNN
- `mobilenet_tuned.keras` — MobileNetV2 (best model)

## 👤 Author

**gmxn** — AI/ML Internship Project 2026
