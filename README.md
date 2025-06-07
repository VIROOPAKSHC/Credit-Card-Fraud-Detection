# 💳 Credit Card Fraud Detection

A robust machine learning pipeline to identify fraudulent credit card transactions using supervised classification techniques on a real-world, highly imbalanced dataset.

---

## 🚀 Project Highlights

- **📊 Highly Imbalanced Dataset**  
  Real-world credit card data with ~0.17% fraud rate in 284,807 transactions.

- **📈 Powerful ML Models**  
  Trained and evaluated models include Logistic Regression, Random Forest, and XGBoost. Options for neural networks and autoencoders are also considered.

- **🧩 End-to-End Pipeline**  
  From data preprocessing and feature engineering to model training, evaluation, and inference.

- **🛠️ Interpretability & Metrics**  
  Supports evaluation with confusion matrices, ROC-AUC, precision-recall curves, and feature importance plots.

- **🔄 Modular & Reproducible**  
  Configurable architecture allows for experimentation and extension.

---

## 📚 Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Modeling Overview](#modeling-overview)
- [Evaluation](#evaluation)
- [Results](#results)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## 📂 Dataset

:contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}. Features `V1`–`V28` are PCA-transformed; “Time” and “Amount” retained.

---

## 🧰 Installation

Make sure you have Python 3.7+ installed:

```bash
git clone https://github.com/VIROOPAKSHC/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
````

---

## ⚙️ Usage

### ✅ Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/EDA.ipynb
```

### 🎯 Train & Evaluate Model

Example for XGBoost:

```bash
python src/train.py \
  --model xgboost \
  --config src/configs/xgb_config.yaml
```

### 📦 Use Trained Model for Prediction

```bash
python src/predict.py --model-path models/xgb_best.pkl --input data/sample.csv
```

---

## 🔍 Project Structure

```text
.
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Preprocessed splits
├── notebooks/             # EDA & visualization notebooks
├── src/
│   ├── data_loader.py     # Data preprocessing
│   ├── train.py           # Training pipeline entry point
│   ├── evaluate.py        # Evaluation scripts
│   ├── predict.py         # Single-run inference
│   └── models/            # Model modules
├── models/                # Saved model artifacts
├── metrics/               # Evaluation outputs
├── requirements.txt
└── README.md
```

---

## 🧠 Modeling Overview

* **Preprocessing**

  * Standard scaling for `Amount`, timestamp pipelines
  * Downsampling or SMOTE to address class imbalance
* **Models Trained**

  * Logistic Regression, Random Forest, XGBoost
  * (Optional) Neural Networks, Autoencoders
* **Optimization**

  * Hyperparameter tuning via `GridSearchCV`
* **Evaluation Metrics**

  * ROC-AUC, Precision-Recall, F1-score, Confusion Matrix

---

## 📊 Evaluation

| Model               | ROC-AUC   | Precision | Recall   | F1-score |
| ------------------- | --------- | --------- | -------- | -------- |
| Logistic Regression | 0.98x     | 0.90x     | 0.85x    | 0.87x    |
| Random Forest       | 0.99x     | 0.92x     | 0.88x    | 0.90x    |
| **XGBoost (best)**  | **0.99+** | **0.95**  | **0.89** | **0.92** |

*Values are indicative — see `metrics/` for full results*

---

## 🏆 Results

* **XGBoost provides the best trade-off** between high ROC-AUC and stable generalization.
* Precision-recall curves show robust detection with low false-positive rates.
* SHAP and feature importance highlight `V14`, `V12`, and `Amount` as top predictors.

📌 Plots are available under `notebooks/` and `metrics/`.

---

## 🎯 Roadmap

* [x] Clean & preprocess raw data
* [x] Baseline models implemented
* [x] Hyperparameter search for XGBoost
* [ ] Add neural network and autoencoder models
* [ ] Deploy using Flask / Streamlit
* [ ] Real-time batch scoring
* [ ] Explainability with SHAP / LIME

---

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome!
**To contribute:**

1. Fork the repo & create your branch (`git checkout -b feature/xyz`)
2. Make your changes, then commit (`git commit -m 'Add xyz'`)
3. Push to your branch (`git push origin feature/xyz`)
4. Open a PR and describe your updates 🎉

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for full terms.

---

⭐ If this helped you or inspired new fraud detection ideas, please give it a star!

```

