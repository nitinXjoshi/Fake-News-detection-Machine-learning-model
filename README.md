# 🚀 Fake News Detection on WELFake Dataset
### TF-IDF + Meta Features + LightGBM (12GB RAM Optimized)

> A high-performance, memory-efficient Fake News Detection pipeline built using Sparse TF-IDF and LightGBM.  
> Designed to run safely on Google Colab (12GB RAM).

---

## 🧠 Overview

This project builds a robust binary classifier to detect **Fake vs Real News** using:

- Advanced text preprocessing (lemmatization + optimized stopwords)
- Sparse TF-IDF (1–2 grams)
- Custom handcrafted meta features
- LightGBM with early stopping
- Memory-optimized architecture

The system is built to handle large-scale text data efficiently without exceeding Colab memory limits.

---

## 🏗️ Model Architecture

```
Raw Text
   ↓
Text Cleaning + Lemmatization
   ↓
TF-IDF (25K Sparse Features)
   +
Meta Features (4 Linguistic Signals)
   ↓
Feature Concatenation (Sparse Matrix)
   ↓
LightGBM Classifier
   ↓
Prediction
```

---

## 📂 Dataset

- **Training File:** `WELFake_Dataset.csv`
- **Test File:** `test.csv`

### Target Labels:
- `0` → Real News  
- `1` → Fake News  

---

## ⚙️ Feature Engineering

### 🔤 Text Features
- TF-IDF Vectorizer
- Max Features: 25,000
- N-grams: (1,2)
- min_df = 5
- max_df = 0.9
- Sublinear TF scaling
- English stopword removal
- WordNet Lemmatization
- Preserved negation words: *no, not, never*

### 📊 Meta Features

Additional handcrafted linguistic features:

- `char_len` → Total character length
- `word_len` → Total word count
- `caps_ratio` → Ratio of uppercase characters
- `punct_count` → Count of punctuation (!?.)

These features capture stylistic patterns often seen in misinformation content.

---

## 🌳 LightGBM Configuration

| Parameter | Value |
|------------|--------|
| n_estimators | 3000 |
| learning_rate | 0.03 |
| num_leaves | 64 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| class_weight | balanced |
| early_stopping | 100 rounds |

---

## 🧪 Training Strategy

- Train/Validation Split: **85% / 15%**
- Stratified sampling
- Evaluation Metrics:
  - Accuracy
  - Classification Report
  - Binary Log Loss

---

## 💾 Memory Optimization (12GB Safe)

- Sparse TF-IDF matrix
- Reduced feature space (25k)
- No cross-validation (single split)
- Explicit garbage collection
- Efficient LightGBM implementation

Built specifically to prevent Colab crashes.

---

## 🛠️ Installation

```bash
pip install lightgbm nltk scikit-learn
```

---

## ▶️ How to Run

1. Upload the dataset files to Google Colab:
   - `WELFake_Dataset.csv`
   - `test.csv`

2. Run the script or notebook.

3. Output generated:

```
submission_final.csv
```

Ready for submission.

---

## 📁 Project Structure

```
├── WELFake_Dataset.csv
├── test.csv
├── fake_news_model.ipynb
├── submission_final.csv
└── README.md
```

---

## 📊 Output

The final output file:

```
submission_final.csv
```

Contains:

| id | label |
|----|-------|
| Article ID | Predicted Class (0 or 1) |

---

## 🚀 Why This Approach?

- Handles large datasets efficiently
- Combines lexical + stylistic signals
- Fast training with strong performance
- Production-ready scalable pipeline
- Clean and interpretable architecture

---

## 🔮 Future Improvements

- Add cross-validation
- Hyperparameter tuning with Optuna
- Add sentiment polarity features
- Experiment with transformer embeddings
- Model ensembling

---

## 📜 License

This project is open-source and available under the MIT License.

---

⭐ If you found this useful, consider giving the repository a star!
