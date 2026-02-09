# ===================== 12GB RAM-SAFE FINAL COLAB CODE =====================
# Fake News Detection on WELFake
# TF-IDF (sparse) + Meta Features + LightGBM
# ========================================================================

!pip install -q lightgbm nltk scikit-learn

# -------- IMPORTS --------
import re, random, warnings, gc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------- PATHS --------
TRAIN_PATH = "/content/WELFake_Dataset.csv"
TEST_PATH  = "/content/test.csv"

# -------- LOAD & FIX DATA --------
train_df = pd.read_csv(TRAIN_PATH, encoding="latin-1", low_memory=False)
test_df  = pd.read_csv(TEST_PATH, encoding="latin-1")

train_df = train_df[["id", "title", "text", "label"]]

# Robust label fix
train_df["label"] = train_df["label"].astype(str).str.strip()
train_df = train_df[train_df["label"].isin(["0", "1"])]
train_df["label"] = train_df["label"].astype(int)

print("Label distribution:")
print(train_df["label"].value_counts())

# -------- TEXT PREPROCESSING --------
import nltk
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english")) - {"no", "not", "never"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOPWORDS and len(w) > 2
    ]
    return " ".join(tokens)

for df in [train_df, test_df]:
    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)

# -------- META FEATURES --------
def add_meta(df):
    df["char_len"] = df["content"].apply(len)
    df["word_len"] = df["content"].apply(lambda x: len(x.split()))
    df["caps_ratio"] = df["text"].apply(lambda x: sum(c.isupper() for c in x) / max(len(x),1))
    df["punct_count"] = df["text"].apply(lambda x: sum(c in "!?." for c in x))
    return df

train_df = add_meta(train_df)
test_df  = add_meta(test_df)

# -------- TF-IDF (REDUCED & SPARSE) --------
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

tfidf = TfidfVectorizer(
    max_features=25000,   # 🔥 reduced
    ngram_range=(1,2),
    min_df=5,
    max_df=0.9,
    stop_words="english",
    sublinear_tf=True
)

X_text = tfidf.fit_transform(train_df["content"])
X_meta = train_df[["char_len","word_len","caps_ratio","punct_count"]].values

X = hstack([X_text, X_meta]).tocsr()
y = train_df["label"].values

X_test_text = tfidf.transform(test_df["content"])
X_test_meta = test_df[["char_len","word_len","caps_ratio","punct_count"]].values
X_test = hstack([X_test_text, X_test_meta]).tocsr()

print("Feature matrix shape:", X.shape)

gc.collect()

# -------- TRAIN / VALIDATION SPLIT (NO CV) --------
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)

# -------- LIGHTGBM (MEMORY-SAFE) --------
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

model = lgb.LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=SEED
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(100)]
)

# -------- EVALUATION --------
val_preds = model.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, val_preds))
print("\nClassification Report:")
print(classification_report(y_val, val_preds))

# -------- TEST PREDICTIONS --------
test_preds = model.predict(X_test)

submission = pd.DataFrame({
    "id": test_df["id"],
    "label": test_preds
})

submission.to_csv("submission_final.csv", index=False)
print("\n✅ submission_final.csv generated successfully")
# ========================================================================
