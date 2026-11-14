# prleim task 1 : HTMl scraping

import pandas as pd
import pyreadr
from lxml import html
from lxml import etree

claims_raw = pyreadr.read_r("data/claims-raw.RData")["claims_raw"]


import re

def parse_html(raw_html, include_headers = False):
    if not isinstance(raw_html, str) or "<!" not in raw_html:
        return ""
    try:
        page = html.fromstring(raw_html.encode("utf-8"))
    except (etree.ParserError, ValueError):
        return ""

    if include_headers:
        tags = "//p | //h1 | //h2 | //h3 | //h4 | //h5 | //h6"
    else:
        tags = "//p"

    text = " ".join(element.text_content() for element in page.xpath(tags))

    # clean text (based on preprocessing.R)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"[\W\d_]|nbsp", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


# HTML scraping to all webpages/ preprocessing.R steps

claims_clean = claims_raw[
    claims_raw["text_tmp"].fillna("").str.contains("<!", regex = False)
].copy()

claims_clean["paragraph_text"] = claims_clean["text_tmp"].apply(
    parse_html
)

claims_clean["header_text"] = claims_clean["text_tmp"].apply(
    lambda x: parse_html(x, include_headers = True)
)

# compare the number of words
claims_clean["paragraph_words"] = claims_clean["paragraph_text"].str.split().str.len()
claims_clean["header_words"] = claims_clean["header_text"].str.split().str.len()

print("\nTotal webpages:", len(claims_clean))
print("Pages with additional header words:",
      (claims_clean["header_words"] > claims_clean["paragraph_words"]).sum())


# Preliminary Task #1 the train/test split partioning

from sklearn.model_selection import train_test_split

claims_clean = claims_clean[
  claims_clean["paragraph_text"].str.len() > 5
].copy()

# binary class: 1(relevant), 0(irrelevant)
claims_clean["binary_label"] = (
  claims_clean["bclass"] == "Relevant claim content"
).astype(int)

train_data, test_data = train_test_split(
  claims_clean,
  test_size = 0.2,
  random_state = 110122,
  stratify = claims_clean["binary_label"]
)

print("\nTraining observations:", len(train_data))
print("Testing observations:", len(test_data))
print("\nBinary class counts:")
print(claims_clean["binary_label"].value_counts())

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF paragraphs
tfidf_p = TfidfVectorizer(
    stop_words = "english",
    token_pattern = r"(?u)\b[a-z]{3,}\b",
    min_df = 2
)

x_train_p = tfidf_p.fit_transform(train_data["paragraph_text"])
x_test_p = tfidf_p.transform(test_data["paragraph_text"])

# TF-IDF paragraphs + headers
tfidf_h = TfidfVectorizer(
    stop_words = "english",
    token_pattern = r"(?u)\b[a-z]{3,}\b",
    min_df = 2
)

x_train_h = tfidf_h.fit_transform(train_data["header_text"])
x_test_h = tfidf_h.transform(test_data["header_text"])

print("\nParagraph TF-IDF dimensions:", x_train_p.shape)
print("Paragraph + header TF-IDF dimensions:", x_train_h.shape)

# Logistic Principal Component Regression

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

y_train = train_data["binary_label"]
y_test = test_data["binary_label"]

# split training data into training and validation
dev_train, dev_val = train_test_split(
    train_data,
    test_size = 0.2,
    random_state = 110122,
    stratify = train_data["binary_label"]
)

pc_options = [20, 40, 80, 120]
pc_results = []

for text_col in ["paragraph_text", "header_text"]:

    # TF-IDF using development training data 
    tfidf = TfidfVectorizer(
        stop_words = "english",
        token_pattern = r"(?u)\b[a-z]{3,}\b",
        min_df = 2
    )

    x_dev = tfidf.fit_transform(dev_train[text_col])
    x_val = tfidf.transform(dev_val[text_col])

    # compute components from training data
    svd = TruncatedSVD(
        n_components = max(pc_options),
        random_state = 110122
    )

    dev_pc = svd.fit_transform(x_dev)
    val_pc = svd.transform(x_val)

    for k in pc_options:
        scaler = StandardScaler()
        x1 = scaler.fit_transform(dev_pc[:, :k])
        x2 = scaler.transform(val_pc[:, :k])

        model = LogisticRegression(max_iter = 1000)
        model.fit(x1, dev_train["binary_label"])

        preds = model.predict(x2)
        acc = accuracy_score(dev_val["binary_label"], preds)

        pc_results.append({
            "text": text_col,
            "n_pc": k,
            "val_accuracy": acc,
            "variance": svd.explained_variance_ratio_[:k].sum()
        })

pc_results = pd.DataFrame(pc_results)

print("\nComponent comparison:")
print(pc_results.round(3).to_string(index = False))

# selecting/finding one component count for the models
avg_accuracy = pc_results.groupby("n_pc")["val_accuracy"].mean()
best_k = int(avg_accuracy.idxmax())

print("\nSelected components:", best_k)

# using selected # of components 
n_pc = best_k

def fit_pcr(x_train, x_test, name):
    svd = TruncatedSVD(n_components = n_pc, random_state = 110122)
    train_pc = svd.fit_transform(x_train)
    test_pc = svd.transform(x_test)

    # standardizing PC scores
    scaler = StandardScaler()
    train_pc = scaler.fit_transform(train_pc)
    test_pc = scaler.transform(test_pc)

    # logistic regression
    model = LogisticRegression(max_iter = 1000)
    model.fit(train_pc, y_train)

    probabilities = model.predict_proba(test_pc)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\n", name)
    print("Variance retained:", round(svd.explained_variance_ratio_.sum(), 3))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 3))

    return model, svd, scaler


# model 1: paragraphs 
model_p, svd_p, scaler_p = fit_pcr(x_train_p, x_test_p, "Paragraphs only")

# model 2: paragraphs + headers
model_h, svd_h, scaler_h = fit_pcr(x_train_h, x_test_h, "Paragraphs + headers")

# Premilinary Task #2 Bigram Tokenization

tfidf_bigram = TfidfVectorizer(
  ngram_range = (2,2),
  token_pattern = r"(?u)\b[a-z]+\b",
  min_df = 2
)

x_train_bigram = tfidf_bigram.fit_transform(
  train_data["paragraph_text"]
)

x_test_bigram = tfidf_bigram.transform(
  test_data["paragraph_text"]
)

print("\nBigram TF-IDF dimensions:", x_train_bigram.shape)

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# word model steps as Task 1
word_pipeline = make_pipeline(
    TfidfVectorizer(
        stop_words = "english",
        token_pattern = r"(?u)\b[a-z]{3,}\b",
        min_df = 2
    ),
    TruncatedSVD(n_components = n_pc, random_state = 110122),
    StandardScaler(),
    LogisticRegression(max_iter = 1000)
)

# out of fold predictions during training
folds = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 110122
)

word_train_logodds = cross_val_predict(
    word_pipeline,
    train_data["paragraph_text"],
    y_train,
    cv = folds,
    method = "decision_function"
)

# test predictions with fitted word model Task 1
word_test_pc = scaler_p.transform(svd_p.transform(x_test_p))
word_test_logodds = model_p.decision_function(word_test_pc)

# validating # of bigram PCs

# original split
y_dev = dev_train["binary_label"]
y_val = dev_val["binary_label"]

# word log odds development training data
dev_word_logodds = cross_val_predict(
    word_pipeline,
    dev_train["paragraph_text"],
    y_dev,
    cv = folds,
    method = "decision_function"
)

# log odds validation data
word_pipeline.fit(dev_train["paragraph_text"], y_dev)
val_word_logodds = word_pipeline.decision_function(
    dev_val["paragraph_text"]
)

# word second stage baseline
baseline = LogisticRegression(max_iter = 1000)
baseline.fit(dev_word_logodds.reshape(-1, 1), y_dev)

baseline_probs = baseline.predict_proba(
    val_word_logodds.reshape(-1, 1)
)[:, 1]

bigram_results = [{
    "n_bigram_pc": 0,
    "val_accuracy": accuracy_score(y_val, baseline_probs >= 0.5),
    "val_auc": roc_auc_score(y_val, baseline_probs)
}]

# bigram TF-IDF training data
bigram_tfidf_val = TfidfVectorizer(
    ngram_range=(2, 2),
    token_pattern=r"(?u)\b[a-z]+\b",
    min_df=2
)

x_big_dev = bigram_tfidf_val.fit_transform(
    dev_train["paragraph_text"]
)
x_big_val = bigram_tfidf_val.transform(
    dev_val["paragraph_text"]
)

# compute bigram components
bigram_options = [10, 20, 40, 80]

svd_val = TruncatedSVD(
    n_components = max(bigram_options),
    random_state = 110122
)

dev_big_pc = svd_val.fit_transform(x_big_dev)
val_big_pc = svd_val.transform(x_big_val)

for k in bigram_options:
    scaler = StandardScaler()

    dev_pc = scaler.fit_transform(dev_big_pc[:, :k])
    val_pc = scaler.transform(val_big_pc[:, :k])

    x1 = np.column_stack((dev_word_logodds, dev_pc))
    x2 = np.column_stack((val_word_logodds, val_pc))

    model = LogisticRegression(max_iter = 1000)
    model.fit(x1, y_dev)

    probs = model.predict_proba(x2)[:, 1]

    bigram_results.append({
        "n_bigram_pc": k,
        "val_accuracy": accuracy_score(y_val, probs >= 0.5),
        "val_auc": roc_auc_score(y_val, probs)
    })

bigram_results = pd.DataFrame(bigram_results)

print("\nBigram component comparison:")
print(bigram_results.round(3).to_string(index=False))

# bigram principal components

# selected on basis of highest validation ROC AUC
n_bigram_pc = 40

bigram_svd = TruncatedSVD(
  n_components = n_bigram_pc,
  random_state = 110122
)

bigram_train_pc = bigram_svd.fit_transform(x_train_bigram)
bigram_test_pc = bigram_svd.transform(x_test_bigram)

bigram_scaler = StandardScaler()
bigram_train_pc = bigram_scaler.fit_transform(bigram_train_pc)
bigram_test_pc = bigram_scaler.transform(bigram_test_pc)

print("Bigram variance retained:",
  round(bigram_svd.explained_variance_ratio_.sum(), 3))

# 2nd logistic regression: word log odds/bigram PCs

x_train_second = np.column_stack(
  (word_train_logodds, bigram_train_pc)
)

x_test_second = np.column_stack(
  (word_test_logodds, bigram_test_pc)
)

# fitting
second_model = LogisticRegression(max_iter = 1000)
second_model.fit(x_train_second, y_train)

second_probs = second_model.predict_proba(x_test_second)[:, 1]
second_preds = (second_probs >= 0.5).astype(int)

print("\nWord + Bigram Model")
print("Accuracy:", round(accuracy_score(y_test, second_preds), 3))
print("ROC AUC:", round(roc_auc_score(y_test, second_probs), 3))

word_only_second = LogisticRegression(max_iter = 1000)

word_only_second.fit(
    word_train_logodds.reshape(-1, 1),
    y_train
)

word_only_probs = word_only_second.predict_proba(
    word_test_logodds.reshape(-1, 1)
)[:, 1]

word_only_preds = (word_only_probs >= 0.5).astype(int)

print("\nWord-only second stage model")
print("Accuracy:", round(accuracy_score(y_test, word_only_preds), 3))
print("ROC AUC:", round(roc_auc_score(y_test, word_only_probs), 3))

# Export data for Preliminary Task 3: Neural Network

# Keep only the columns needed for neural network training
nn_train = train_data[
    [".id", "paragraph_text", "binary_label"]
].copy()

nn_test = test_data[
    [".id", "paragraph_text", "binary_label"]
].copy()

# Save training and testing data
nn_train.to_csv("data/nn-train.csv", index=False)
nn_test.to_csv("data/nn-test.csv", index=False)

print("Neural network data exported.")
print("Training observations:", len(nn_train))
print("Testing observations:", len(nn_test))

# AI Use:
# AI was used to translate and adapt course R code into Python assisting with logistic PCR/ bigram modeling + troubleshooting errors
