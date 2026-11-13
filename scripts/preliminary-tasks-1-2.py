# prleim task 1 : HTMl scraping

import pandas as pd
import pyreadr
from lxml import html
from lxml import etree

claims_raw = pyreadr.read_r("data/claims-raw.RData")["claims_raw"]

print(claims_raw.shape)
print(claims_raw[[".id", "bclass"]].head())

import re

def parse_html(raw_html, include_headers=False):
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


first_page = claims_raw["text_tmp"].iloc[0]

paragraphs = parse_html(first_page)
paragraphs_headers = parse_html(first_page, include_headers=True)

print("Paragraphs only:")
print(paragraphs[:200])

print("\nParagraphs and headers:")
print(paragraphs_headers[:200])

print("\nWord counts:")
print("Paragraphs:", len(paragraphs.split()))
print("Paragraphs + headers:", len(paragraphs_headers.split()))

# HTML scraping to all webpages/ preprocessing.R steps

claims_clean = claims_raw[
    claims_raw["text_tmp"].fillna("").str.contains("<!", regex=False)
].copy()

claims_clean["paragraph_text"] = claims_clean["text_tmp"].apply(
    parse_html
)

claims_clean["header_text"] = claims_clean["text_tmp"].apply(
    lambda x: parse_html(x, include_headers=True)
)

# compare the number of words
claims_clean["paragraph_words"] = claims_clean["paragraph_text"].str.split().str.len()
claims_clean["header_words"] = claims_clean["header_text"].str.split().str.len()

print("\nTotal webpages:", len(claims_clean))
print("Pages with additional header words:",
      (claims_clean["header_words"] > claims_clean["paragraph_words"]).sum())

print(claims_clean[[".id", "paragraph_words", "header_words"]].head())

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
    stop_words="english",
    token_pattern=r"(?u)\b[a-z]{3,}\b",
    min_df=2
)

x_train_p = tfidf_p.fit_transform(train_data["paragraph_text"])
x_test_p = tfidf_p.transform(test_data["paragraph_text"])

# TF-IDF paragraphs + headers
tfidf_h = TfidfVectorizer(
    stop_words="english",
    token_pattern=r"(?u)\b[a-z]{3,}\b",
    min_df=2
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

# 40 principal components 
n_pc = 40

def fit_pcr(x_train, x_test, name):
    svd = TruncatedSVD(n_components=n_pc, random_state=110122)
    train_pc = svd.fit_transform(x_train)
    test_pc = svd.transform(x_test)

    # standardizing PC scores
    scaler = StandardScaler()
    train_pc = scaler.fit_transform(train_pc)
    test_pc = scaler.transform(test_pc)

    # logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(train_pc, y_train)

    probabilities = model.predict_proba(test_pc)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("\n", name)
    print("Variance retained:", round(svd.explained_variance_ratio_.sum(), 3))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 3))

    return model


# model 1: paragraphs 
model_p = fit_pcr(x_train_p, x_test_p, "Paragraphs only")

# model 2: paragraphs + headers
model_h = fit_pcr(x_train_h, x_test_h, "Paragraphs + headers")



# AI Use:
# AI was used to translate and adapt R code preprocessing.R into Python.
