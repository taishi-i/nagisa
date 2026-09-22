import io
import os
import tarfile
import urllib.request

import nagisa
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CORPUS_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
CORPUS_DIR = "ldcc"

# Content words used as features
TARGET_POSTAGS = ["名詞", "動詞", "形容詞", "形状詞", "英単語"]

# News site names, which tell the category directly
SOURCE_NAMES = ["独女通信", "ITライフハック", "家電チャンネル", "livedoor HOMME", "MOVIE ENTER",
                "Peachy", "S-MAX", "エスマックス", "smaxjp", "Sports Watch"]


def download_corpus():
    if os.path.isdir(CORPUS_DIR):
        return

    with urllib.request.urlopen(CORPUS_URL, timeout=120) as resp:
        data = resp.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        # The filter argument is not available in older Python versions.
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=".", filter="data")
        else:
            tar.extractall(path=".")
    os.rename("text", CORPUS_DIR)


def load_corpus():
    texts = []
    labels = []
    for category in sorted(os.listdir(CORPUS_DIR)):
        cat_path = os.path.join(CORPUS_DIR, category)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith(".txt") or fname.startswith("LICENSE"):
                continue
            with open(os.path.join(cat_path, fname), encoding="utf-8") as f:
                lines = f.read().splitlines()

            # line 1: URL, line 2: date, line 3: title, line 4-: body
            texts.append("\n".join(lines[2:]))
            labels.append(category)
    return texts, labels


def tokenize(text):
    for name in SOURCE_NAMES:
        text = text.replace(name, " ")

    # nagisa works on a sentence, so tokenize the text line by line.
    words = []
    for line in text.splitlines():
        tokens = nagisa.extract(line, extract_postags=TARGET_POSTAGS)
        words += [word.lower() for word in tokens.words]
    return [word for word in words if word not in nagisa.stopwords]


def identity(tokens):
    # The documents are already tokenized.
    return tokens


def main():
    download_corpus()
    texts, labels = load_corpus()
    docs = [tokenize(text) for text in tqdm(texts)]

    # Split the corpus into the train (80%) and test (20%) sets
    train_docs, test_docs, train_labels, test_labels = train_test_split(
        docs, labels, test_size=0.2, stratify=labels, random_state=1234)

    # Convert the documents into TF-IDF vectors
    vectorizer = TfidfVectorizer(analyzer=identity, min_df=2)
    train_vecs = vectorizer.fit_transform(train_docs)
    test_vecs = vectorizer.transform(test_docs)
    print("train: {}, test: {}, features: {}".format(
        len(train_docs), len(test_docs), len(vectorizer.vocabulary_)))

    # Train a classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(train_vecs, train_labels)

    # Evaluate the classifier on the test set
    pred_labels = model.predict(test_vecs)
    print("accuracy: {:.4f}".format(accuracy_score(test_labels, pred_labels)))
    print(classification_report(test_labels, pred_labels, digits=3))

    # Words with the largest weights for each category
    words = vectorizer.get_feature_names_out()
    for category, coef in zip(model.classes_, model.coef_):
        top_words = [words[i] for i in np.argsort(-coef)[:8]]
        print("{:<15} {}".format(category, ", ".join(top_words)))

    # Classify new texts
    new_texts = [
        "新しいAndroidスマートフォンが発売され、アプリの動作も快適になった。",
        "日本代表は昨日の試合で2対1で勝利した。",
        "監督が新作映画の見どころを語った。",
    ]
    new_vecs = vectorizer.transform([tokenize(text) for text in new_texts])
    for text, probs in zip(new_texts, model.predict_proba(new_vecs)):
        best = np.argmax(probs)
        print("\n{}\n  => {} ({:.2f})".format(text, model.classes_[best], probs[best]))


if __name__ == "__main__":
    main()
