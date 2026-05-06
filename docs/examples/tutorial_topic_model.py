import io
import os
import tarfile

import nagisa
import requests
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

CORPUS_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
CORPUS_DIR = "ldcc"
N_TOPICS = 9
N_TOP_WORDS = 10
MAX_ITER = 20
TEXT_TYPE = "title"  # or body


def download_corpus():
    if os.path.isdir(CORPUS_DIR):
        return CORPUS_DIR

    resp = requests.get(CORPUS_URL, timeout=120)
    resp.raise_for_status()

    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:*") as tar:
        tar.extractall(path=".", filter="data")

    extracted = "text"
    os.rename(extracted, CORPUS_DIR)
    return CORPUS_DIR


def load_documents(corpus_dir, text_type=TEXT_TYPE):
    documents = []
    for category in sorted(os.listdir(corpus_dir)):
        cat_path = os.path.join(corpus_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.endswith(".txt") or fname.startswith("LICENSE"):
                continue
            fpath = os.path.join(cat_path, fname)
            with open(fpath, encoding="utf-8") as f:
                lines = f.readlines()

            if text_type == "title":
                text = lines[3].strip() if len(lines) > 3 else ""
            else:
                text = "".join(lines[2:]).strip()

            if text:
                documents.append(text)
    return documents


def tokenize_documents(documents):
    stopwords = nagisa.stopwords
    tokenized = []
    for doc in tqdm(documents):
        tokens = nagisa.tagging(doc)
        words = [w for w in tokens.words if len(w) > 1 and w not in stopwords]
        tokenized.append(" ".join(words))
    return tokenized


def run_lda(tokenized_docs):
    vectorizer = CountVectorizer(max_df=0.85, min_df=5)
    dtm = vectorizer.fit_transform(tokenized_docs)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=MAX_ITER,
        learning_method="online",
        random_state=1234,
    )
    lda.fit(dtm)

    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-N_TOP_WORDS:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"\nTopic {topic_idx + 1}:")
        print(f"\t{', '.join(top_words)}")


def main():
    corpus_dir = download_corpus()
    documents = load_documents(corpus_dir)
    tokenized_documents = tokenize_documents(documents)
    run_lda(tokenized_documents)


if __name__ == "__main__":
    main()
