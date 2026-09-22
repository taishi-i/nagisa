import io
import math
import os
import tarfile
import urllib.request
from collections import Counter

import nagisa
from tqdm import tqdm

CORPUS_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
CORPUS_DIR = "ldcc"

# Content words used as index terms
TARGET_POSTAGS = ["名詞", "動詞", "形容詞", "形状詞", "英単語"]


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
    titles = []
    texts = []
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
            title = lines[2]
            body = "\n".join(lines[3:])
            titles.append(title)
            texts.append(title+"\n"+body)
    return titles, texts


def split_words(text):
    # nagisa works on a sentence, so tokenize the text line by line.
    words = []
    for line in text.splitlines():
        tokens = nagisa.extract(line, extract_postags=TARGET_POSTAGS)
        words += [word.lower() for word in tokens.words]
    return words


def tokenize(text):
    return [word for word in split_words(text) if word not in nagisa.stopwords]


class BM25:

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, corpus_tokens):
        self.num_docs = len(corpus_tokens)
        self.doc_lens = [len(tokens) for tokens in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / self.num_docs

        # Inverted index: term -> {doc_id: term frequency}
        self.index = {}
        for doc_id, tokens in enumerate(corpus_tokens):
            for term, tf in Counter(tokens).items():
                self.index.setdefault(term, {})[doc_id] = tf

        # Inverse document frequency of each term
        self.idf = {}
        for term, postings in self.index.items():
            df = len(postings)
            self.idf[term] = math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
        return self

    def term_weight(self, term, doc_id):
        tf = self.index.get(term, {}).get(doc_id, 0)
        if tf == 0:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * self.doc_lens[doc_id] / self.avgdl)
        return self.idf[term] * tf / (tf + norm)

    def search(self, query_tokens, k=5):
        # Only documents that contain a query term get a score.
        scores = Counter()
        for term in query_tokens:
            for doc_id in self.index.get(term, {}):
                scores[doc_id] += self.term_weight(term, doc_id)
        return scores.most_common(k)

    def keywords(self, doc_tokens, doc_id, k=10):
        weights = {term: self.term_weight(term, doc_id) for term in dict.fromkeys(doc_tokens)}
        # Sort by weight, and by term for ties to make the order reproducible.
        return sorted(weights.items(), key=lambda x: (-x[1], x[0]))[:k]


def main():
    download_corpus()
    titles, texts = load_corpus()
    corpus_tokens = [tokenize(text) for text in tqdm(texts)]

    bm25 = BM25().fit(corpus_tokens)
    print("documents: {}, vocabulary: {}".format(bm25.num_docs, len(bm25.index)))

    # Keyword extraction
    for doc_id in [1100, 4058, 6233]:
        print("\n[{}] {}".format(doc_id, titles[doc_id]))
        keywords = bm25.keywords(corpus_tokens[doc_id], doc_id, k=10)
        print("  "+", ".join(term for term, weight in keywords))

    # Document search
    for query in ["iPhoneの便利なアプリ", "サッカー日本代表の試合"]:
        print("\nquery: {}".format(query))
        for doc_id, score in bm25.search(tokenize(query), k=5):
            print("  {:.2f}  {}".format(score, titles[doc_id]))


if __name__ == "__main__":
    main()
