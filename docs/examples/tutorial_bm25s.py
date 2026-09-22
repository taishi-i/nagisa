import bm25s
import nagisa

from tutorial_bm25 import download_corpus, load_corpus, split_words

INDEX_DIR = "ldcc_bm25s_index"


def build_tokenizer():
    # split_words() lowercases the words after tokenization, so set lower=False.
    return bm25s.tokenization.Tokenizer(lower=False, splitter=split_words,
                                        stopwords=nagisa.stopwords)


def main():
    download_corpus()
    titles, texts = load_corpus()

    # The default tokenizer of bm25s cannot split Japanese text into words.
    print(bm25s.tokenize(["日本語の文書を検索します。"], return_ids=False, show_progress=False))

    # Create the index with the nagisa tokenizer
    tokenizer = build_tokenizer()
    corpus_tokens = tokenizer.tokenize(texts, return_as="tuple")
    retriever = bm25s.BM25(k1=1.5, b=0.75)
    retriever.index(corpus_tokens)

    # Save the index with the titles, and the vocabulary of the tokenizer
    retriever.save(INDEX_DIR, corpus=titles)
    tokenizer.save_vocab(INDEX_DIR)

    # Load them again and search documents
    retriever = bm25s.BM25.load(INDEX_DIR, load_corpus=True)
    tokenizer = build_tokenizer()
    tokenizer.load_vocab(INDEX_DIR)

    queries = ["iPhoneの便利なアプリ", "サッカー日本代表の試合"]
    query_tokens = tokenizer.tokenize(queries, update_vocab=False)
    results, scores = retriever.retrieve(query_tokens, k=5)

    for query, docs, doc_scores in zip(queries, results, scores):
        print("\nquery: {}".format(query))
        for doc, score in zip(docs, doc_scores):
            print("  {:.2f}  {}".format(score, doc["text"]))


if __name__ == "__main__":
    main()
