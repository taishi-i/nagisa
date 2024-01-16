from datasets import load_dataset

dataset = load_dataset("taishi-i/nagisa_stopwords")

# the top 100 most commonly used words
words = dataset["nagisa_stopwords"]["words"]

# the part-of-speech list for the top 100 most commonly used words
postags = dataset["nagisa_stopwords"]["postags"]
