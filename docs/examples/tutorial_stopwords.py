import nagisa

text = "日本語のストップワードを簡単に利用できます。"
tokens = nagisa.tagging(text)
print(tokens.words)
# => ['日本', '語', 'の', 'ストップ', 'ワード', 'を', '簡単', 'に', '利用', 'でき', 'ます', '。']

# Filter out stopwords from the tokenized result
words = [word for word in tokens.words if word not in nagisa.stopwords]
print(words)
# => ['日本', '語', 'ストップ', 'ワード', '簡単', '利用', '。']
