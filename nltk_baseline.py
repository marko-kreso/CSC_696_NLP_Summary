import pandas as pd
from rank_bm25 import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from datasets import load_metric

def spl_chars_removal(words):
    return [re.sub("^A-Za-z0-9]+", " ", element) for element in words]

sentence = "I like to make eggs every morning"

# Tokenize the sentence (split into words)
words = word_tokenize(sentence)

# Make all words lowercase
words = [w.lower() for w in words]

# Remove special characters
words = spl_chars_removal(words)

# Remove stop words
stopwords = set(stopwords.words('english'))
filtered_sent = [w for w in words if not w.lower() in stopwords]

# Stemming on the words
ps = PorterStemmer()
stemmed_filtered = [ps.stem(w) for w in filtered_sent]

# Test out querying on a paragraph
document = "He wondered if he should disclose the truth to his friends. It would be a risky move. Yes, the truth would make things a lot easier if they all stayed on the same page, but the truth might fracture the group leaving everything in even more of a mess than it was not telling the truth. It was time to decide which way to go."

# Tokenize the document into sentences
sents = sent_tokenize(document)
# Tokenize the sentences into words
sents = [word_tokenize(sent) for sent in sents]

# Lowercase words
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
sents = [[w.lower() for w in sent if w.lower() not in punc] for sent in sents]

# Take out stopwords
sents = [[w for w in sent if not w in stopwords] for sent in sents]

# Make BM25 index
bm25 = BM25Okapi(sents)
query = ["risky"]

# Tokenize the query here

# Get scores
doc_scores = bm25.get_top_n(query, sents, n=2)
print(doc_scores)

# Base BART
# 0.35135571633623885
# 0.1268504168284072

# OUR Bart
# top 10
# 0.3692741559320443
# 0.14688929018904584

# top 15
# 0.33911876977727123
# 0.14658512308827262

# top 7


# t5
# 0.349624017971481
# 0.12577508071789925
