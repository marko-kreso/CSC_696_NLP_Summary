from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from datasets import load_metric
import torch
from torch import *
import pandas as pd
from rank_bm25 import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
from tqdm import tqdm

import numpy as np
nltk.download('stopwords')

# Perhaps put this before you call the function so that its not called everytime
stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
def query_predict(abs_sum, i):
    def normalize_collection(input):
        # Tokenize sentence (split into words)
        sents = sent_tokenize(input)

        # Tokenize sentences into words
        sents = [word_tokenize(sent) for sent in sents]

        # Lowercase words and remove punctuation
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        sents = [[w.lower() for w in sent if w.lower() not in punc] for sent in sents]

        # Take out stopwords
        sents = [[w for w in sent if w not in stopwords] for sent in sents]

        # Stemming on the words
        sents = [[ps.stem(w) for w in sent] for sent in sents]

        return sents

    def normalize_query(input):
        # Tokenize query into words
        sent = word_tokenize(input)

        # Lowercase words and remove punctuation
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        sent = [w.lower() for w in sent if w.lower() not in punc]

        # Take out stopwords
        sent = [w for w in sent if w not in stopwords]

        # Stemming on the words
        sent = [ps.stem(w) for w in sent]

        return sent

    pred_sum = dataset['validation'][i]['article']

    if i == 0:
        print(dataset['validation'][i]['abstract'])

    query = normalize_query(abs_sum)
    doc_col = normalize_collection(pred_sum)

    if len(doc_col) == 0:
        print('zero doc length')
        return ""

    if len(query) == 0:
        print('zero query len')
        return ""
    
    bm25 = BM25Okapi(doc_col)
    scores = bm25.get_scores(query)

    top_10_idx = np.argsort(scores)[-7:]

    top_10_idx = sorted(top_10_idx)

    pred_sum = sent_tokenize(pred_sum)

    final_summary = ""
    for idx in top_10_idx:
        final_summary += pred_sum[idx]

    return final_summary

rouge_scores = [0,0]

dataset = load_dataset("ccdv/pubmed-summarization")

rouge = load_metric('rouge')

model = AutoModelForSeq2SeqLM.from_pretrained("./BART-Pubmed_summarizer/")
tokenizer = AutoTokenizer.from_pretrained("./BART-Pubmed_summarizer/")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

model = model.to(device)

i = 0

for data in tqdm(dataset['validation']):
    summary = data['article']
    abstract = data['abstract']
    inputs = tokenizer("summarize: " + summary,return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs["input_ids"], max_length=150, min_length=40,length_penalty=2.0, num_beams=4, early_stopping=True).to(device)
    pred_sum = tokenizer.decode(outputs[0], skip_special_tokens=True)

    final_summary = query_predict(summary, pred_sum)

    results = rouge.compute(predictions=[final_summary],references=[abstract])
    
    rouge_scores[0] += results['rouge1'][1][2]
    rouge_scores[1] += results['rouge2'][1][2]

    i += 1
    if i >= 1000:
        break

# print(rouge_scores[0] / float(len(dataset['validation'])))
# print(rouge_scores[1] / float(len(dataset['validation'])))
print(rouge_scores[0] / 1000.0)
print(rouge_scores[1] / 1000.0)