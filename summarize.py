from transformers import pipeline
from datasets import load_dataset
from datasets import load_metric

dataset = load_dataset("ccdv/pubmed-summarization")
print(dataset)
#print(dataset['train'][0])

rouge = load_metric('rouge')

classifier = pipeline("summarization")


summary = dataset['train'][0]['article']
abstract = dataset['train'][0]['abstract']
#print('SUM1:',summary[:10])
#print('SUM2:',summary[:512])

pred_sum = classifier(summary[:1030])

results = rouge.compute(predictions=[pred_sum],references=[abstract])

print(results["rouge1"])

