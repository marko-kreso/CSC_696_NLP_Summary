from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from datasets import load_metric

dataset = load_dataset("ccdv/pubmed-summarization")
print(dataset)
#print(dataset['train'][0])

rouge = load_metric('rouge')

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

rouge_scores = [0, 0]

print(len(dataset['train']))

i = 0
for data in dataset['train']:
    if i % 10 == 0:
        print(i)
    summary = data['article']
    abstract = data['abstract']
    inputs = tokenizer("summarize: " + summary,return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, min_length=40,length_penalty=2.0, num_beams=4, early_stopping=True)

    pred_sum = tokenizer.decode(outputs[0], skip_special_tokens=True)
    results = rouge.compute(predictions=[pred_sum],references=[abstract])

    rouge_scores[0] += results['rouge1'][1][2]
    rouge_scores[1] += results['rouge2'][1][2]
    i += 1
    if i == 100:
        break

print(rouge_scores[0] / float(100.0))
print(rouge_scores[1] / float(100.0))

# summary = dataset['train'][0]['article']
# abstract = dataset['train'][0]['abstract']
# inputs = tokenizer("summarize: " + summary,return_tensors="pt", max_length=512, truncation=True)
# outputs = model.generate(inputs["input_ids"], max_length=150, min_length=40,length_penalty=2.0, num_beams=4, early_stopping=True)
# #print('SUM1:',summary[:10])
# #print('SUM2:',summary[:512])

# pred_sum = tokenizer.decode(outputs[0], skip_special_tokens=True)
# results = rouge.compute(predictions=[pred_sum],references=[abstract])

# print(results["rouge1"])
# print(results["rouge2"])
