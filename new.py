from lib2to3.pgen2 import token
import torch
from os import truncate
from transformers import BartForConditionalGeneration, BartTokenizerFast, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from text_rankcopy import query_predict
import pickle

max_input_length = 1024

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples['article'],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    labels = tokenizer(text_target=examples['abstract'], truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["labels_mask"] = labels["attention_mask"]
    return model_inputs
    #return tokenizer(example['article'], truncation=True,padding=True)
device = torch.device("cuda")
model = BartForConditionalGeneration.from_pretrained('./BART-Pubmed-Long/',)
model.to(device=device)
tokenizer = BartTokenizerFast.from_pretrained('./BART-Pubmed-Long/')

exclude_idx = [2320, 4923, 5210]
raw_dataset = load_dataset("ccdv/pubmed-summarization").filter(lambda example, i: i not in exclude_idx, with_indices=True)
label_pad_token_id = tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 
    )
training_args = TrainingArguments('./BART-test')
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, batch_size=10)
print(type(torch.Tensor(tokenized_dataset['validation']['input_ids']).to('cuda')))
tokenized_dataset = tokenized_dataset.remove_columns(
    raw_dataset['train'].column_names
)
#tokenized_dataset.save_to_disk('./')
summaries = list()
for i in range(0,6630,10):
    t = torch.LongTensor(tokenized_dataset['validation']['input_ids'][i:i+10]).to('cuda')
    summary_ids = model.generate(t, max_length=550, min_length=500, num_beams=4)
    summaries.append(summary_ids)
    print(i)

summaries = [tokenizer.batch_decode(summary,skip_special_tokens=True) for summary in summaries]

with open('sums500.pickle', 'wb') as f:
    pickle.dump(summaries, f)

    
    



print(tokenized_dataset)
# for i in range(6630):
#     summary = dataset['validation'][i]['article']
