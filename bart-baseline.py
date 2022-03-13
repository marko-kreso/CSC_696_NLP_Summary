from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments

dataset = load_dataset("ccdv/pubmed-summarization")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['article'], padding='max_length', max_length=1024, truncation=True)
    target_encodings = tokenizer.batch_encode_plus(example_batch['abstract'], padding='max_length', max_length=1024, truncation=True)
    labels = target_encodings['input_ids']
    
    
    labels = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
    ]
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels,
    }

    return encodings
dataset = dataset.map(convert_to_features,batched=True)

print(dataset)

training_args = TrainingArguments(
    output_dir='./models/bart-summarizer',          
    num_train_epochs=1,           
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,   
    warmup_steps=500,               
    save_steps=10000,
    weight_decay=0.01,              
    logging_dir='./logs',          
)

trainer = Trainer(
    model=model,                       
    args=training_args,                  
    train_dataset=dataset['train'],        
    eval_dataset=dataset['validation']   
)
trainer.train(resume_from_checkpoint=True)
trainer.save_model()
# print(BartForConditionalGeneration)
