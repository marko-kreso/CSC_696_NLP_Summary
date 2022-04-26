---
tags:
- generated_from_trainer
datasets:
- ccdv/pubmed-summarization
model-index:
- name: BARTBM25-Pubmed_eval
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# BARTBM25-Pubmed_eval

This model is a fine-tuned version of [./BART-Pubmed_summarizer](https://huggingface.co/./BART-Pubmed_summarizer) on the ccdv/pubmed-summarization dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 15
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.19.0.dev0
- Pytorch 1.11.0+cu102
- Datasets 2.0.0
- Tokenizers 0.11.6
