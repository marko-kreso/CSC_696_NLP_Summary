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

This model is a fine-tuned version of [./BART-Pubmed-Long](https://huggingface.co/./BART-Pubmed-Long) on the ccdv/pubmed-summarization dataset.
It achieves the following results on the evaluation set:
- eval_loss: 1.8440
- eval_rouge1: 0.4608
- eval_rouge2: 0.1537
- eval_rougeL: 0.239
- eval_rougeLsum: 0.4105
- eval_gen_len: 291.6
- eval_runtime: 17.2134
- eval_samples_per_second: 0.581
- eval_steps_per_second: 0.058
- step: 0

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
- eval_batch_size: 10
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Framework versions

- Transformers 4.22.1
- Pytorch 1.12.1+cu102
- Datasets 2.5.1
- Tokenizers 0.12.1
