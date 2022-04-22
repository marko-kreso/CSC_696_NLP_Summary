---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- ccdv/pubmed-summarization
metrics:
- rouge
model-index:
- name: tst_sum
  results:
  - task:
      name: Summarization
      type: summarization
    dataset:
      name: ccdv/pubmed-summarization
      type: ccdv/pubmed-summarization
      args: document
    metrics:
    - name: Rouge1
      type: rouge
      value: 43.6111
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# tst_sum

This model is a fine-tuned version of [facebook/bart-base](https://huggingface.co/facebook/bart-base) on the ccdv/pubmed-summarization dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6125
- Rouge1: 43.6111
- Rouge2: 19.3419
- Rougel: 29.1009
- Rougelsum: 38.6436
- Gen Len: 127.4776

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
- train_batch_size: 3
- eval_batch_size: 3
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 4.0

### Training results



### Framework versions

- Transformers 4.19.0.dev0
- Pytorch 1.11.0+cu102
- Datasets 2.0.0
- Tokenizers 0.11.6
