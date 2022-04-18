---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- ccdv/pubmed-summarization
metrics:
- rouge
model-index:
- name: tst_sum2
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
      value: 43.746
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# tst_sum2

This model is a fine-tuned version of [t5-base](https://huggingface.co/t5-base) on the ccdv/pubmed-summarization dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3008
- Rouge1: 43.746
- Rouge2: 20.5504
- Rougel: 31.1213
- Rougelsum: 38.9709
- Gen Len: 126.8901

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
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2.0

### Training results



### Framework versions

- Transformers 4.19.0.dev0
- Pytorch 1.11.0+cu102
- Datasets 2.0.0
- Tokenizers 0.11.6
