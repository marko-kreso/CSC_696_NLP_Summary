# CSC_696_NLP_Summary
## Requirements

- Install latest version of huggingface transformers from **source code.**

**Note** that in order for the scripts to work huggingface transformers must be 
installed from source and not from any other method. Instructions on installing 
from [source code](https://huggingface.co/docs/transformers/installation#installing-from-source). 

- Include dependencies for huggingface transformers

- Install nltk
 

## Getting fine-tuned pubmed parameters/weights

Github does not allow for large files to be uploaded for free thus the weights used are not directly added to the project. To get the weights for the models used in the report download the [project](https://drive.google.com/file/d/1MKY6KJ6WE2mxsrd24Cr1fUk6CAYtZqbG/view?usp=sharing) from google drive, in BART-Pubmed\_summarizer and T5-Pubmed\_Summarizer the models will be stored as torch.bin files. These two files should be copied to the latest version of this project..


## Running training/evaluation

To train and evaulate Bart and T5 run the scripts train\_and\_eval\_bart and train\_and\_eval\_t5 respectively. If you only want to do evaulation modify the files by removing the --do\_train commandline argument. This will ensure the model will skip training and only do evaluation.

Running eval\_bm25Bart will automatically run evaluation on bm25Bart. It will load the Bart model in BART-Pubmed\_summarizer
