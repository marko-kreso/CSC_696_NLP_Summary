# CSC_696_NLP_Summary
#Requirements
    Install latest version of huggingface transformers from **source code.**
    Note that in order for the scripts to work huggingface transformers must be installed from source and not from any other method. Instructions on installing from source code: https://huggingface.co/docs/transformers/installation#installing-from-source

#Getting model fine-tuned pubmed used
    Github does not allow for large files to be uploaded to get the actual pytorch models that stores the weights. Use this link to download the project, in BART-Pubmed_summarizer and T5-Pubmed_Summarizer the models will be stored as torch.bin files. These files correspond to each model according to directory name


#Running training/evaluation
 To train and evaulate Bart and T5 run the scripts train\_and\_eval\_bart and train\_and\_eval\_t5 respectively. If you only want to do evaulation modify the files by removing the --do\_train commandline argument. This will ensure the model will skip training and only do evaluation.

Running eval\_bm25Bart will automatically run evaluation on bm25Bart. It will load the Bart model in BART-Pubmed\_summarizer
