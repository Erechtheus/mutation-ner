## A natural language processing project based on HuggingFace Transformers for named entity recognition, specifically mutation recognition in pubMed abstracts of the SETH corpus

## Technologies / Requirements
* anaconda 4.10.1
* python 3.7.11 (64-bit)
* pytorch 1.5
* transformers 4.9.2 
* datasets 1.11.0
* seqeval 1.2.2
* pandas 1.3.2
* numpy 1.19.2
* scikit-learn 0.24.2
* wandb 0.12.1

## Files
### toIOB.py
Script to transform the SETH corpus from .ann to .iob format.
* SETH corpus is accessed from: https://raw.githubusercontent.com/Erechtheus/mutationCorpora/master/corpora/original/SETH/corpus.txt
* Output file: corpus_IOB.csv
    *  2 coloumns: Word | Tag
    * each row contains a token (tokenized with spacy) with the corresponding IOB tag
    * rows starting with '#' contain the pubMed ID of the following tokenized abstract
    * after each sentence there is an empty row
#### Libraries
* pandas 1.3.2
* urllib3 1.26.6
* spacy 3.1.2

### data-analysis.py
Data exploration script for plotting the distribution of the labels over the training and test set.
#### Libraries
* pandas 1.3.2
* scikit-learn 0.24.2
* matplotlib 3.2.2
* seaborn 0.11.2

### NER.py
Main named entity recognition script including the following steps:
* loading and preprocessing the corpus in IOB format in such a way that a transformer model can be trained on it
* splitting into train and final test set
* splitting the train set into k subsets using k-fold cross validation
* for each fold: tokenization and dataset creation in such a way that the token classification model can be trained on it
    * training and evaluation on evaluation set using the Trainer API of HuggingFace 
    * saving the model with the highest overall f1 score on the evaluation set
* load model that achieved best overall f1 score and evaluate on final test set 
* return metrics and average f1 score of the k-fold cross validation
* wandb wrapup for model monotoring and hyperparameter tuning
* pipeline for making NER predictions using the best model
#### Libraries 
* pytorch 1.5
* transformers 4.9.2 
* datasets 1.11.0
* seqeval 1.2.2
* pandas 1.3.2
* numpy 1.19.2
* scikit-learn 0.24.2
* wandb 0.12.1

## Hyperparameter tuning
Following hyperparameters were tuned with bayes search using wandb.to:
* batch size
* epochs
* learning rate
* bert model name
* seed

### Run training on gpu server using wandb 
* create a new sweep on wandb and set parameters
* open terminal inside directory that contains NER_tuning.py file and run:
    * general: `CUDA_VISIBLE_DEVICES=GPU_NUM wandb agent wandb_user_name/wandb_project_name/wandab_sweep_id`
    * example: `CUDA_VISIBLE_DEVICES=0 wandb agent seyamy/NER/5x9xgg4p`

## Make predictions using pipeline
* open the file NER_prediction.py and set the text variable to the text to be predicted. Make sure that the name of the model to be used is correct.
* the name of the output file containing the results is set to "output.txt"
* open terminal inside directory that contains NER_prediction.py file and run: `python NER_prediction.py`
* output.txt file:
    * first line: text to be predicted
    * other lines: predictions/labeling per token 

## Sources
* https://huggingface.co/transformers/training.html
* https://github.com/huggingface/notebooks/blob/master/transformers_doc/custom_datasets.ipynb
* https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb
* https://huggingface.co/dslim/bert-base-NER
