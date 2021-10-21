# -*- coding: utf-8 -*-
# Transformer based Mutation Recognition of SETH Corpus (NLP-NER)
# Training / Tuning Script
# inspired by: https://github.com/huggingface/notebooks/blob/master/transformers_doc/custom_datasets.ipynb

# conda install -c conda-forge seqeval
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report # conda install -c conda-forge scikit-learn 
from sklearn.model_selection import train_test_split, KFold
import torch # conda install pytorch=1.5
from datasets import load_metric # conda install -c huggingface -c conda-forge datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import wandb #conda install -c conda-forge wandb

wandb.login()
wandb.init()

MODEL_NAME = wandb.config.model_name #"bert-base-cased" #"microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
EPOCHS = wandb.config.epochs #4
BATCH_SIZE = wandb.config.batch_size #16
LEARNING_RATE = wandb.config.learning_rate #5e-5 #2e-5
SEED = wandb.config.seed # 42
K = 5 #wandb.config.K #5

UNIQUE_TAGS = ['O', 'B-Gene', 'I-Gene', 'B-SNP', 'I-SNP', 'B-RS']

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n Device: {DEVICE} \n")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def dataloader(df):
    data_noIDs = df[~df["Tag"].str.contains('#')] # remove ID rows
    data_noN = data_noIDs[~data_noIDs["Word"].str.contains('\n', na=False)] # remove '\n' rows after each abstract
    data = data_noN.dropna() # drop NAs

    word_list = data['Word'].tolist()
    tag_list = data['Tag'].tolist()

    # split list intow sublists (sentences) by ' ' indicating sepreate sentences
    # word vector
    word_size = len(word_list)
    word_idx_list = [idx + 1 for idx, val in enumerate(word_list) if val == ' ']
    word_res = [word_list[i: j] for i, j in zip([0] + word_idx_list, word_idx_list + ([word_size] if word_idx_list[-1] != word_size else []))]
    for i in word_res: i.remove(' ')
    token_docs = [x for x in word_res if not len(x)==0]
    # tag vector
    tag_size = len(tag_list)
    tag_idx_list = [idx + 1 for idx, val in enumerate(tag_list) if val == ' ']
    tag_res = [tag_list[i: j] for i, j in zip([0] + tag_idx_list, tag_idx_list + ([tag_size] if tag_idx_list[-1] != tag_size else []))]
    for i in tag_res: i.remove(' ')
    tag_docs = [x for x in tag_res if not len(x)==0]

    return token_docs, tag_docs
    
def encode_tags(tag2id, tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0 (subtokens case)
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

def create_dataset(texts, tags):
    # create encodings for  tokens and tags (for the tags just create a simple mapping)
    tag2id = {tag: id for id, tag in enumerate(UNIQUE_TAGS)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # encode Tokens 
    encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True) # padding="max_length"
    labels = encode_tags(tag2id, tags, encodings)
    encodings.pop("offset_mapping") # we don't want to pass this to the model
  
    dataset = Dataset(encodings, labels)

    return dataset

def compute_metrics(pred):
    metric = load_metric("seqeval")
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Remove ignored index (special tokens)
    true_predictions = [[UNIQUE_TAGS[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(preds, labels)]
    true_labels = [[UNIQUE_TAGS[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(preds, labels)]

    report = classification_report(
            y_true=[val for sublist in true_labels for val in sublist],
            y_pred=[val for sublist in true_predictions for val in sublist],
            labels=UNIQUE_TAGS)#, output_dict=True)
    print(f"\n Classification report from sklearn (calculated results per unique label): \n\n {report} \n\n")

    results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB1", suffix=False,)
    print(f"\n seqeval results (combines B/I prefixes): \n\n {results} \n\n")

    # Log metrics over time to visualize performance in wandb
    wandb.log({"eval_overall_f1": results['overall_f1']})
 
    return results

def training(model, train_dataset, test_dataset):
    training_args = TrainingArguments(
        output_dir = './results',          # output directory
        overwrite_output_dir = True, 
        num_train_epochs = EPOCHS,              # total number of training epochs
        per_device_train_batch_size = BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size = BATCH_SIZE,   # batch size for evaluation
        learning_rate = LEARNING_RATE,
        warmup_steps = 500,                # number of warmup steps for learning rate scheduler
        weight_decay = 0.01,               # strength of weight decay
        #logging_dir = './logs',            # directory for storing logs
        #logging_steps = 10,
        evaluation_strategy = "epoch", # // "steps"
        save_strategy = "epoch",
        #eval_steps = 10000,                 # Evaluation and Save happens every 10 steps
        save_total_limit = 1,            # Only last (n) models are saved. Older ones are deleted.
        load_best_model_at_end = True,   # load the best model when finished training (default metric is loss) # the model loaded at the end of training is the one that had the best performance on validation set. So when you save that model, you have the best model on this validation set.
        metric_for_best_model = "eval_overall_f1", # name of a metric returned by the evaluation 
        greater_is_better = True,        # higher eval_overall_f1 score is better
        seed = SEED,
        report_to = "wandb",             # enable logging to W&B
        run_name="run-1",  # name of the W&B run (optional)
    )

    trainer = Trainer(
        model=model,                         # the instantiated Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,           # evaluation dataset
        compute_metrics=compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    # fine tune model
    trainer.train()
    evaluation = trainer.evaluate() # allows to evaluate again on the evaluation dataset or on another dataset
    return trainer, evaluation

def final_evaluation(best_model_path, final_test_dataset):
    # load best saved model
    best_model = AutoModelForTokenClassification.from_pretrained(best_model_path, num_labels=len(UNIQUE_TAGS)).to(DEVICE)
    # Define test trainer
    final_test_trainer = Trainer(model = best_model, compute_metrics = compute_metrics)
    final_test_evaluation = final_test_trainer.evaluate(eval_dataset = final_test_dataset)
    return final_test_evaluation


def main():
    print("\n -- Load & Preprocess Data -- \n")
    # load data 
    data_path = "corpus_IOB.csv"
    data_raw = pd.read_csv(data_path, encoding="latin1" )
    data_raw = data_raw.dropna() # drop NAs

    # split raw dataframe into multiple dataframes each representing an abstract
    id_idx_list = [idx for idx, val in enumerate(data_raw['Word'].tolist()) if '#' in val] # indices where to split dataframe (based on #ID rows)
    idx_mod = id_idx_list + [len(data_raw)] #[max(id_idx_list)+1]
    list_of_dfs = [data_raw.iloc[idx_mod[n]:idx_mod[n+1]] for n in range(len(idx_mod)-1)]

    # extract final test set
    list_dfs_train , list_dfs_test = train_test_split(list_of_dfs, test_size=0.1, random_state=42)
    final_test = pd.concat(list_dfs_test)
    final_test_texts, final_test_tags = dataloader(final_test)
    # tokenization & create dataset
    final_test_dataset = create_dataset(final_test_texts, final_test_tags)

    # k-fold Cross Validation Training Loop
    best_f1_score = 0
    average_overall_f1_score = 0
    average_SNP_f1_score = 0
    # prepare cross validation
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)
    # enumerate splits
    for i, (train, test) in enumerate(kfold.split(list_dfs_train)):
        print(f"Fold {i}:")
        #print(f"train indices: {train}, test indices: {test}")
        
        train_dfs = []
        for f in train: train_dfs.append(list_dfs_train[f])
        train_fold = pd.concat(train_dfs)

        test_dfs = []
        for f in test: test_dfs.append(list_dfs_train[f])
        test_fold = pd.concat(test_dfs)
    
        train_texts, train_tags = dataloader(train_fold)    
        test_texts, test_tags = dataloader(test_fold)

        # global UNIQUE_TAGS # prevents creation of a local variable called myglobal
        # UNIQUE_TAGS = np.unique(np.array([tag for doc in train_tags for tag in doc])) # look only at training tags because cant predict (on test set) what was seen
        # print(UNIQUE_TAGS, '\n')

        # tokenization & create datasets
        train_dataset = create_dataset(train_texts, train_tags)
        test_dataset = create_dataset(test_texts, test_tags)

        #  load token classification model and specify the number of labels
        print("\n -- Load Model -- \n")
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(UNIQUE_TAGS)).to(DEVICE)

        # training & evaluation
        print("\n -- Start Training -- \n")
        trainer, evaluation = training(model, train_dataset, test_dataset)
        #training_native(model, train_dataset, test_dataset)

        print("\n -- Evaluation Results -- \n")
        print("current overall f1 score: ", evaluation['eval_overall_f1'], '\n')
        print("current SNP f1 score: ", evaluation['eval_SNP']['f1'], '\n')
        
        average_overall_f1_score += evaluation['eval_overall_f1']
        average_SNP_f1_score += evaluation['eval_SNP']['f1']

        if evaluation['eval_overall_f1'] > best_f1_score:
            print(f"-> best SNP f1 score so far (previous best f1 score: {best_f1_score} ) -> save model \n")
            best_f1_score = evaluation['eval_overall_f1']
            model.save_pretrained("saved_model")  
        else:
            print(f"SNP f1 score is worse, best SNP f1 score so far is: {best_f1_score}  \n")

        print(f"\n - Fold {i} done - \n")

    print("\n -- Complete training done -- \n")

    average_overall_f1_score = average_overall_f1_score/K
    average_SNP_f1_score = average_SNP_f1_score/K
    print(f"\n Average overall f1 score (k = {K}): {average_overall_f1_score}")
    print(f"\n Average SNP f1 score (k = {K}): {average_SNP_f1_score}")

    print("\n -- Evaluate best model on final test set -- \n")

    final_test_evaluation = final_evaluation("saved_model", final_test_dataset)

    print("\n -- Done Tuning --")

    return 0

if __name__ == "__main__":
    main()