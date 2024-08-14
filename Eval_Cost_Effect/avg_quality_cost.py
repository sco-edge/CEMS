import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from transformers import (
    AutoTokenizer, Trainer, DataCollatorWithPadding
    )
from datasets import Dataset
from collections import Counter
import sys
sys.path.append('.')
from quality_prediction import QoG_CLIP

class ClassificationTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return (loss, {'outputs': outputs}) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels,predictions)
    precision = precision_score(labels,predictions,average='macro', zero_division=0)
    recall = recall_score(labels,predictions,average='macro', zero_division=0)   
    f1 = f1_score(labels,predictions,average='macro', zero_division=0)
    probs = torch.nn.functional.softmax(torch.DoubleTensor(logits),dim=1)
    auc = roc_auc_score(labels,probs,average='macro',multi_class='ovr')

    return {
        'Acc': round(accuracy,4),
        'Pre': round(precision,4),
        'Rec': round(recall,4),
        'F1': round(f1,4),
        'AUC': round(auc,4),
    }

def get_test_set(model_list, score_type):
    df_list = []
    for m in model_list:
        base_df = pd.read_csv(f'./Model_Perf/{score_type}_score/{score_type}_score_with_time_{m}.csv')
        aug_df = pd.read_csv(f'./Model_Perf/aug_10k/{score_type}_score_with_time_{m}.csv')
        base_df.insert(0,'dataset_id',base_df['prompt_id'].str[:2])
        base_df = base_df[['dataset_id','prompt',f'{score_type}_score']]
        aug_df = aug_df[['dataset_id','prompt',f'{score_type}_score']]
        agg_df = pd.concat([base_df,aug_df],axis=0).reset_index(drop=True)
        df_list.append(agg_df)

    score_list = []
    for m, df in zip(model_list, df_list):
        score_list.append(df[f'{score_type}_score'].rename(m))
    
    agg_data = df_list[0][['dataset_id','prompt']]
    data_list = pd.concat([agg_data]+score_list, axis=1)
    labels = np.argmax(pd.concat(score_list, axis=1).values, axis=1)
    data_list = data_list.assign(label=labels)
    
    # splitting train-val-test
    train_set, val_test = train_test_split(data_list, test_size=0.2, shuffle=True, stratify=data_list['dataset_id'], random_state=42)
    val_set, test_set = train_test_split(val_test, test_size=0.5, shuffle=True, stratify=val_test['dataset_id'], random_state=42)

    return test_set

if __name__ == '__main__':
    if len(sys.argv) != 2:
        score_type = 'nima'
    else:
        score_type = sys.argv[1]
        assert(score_type in ['nima','clip'])

    SEED = 42
    torch.manual_seed(SEED)
    IMG_MODELS = ['SDXL-Turbo','SD-Turbo','amused','TAESD']
    dataset = get_test_set(IMG_MODELS, score_type=score_type).reset_index(drop=True)

    # Per-request cost for each model
    mem_footprint = [9.51,4.64,3.75,3.48] # GB
    exec_time = [0.616,0.176,0.489,1.588]  # s
    unit_cost = 0.0000166667 # unit cost for GB-s
    cost_list = dict()
    for idx, m in enumerate(IMG_MODELS):
        cost_list[m] = exec_time[idx] * math.ceil(mem_footprint[idx]) * unit_cost
    print('[+]Cost of each request: ')
    print('\n'.join([f'{m}: {c}' for m, c in cost_list.items()]))

    # Oracle
    avg_QoG = dataset[IMG_MODELS].max(axis=1).mean()
    total_cost = sum([cost_list[m] for m in dataset[IMG_MODELS].idxmax(axis=1).values])
    print(f'[Oracle] Avg. QoG: {avg_QoG:.4f}, Cost: {total_cost:.4f}')

    # One-model Only
    for m in IMG_MODELS:
        avg_QoG = dataset[m].mean()
        total_cost = cost_list[m] * len(dataset)
        print(f'[{m}] Avg. QoG: {avg_QoG:.4f}, Cost: {total_cost:.4f}')

    # Load pretrained quality prediction model
    model_path = f'./quality_prediction_{score_type}_score'
    model = QoG_CLIP.from_pretrained(model_path, num_output=4, drop_prob=0.1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_input = Dataset.from_dict({'prompt':dataset['prompt'].values, 'labels':dataset['label'].values})
    tokenize_function = lambda data : tokenizer(data["prompt"], truncation=True)
    test_input = test_input.map(tokenize_function, batched=True).remove_columns(['prompt'])

    trainer = ClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    test_predictions = trainer.predict(test_input)

    # CEMS's prediction on best model
    predictions = np.argmax(test_predictions.predictions, axis=-1)
    total_QoG, total_cost = (0,0)
    for idx, pred in enumerate(predictions):
        total_QoG += dataset.loc[idx][IMG_MODELS[pred]]
        total_cost += cost_list[IMG_MODELS[pred]]
    print(f'[CEMS] Avg. QoG: {total_QoG/len(dataset):.4f}, Cost: {total_cost:.4f}')