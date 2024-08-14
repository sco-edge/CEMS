import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
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
    SEED = 42
    def get_data_from_csv(score_type):
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
        model_scores = pd.concat(score_list, axis=1)
        labels = np.argmax(model_scores.values, axis=1)
        data_list = pd.concat([agg_data,model_scores], axis=1).assign(label=labels)

        # splitting train-val-test
        train_set, val_test = train_test_split(data_list, test_size=0.2, shuffle=True, stratify=data_list['dataset_id'], random_state=SEED)
        val_set, test_set = train_test_split(val_test, test_size=0.5, shuffle=True, stratify=val_test['dataset_id'], random_state=SEED)
    
        return [train_set, val_set, test_set]

    if score_type=='mixed':
        dataset_per_score_type = dict()
        scaled_scores = {'nima': [], 'clip': []}
        for score_type in ['nima','clip']:
            dataset_per_score_type[score_type] = get_data_from_csv(score_type)
            fit_data = dataset_per_score_type[score_type][0][model_list]    # use train set only
            scaler = MinMaxScaler()
            scaler = scaler.fit(fit_data.values.reshape(-1,1))
            for target_data in dataset_per_score_type[score_type]:
                model_scores = target_data[model_list]
                scaled_data = scaler.transform(model_scores.values.reshape(-1,1)).reshape(model_scores.shape)
                scaled_scores[score_type].append(scaled_data)
                assert(round(model_scores.values[0,0],4) == round(scaled_data[0,0]*(scaler.data_max_.item()-scaler.data_min_.item())+scaler.data_min_.item(),4))

        mixed_data = dict()
        for idx, dataset_type in enumerate(['train','val','test']):
            mixed_scores = scaled_scores['nima'][idx]+scaled_scores['clip'][idx]
            mixed_scores_df = pd.DataFrame(mixed_scores, columns=model_list)
            labels = np.argmax(mixed_scores, axis=1)
            base_data = dataset_per_score_type['nima'][idx][['dataset_id','prompt']].reset_index(drop=True)
            mixed_data[dataset_type] = pd.concat([base_data,mixed_scores_df],axis=1).assign(label=labels)

        train_set, val_set, test_set = [mixed_data[dataset_type] for dataset_type in ['train','val','test']]
    else:
        train_set, val_set, test_set = get_data_from_csv(score_type)

    return test_set

def get_baseline_results(dataset, mixed_dataset, cost_list, model_list):
    # Oracle
    total_QoG, total_cost = (0,0)
    for idx, best_model in enumerate(mixed_dataset['label']):
        total_QoG += dataset.loc[idx][IMG_MODELS[best_model]]
        total_cost += cost_list[IMG_MODELS[best_model]]
    print(f'[Oracle] Avg. QoG: {total_QoG/len(mixed_dataset):.4f}, Cost: {total_cost:.4f}')

    # One-model Only
    for m in model_list:
        avg_QoG = dataset[m].mean()
        total_cost = cost_list[m] * len(dataset)
        print(f'[{m}] Avg. QoG: {avg_QoG:.4f}, Cost: {total_cost:.4f}')

if __name__ == '__main__':
    SEED = 42
    torch.manual_seed(SEED)
    IMG_MODELS = ['SDXL-Turbo','SD-Turbo','amused','TAESD']
    score_type = 'mixed'
    nima_dataset = get_test_set(IMG_MODELS, score_type='nima').reset_index(drop=True)
    clip_dataset = get_test_set(IMG_MODELS, score_type='clip').reset_index(drop=True)
    mixed_dataset = get_test_set(IMG_MODELS, score_type='mixed').reset_index(drop=True)

    # Per-request cost for each model
    mem_footprint = [9.51,4.64,3.75,3.48] # GB
    exec_time = [0.616,0.176,0.489,1.588]  # s
    unit_cost = 0.0000166667 # unit cost for GB-s
    cost_list = dict()
    for idx, m in enumerate(IMG_MODELS):
        cost_list[m] = exec_time[idx] * math.ceil(mem_footprint[idx]) * unit_cost
    print('[+]Cost of each request: ')
    print('\n'.join([f'{m}: {c}' for m, c in cost_list.items()]))

    # Load pretrained quality prediction model
    model_path = f'./quality_prediction_{score_type}_score'
    model = QoG_CLIP.from_pretrained(model_path, num_output=4, drop_prob=0.1)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    test_input = Dataset.from_dict({'prompt':mixed_dataset['prompt'].values, 'labels':mixed_dataset['label'].values})
    tokenize_function = lambda data : tokenizer(data["prompt"], truncation=True)
    test_input = test_input.map(tokenize_function, batched=True).remove_columns(['prompt'])

    trainer = ClassificationTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    test_predictions = trainer.predict(test_input)
    predictions = np.argmax(test_predictions.predictions, axis=-1)

    for data_type, dataset in [('nima',nima_dataset),('clip',clip_dataset)]:
        print(f'\n[+] Results for score_type: {data_type}')
        # Avg. QoG and cost for baselines (Oracle & Each model)
        get_baseline_results(dataset, mixed_dataset, cost_list, IMG_MODELS)

        # QoG enhancement results based on CEMS's prediction on best model
        total_QoG, total_cost = (0,0)
        for idx, pred in enumerate(predictions):
            total_QoG += dataset.loc[idx][IMG_MODELS[pred]]
            total_cost += cost_list[IMG_MODELS[pred]]
        print(f'[CEMS] Avg. QoG: {total_QoG/len(dataset):.4f}, Cost: {total_cost:.4f}')