import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from transformers import (
    CLIPTextConfig, CLIPTextModel, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding, activations
    )
from datasets import Dataset, DatasetDict

CLIP_MODEL = 'openai/clip-vit-base-patch16'

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels,predictions)
    precision = precision_score(labels,predictions,average='macro', zero_division=0)
    recall = recall_score(labels,predictions,average='macro', zero_division=0)   
    f1 = f1_score(labels,predictions,average='macro', zero_division=0)
    probs = torch.nn.functional.softmax(torch.DoubleTensor(logits),dim=1)
    auc = roc_auc_score(labels,probs,average='macro',multi_class='ovr')

    '''
    print('\n\n[+] Eval results')
    print(f'Pre: {precision_score(labels,predictions,average=None)}')
    print(f'Rec: {recall_score(labels,predictions,average=None)}')
    print(f'F1: {f1_score(labels,predictions,average=None)}')
    print(f'AUC: {roc_auc_score(labels,probs,average=None,multi_class="ovr")}')
    print({
        'Acc': round(accuracy,4),
        'Pre': round(precision,4),
        'Rec': round(recall,4),
        'F1': round(f1,4),
        'AUC': round(auc,4),
    })
    print(confusion_matrix(labels,predictions))
    '''
    return {
        'Acc': round(accuracy,4),
        'Pre': round(precision,4),
        'Rec': round(recall,4),
        'F1': round(f1,4),
        'AUC': round(auc,4),
    }

class ClassificationTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        label_stat = torch.bincount(torch.LongTensor(self.train_dataset['labels'])).to(torch.float)
        weights = 1 - (label_stat/label_stat.sum())
        weights = weights**2
        self.weights = weights.to('cuda')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels, weight=self.weights)

        return (loss, {'outputs': outputs}) if return_outputs else loss

class QoG_CLIP(CLIPTextModel):
    def __init__(self, config, num_output, drop_prob):
        super().__init__(config)
        self.clip = CLIPTextModel.from_pretrained(CLIP_MODEL)
        self.hidden_size = self.clip.text_model.final_layer_norm.normalized_shape[0]
        self.dropout = nn.Dropout(drop_prob)
        self.act = activations.QuickGELUActivation()
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, num_output)

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.clip(input_ids, attention_mask)
        o = self.dropout(outputs.pooler_output)
        o = self.act(self.fc(o))
        o = self.dropout(o)
        return self.output(o)

def get_dataset(model_list, score_type):
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
            labels = np.argmax(mixed_scores, axis=1)
            base_data = dataset_per_score_type['nima'][idx][['dataset_id','prompt']]
            mixed_data[dataset_type] = base_data.assign(label=labels)
        
        train_set, val_set, test_set = [mixed_data[dataset_type] for dataset_type in ['train','val','test']]
    else:
        train_set, val_set, test_set = get_data_from_csv(score_type)

    # create huggingface dataset
    dataset = DatasetDict()
    dataset['train'] = Dataset.from_dict({'prompt':train_set['prompt'].values, 
                                          'labels':train_set['label'].values})
    dataset['val'] = Dataset.from_dict({'prompt':val_set['prompt'].values, 
                                          'labels':val_set['label'].values})
    dataset['test'] = Dataset.from_dict({'prompt':test_set['prompt'].values, 
                                          'labels':test_set['label'].values})
    dataset = dataset.shuffle(seed=SEED)
    return dataset

if __name__ == '__main__':
    if len(sys.argv) != 2:
        score_type = 'nima'
    else:
        score_type = sys.argv[1]
        assert(score_type in ['nima','clip','mixed'])

    SEED = 42
    torch.manual_seed(SEED)
    IMG_MODELS = ['SDXL-Turbo','SD-Turbo','amused','TAESD']
    dataset = get_dataset(IMG_MODELS, score_type=score_type)

    config = CLIPTextConfig.from_pretrained(CLIP_MODEL)
    model = QoG_CLIP(config, num_output=len(IMG_MODELS), drop_prob=0.1)
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenize_function = lambda data : tokenizer(data["prompt"], truncation=True)
    dataset = dataset.map(tokenize_function, batched=True).remove_columns(['prompt'])
    RUN_NAME = f'quality_prediction_{score_type}_score'
    OUTPUT = "."
    BATCH_SIZE = 32
    EPOCH = 10
    WEIGHT_DECAY = 0.1
    LEARNING_RATE = 6.4e-6
    MODEL_NAME = f'{OUTPUT}/{RUN_NAME}'
    training_args = TrainingArguments(
        run_name=RUN_NAME,
        output_dir=OUTPUT,
        seed=SEED,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCH,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=2,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        metric_for_best_model="eval_F1",
        load_best_model_at_end=True,
        # maintain labels to compute loss
        remove_unused_columns=False,
    )

    trainer = ClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f'Training start: {RUN_NAME}')
    trainer.train()
    trainer.save_model(MODEL_NAME)
    print(f'Saved trained model as {MODEL_NAME}')

    test_predictions = trainer.predict(dataset['test'])
    print('[+] test set result\n', getattr(test_predictions, 'metrics'))
