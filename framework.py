import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import os
from torch.utils.data import DataLoader, Dataset
import tqdm
from typing import Dict, List
import numpy as np
import math
from enum import Enum


class Device(Enum):
    CUDA = "cuda",
    CPU =  "cpu"

class ContentDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length=128, batch_size=14):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(self.data[idx], padding="max_length", truncation=True, max_length=self.max_length, return_tensors='pt')

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return input_ids, attention_mask, label

class DiplomaDataset():
    data_train: pd.DataFrame
    data_test: pd.DataFrame
    tokenized_train: Dict[str, List[int]]
    tokenized_test: Dict[str, List[int]]
    train_dataloader: DataLoader
    test_dataloader: DataLoader

    def __init__(self, train_path, test_path, tokenizer):
        self.tokenizer =  tokenizer
        self.data_train = pd.read_csv(train_path)
        self.data_test = pd.read_csv(test_path)

    def tokenize_function(self, element):
        return self.tokenizer(element, padding="max_length", truncation=True)

    def prepare_data(self, num_examples_train=3000, num_examples_test=1000, class_count=2, batch_size=24):
        selected_test = self.select_examples(self.data_test, num_examples=num_examples_test, class_count=class_count)
        dataset_test = ContentDataset(selected_test['content'], selected_test['label'], self.tokenizer)
        self.test_dataloader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)
        self.train_dataloader = None

        if num_examples_train is not None and num_examples_train > 0:
            selected_train = self.select_examples(self.data_train, num_examples=num_examples_train, class_count=class_count)
            dataset_train = ContentDataset(selected_train['content'], selected_train['label'], self.tokenizer)
            self.train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)

        return self.train_dataloader, self.test_dataloader

    def collate_fn(batch):
        input_ids =       torch.stack([torch.tensor(item['tokenized'][i]['input_ids']) for i,item in enumerate(batch)])
        attention_mask =  torch.stack([torch.tensor(item['tokenized'][i]['attention_mask']) for i,item in enumerate(batch)])
        labels =          torch.tensor([item['label'] for item in batch])
        return input_ids, attention_mask, labels


    def select_examples(self, data, num_examples, class_count):
        output_data = pd.DataFrame()
        
        for i in range(class_count):
            selected = data[data['label'] == i][:num_examples]
            output_data = pd.concat([output_data, selected])
        output_data.reset_index(inplace=True)
        return output_data

class DiplomaTrainer():
    
    def __init__(self, model, train_dataloader, test_dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.model_static_q = None
        self.model_dynamic_q = None
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def about(self):
        print(f"Model: {self.model}")

    def train(self,epochs=3):
        validation_results = torch.tensor([0]).to(device=self.device)
        self.model.to(device=self.device)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        for epoch in range(epochs):
            print(f'EPOCH: {epoch}')
            for i, batch in enumerate(tqdm.tqdm(self.train_dataloader)):
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device=self.device)
                attention_mask = attention_mask.to(device=self.device)
                labels = labels.to(device=self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if i % math.ceil(len(self.train_dataloader) / 2) == 0:
                    mid_result = self.evaluate()
                    validation_results = torch.cat((validation_results, mid_result['accuracy'].unsqueeze(0)))
                    print(validation_results)
                    print("\nValidation accuracy "+str(mid_result['accuracy'].item()))

                    val_len = len(validation_results)
                    if val_len > 3\
                        and abs(validation_results[-3:][2] - validation_results[-3:][1]) < 0.05\
                        and abs(validation_results[-3:][0] - validation_results[-3:][1]) < 0.05:
                        print('\nModel is not learning anymore')
                        return self.model
        return self.model
    
    
    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def set_model(self, model):
        self.model = model

    def save_model(self, name):
        os.makedirs('../saved_models', exist_ok=True)
        torch.save(self.model, '../saved_models/' + name)

    def load_model(self, path, bertLike=False):
        # if bertLike: # Se ne dela
        #     self.model = torch.load("../quantized_model_bert10000_full.pth", weights_only=False) # se zamenja
        # else:
        self.model = torch.load(path, weights_only=False)

        return self.model

    def accuracy(self, predictions, labels):
        return torch.sum(predictions == labels) / len(labels)

    def evaluate(self):
        # self.device = self.get_device()
        self.model.eval()
        self.model.to(device=self.device)
        all_predictions = torch.tensor([]).to(device=self.device)
        all_labels = torch.tensor([]).to(device=self.device)
        result = {}
        for batch in tqdm.tqdm(self.test_dataloader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            labels = labels.to(device=self.device)
            outputs = self.model(input_ids, attention_mask)
            prediction = outputs.logits.argmax(dim=-1)

            all_predictions = torch.cat((all_predictions, prediction))
            all_labels = torch.cat((all_labels, labels))
            
        all_predictions = all_predictions
        result['accuracy'] = self.accuracy(all_predictions, all_labels)
        return result
    
    def quantize_dynamic(self, inplace=False):
        self.model.eval()
        self.model_dynamic_q = torch.quantization.quantize_dynamic(
        self.model,
        {torch.nn.Linear},
        inplace=True,
        dtype=torch.qint8)

        return self.model_dynamic_q
    
    def print_size_of_model(self):
        os.makedirs("./temp", exist_ok=True)
        torch.save(self.model.state_dict(), "./temp/temp_delme.p")
        print('Size (MB):', os.path.getsize("./temp/temp_delme.p")/1e6)
        os.remove('./temp/temp_delme.p')
    
    def quantize_static(self, inplace=False):
        self.model.to('cpu')
        self.model.eval()
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
        torch.backends.quantized.engine = 'qnnpack'


        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                module.qconfig = torch.quantization.float_qparams_weight_only_qconfig

        # fused_model = torch.ao.quantization.fuse_modules(self.model, [['relu']])
        model_fp32_prepared = torch.ao.quantization.prepare(self.model, inplace=inplace)
        self.evaluate()
        self.model.to(device='cpu')
        self.model_static_q = torch.ao.quantization.convert(model_fp32_prepared)
        if inplace: self.model = self.model_static_q

        return self.model_static_q




    
        




    










