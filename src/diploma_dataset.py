import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import math


class ContentDataset(Dataset):
    def __init__(self, data, labels, tokenizer, max_length=90, batch_size=24):
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
    generator: torch.Generator

    def __init__(self, train_path, test_path, tokenizer):
        self.tokenizer =  tokenizer
        self.data_train = pd.read_csv(train_path)
        self.data_test = pd.read_csv(test_path)

        self.generator = torch.Generator()
        self.generator.manual_seed(22)

    def tokenize_function(self, element):
        return self.tokenizer(element, padding="max_length", truncation=True)

    def prepare_data(self, num_examples_train=3000, num_examples_test=2000, class_count=2, batch_size=24):
        selected_test = self.select_examples(self.data_test, num_all=num_examples_test, class_count=class_count)
        dataset_test = ContentDataset(selected_test['content'], selected_test['label'], self.tokenizer)
        self.test_dataloader = DataLoader(dataset_test, shuffle=True, batch_size=batch_size)
        self.train_dataloader = None

        if num_examples_train is not None and num_examples_train > 0:
            selected_train = self.select_examples(self.data_train, num_all=num_examples_train, class_count=class_count)
            dataset_train = ContentDataset(selected_train['content'], selected_train['label'], self.tokenizer)
            self.train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)

        return self.train_dataloader, self.test_dataloader

    def collate_fn(batch):
        input_ids =       torch.stack([torch.tensor(item['tokenized'][i]['input_ids']) for i,item in enumerate(batch)])
        attention_mask =  torch.stack([torch.tensor(item['tokenized'][i]['attention_mask']) for i,item in enumerate(batch)])
        labels =          torch.tensor([item['label'] for item in batch])
        return input_ids, attention_mask, labels


    def select_examples(self, data, num_all, class_count):
        num_examples = math.ceil(num_all / class_count)
        output_data = pd.DataFrame()
        
        for i in range(class_count):
            selected = data[data['label'] == i][:num_examples]
            output_data = pd.concat([output_data, selected])
        output_data.reset_index(inplace=True)
        return output_data