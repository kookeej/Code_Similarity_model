import argparse
import pickle
from tqdm import tqdm
import gc

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import DefaultConfig

config = DefaultConfig()

# Tokenizer
def tokenizing(dataset, args, mode):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    gc.collect()
    codes1 = dataset['code1'].tolist()
    codes2 = dataset['code2'].tolist()
    length = len(codes1)
    
    input_ids = []
    attention_mask = []
    input_ids2 = []
    attention_mask2 = []
    
    if mode == "train":
        labels = dataset['similar'].tolist()
    else:
        labels= None
    
    gc.collect()
    for i in tqdm(range(0, len(codes1), 256)):
        chunck1 = codes1[i:i+256]
        tokenized = tokenizer(
            chunck1,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_len
        )
        input_ids.extend(tokenized['input_ids'])
        attention_mask.extend(tokenized['attention_mask'])
        gc.collect()
        chunck2 = codes2[i:i+256]
        tokenized2 = tokenizer(
            chunck2,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=args.max_len
        )
        input_ids2.extend(tokenized['input_ids'])
        attention_mask2.extend(tokenized['attention_mask'])
  
#     for key, value in tokenized2.items():
#         tokenized[key+"2"] = value
    tokenized = {'input_ids': input_ids, 'attention_mask': attention_mask,
                 'input_ids2': input_ids2, 'attention_mask2': attention_mask2} 
        
    return tokenized, labels, length


# Dataset 구성.
class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset, labels, length, mode):
        self.tokenized_dataset = tokenized_dataset
        self.length = length
        self.mode = mode
        if self.mode == "train":
            self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        if self.mode == "train":
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.length
    
    
def pro_dataset(dataset, batch_size, args, mode="train"):
    tokenized, labels, length = tokenizing(dataset, args, mode=mode)
    custom_dataset = CustomDataset(tokenized, labels, length, mode=mode)
    if mode == "train":
        OPT = True
    else:
        OPT = False
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size,
        shuffle=OPT,
        drop_last=OPT
    )
    return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../sample_train.csv', help="train dataset path")
    parser.add_argument('--test_path', type=str, default='../test.csv', help="test dataset path")
    parser.add_argument('--test_size', type=float, default=0.1, help="train/test split size")
    parser.add_argument('--max_len', type=int, default=256, help="max token length for tokenizing")

    args = parser.parse_args()
        
    dataset = pd.read_csv(args.train_path)
    train_dataset, valid_dataset = train_test_split(dataset, test_size=args.test_size, random_state=42, stratify=dataset['similar'])
    
    test_dataset = pd.read_csv(args.test_path)
    
    print("train dataset size: {}    |    valid dataset size: {}     |     test dataset size: {}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))
    
    print("Preprocessing dataset...")
    train_dataloader = pro_dataset(train_dataset, config.TRAIN_BATCH, args, mode='train')
    print("complete!")
    valid_dataloader = pro_dataset(valid_dataset, config.VALID_BATCH, args, mode='train')
    print("complete!")
    test_dataloader = pro_dataset(test_dataset, config.TEST_BATCH, args, mode='test')
    print("DataLoaders Complete!")
    
    # Save DataLoader with pickle file.
    print("Save DataLoader...")
    gc.collect()
    pickle.dump(train_dataloader, open('data/train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()
    pickle.dump(valid_dataloader, open('data/valid_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)   
    gc.collect()
    pickle.dump(test_dataloader, open('data/test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print("Data Preprocessing Complete!")     