import torch
import torch.nn as nn
import torch.optim as optim

import transformers

import numpy as np


# seed 고정 
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    

# cost function
def get_criterion(name='cross_entropy_loss'):
    if name == 'cross_entropy_loss':
        return nn.CrossEntropyLoss()

# optimizer 정의
def get_optimizer(model, name='adam'):
    if name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=0.0001,
            weight_decay=0.01,
            eps=1e-8
        )
    return optimizer

# scheduler 정의
def get_scheduler(optimizer, train_dataloader, args, name='linear'):
    if name == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    elif name == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    return scheduler