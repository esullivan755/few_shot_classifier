
from sklearn.model_selection import train_test_split

import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def inner_loop(metalearner,support_batch,query_batch,inner_steps,device='cuda'):

    params = metalearner.get_all_params_dict()
    inputs = {}
    query_inputs = {}

    #Flattening batches from B, Support_Size, Sequence_Length -> B*Support_Size, Sequence_Length
    inputs['input_ids'] = support_batch['input_ids'].view(-1, support_batch['input_ids'].shape[-1]).to(device)
    inputs['attention_mask'] = support_batch['attention_mask'].view(-1, support_batch['attention_mask'].shape[-1]).to(device)
    inputs['labels'] = support_batch['labels'].view(-1).to(device)

    query_inputs['input_ids'] = query_batch['input_ids'].view(-1, support_batch['input_ids'].shape[-1]).to(device)
    query_inputs['attention_mask'] = query_batch['attention_mask'].view(-1, support_batch['attention_mask'].shape[-1]).to(device)
    query_inputs['labels'] = query_batch['labels'].view(-1).to(device)

    if any('token_type_ids' in key for key in support_batch.keys()):
        inputs['token_type_ids'] = support_batch['token_type_ids'].view(-1, support_batch['token_type_ids'].shape[-1]).to(device)
        query_inputs['token_type_ids'] = query_batch['token_type_ids'].view(-1,query_batch['token_type_ids'].shape[-1]).to(device)


    #creating empty token type ids on GPU because RoBERTa will autocreate them on cpu down the line if not already defined, causing problems
    else:
        inputs['token_type_ids'] = torch.zeros_like(inputs['input_ids']).to(device)
        query_inputs['token_type_ids'] = torch.zeros_like(query_inputs['input_ids']).to(device)



    for i in range(inner_steps):
        params = metalearner(params, inner=True, **inputs)


    #Trying this out

    query_loss, accuracy = metalearner(params, inner=False, **query_inputs)

    return query_loss, accuracy



def meta_loop(
            train_dataset, val_dataset, tokenizer, metalearner, meta_optimizer,
            inner_steps=1, test_size=.3, epochs=5, device='cuda', max_length=128
        ):
    """
    Expects _dataset as a few_shot_text() object
    """


    train = DataLoader(train_dataset,batch_size = 1)
    val = DataLoader(val_dataset)


    metalearner = metalearner.to(device)





    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0

        metalearner.train()
        for support_batch, query_batch in tqdm(train):
            support_batch = {k: v.to(device) for k, v in support_batch.items()}
            query_batch = {k: v.to(device) for k, v in query_batch.items()}





            loss, accuracy = inner_loop(metalearner, support_batch, query_batch, inner_steps, device=device)

            loss.backward()

            meta_optimizer.step()
            meta_optimizer.zero_grad()
            total_loss += loss.item()
            total_accuracy += accuracy


            
        # Validating
        val_loss = 0
        val_accuracy = 0
        metalearner.eval()
        for support_batch, query_batch in tqdm(val):

            support_batch = {k: v.to(device) for k, v in support_batch.items()}
            query_batch = {k: v.to(device) for k, v in query_batch.items()}
            meta_optimizer.zero_grad()

            loss, accuracy = inner_loop(metalearner, support_batch, query_batch,inner_steps)




            val_loss += loss.item()
            val_accuracy += accuracy

        print(f"Epoch {epoch+1}: train loss = {total_loss / len(train)}")
        print(f"Epoch {epoch+1}: train accuracy = {total_accuracy / len(train)}")
        print(f"Epoch {epoch+1}: validation loss = {val_loss / len(val)}")
        print(f"Epoch {epoch+1}: validation accuracy = {val_accuracy / len(val)}")
