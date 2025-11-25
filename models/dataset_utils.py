import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from typing import Optional, Union, List, Dict
from collections import OrderedDict
import random

def insert_entity_markers(tokens,heads_pos,tails_pos) -> List[str]:
    """
    """

    head_starts, head_ends = [min(s) for s in heads_pos], [max(s) for s in heads_pos]
    tail_starts, tail_ends = [min(s) for s in tails_pos], [max(s) for s in tails_pos]

    # in Debug Assert len(heads_starts) = len(heads_ends), also for tails
    spans = {}

    for start,end in zip(head_starts,head_ends):
      spans[(start,end)] = 'HEAD'


    for start,end in zip(tail_starts,tail_ends):
      spans[(start,end)] = 'TAIL'


    # Creating a sorted ordered dict so that we can get each insert position in descending order iwth the correct label
    # This is so we only insert in an earlier index each iteration, so as not to mess up the indexing
    ordered_spans = OrderedDict(sorted(spans.items(), key = lambda x: x[0][0], reverse = True))


    # copy tokens to modify
    copy_tokens = tokens[:]


    for (start,end), label in ordered_spans.items():
      copy_tokens.insert(end + 1, f"[{label}_END]")

      copy_tokens.insert(start,f"[{label}_START]")

    return copy_tokens


def load_fewrel(path,entity_markers = True):
    """
    """

    file = open(path, 'r')
    data_dict = json.load(file)

    if entity_markers:
        new_json = {k: [] for k in data_dict.keys()}
        for i in data_dict.keys():
            for j in data_dict[i]:
                tokens = j['tokens']
                heads_pos = j['h'][2]
                tails_pos = j['h'][2]
                new_json[i].append({'tokens':insert_entity_markers(tokens,heads_pos,tails_pos)})
        return new_json
    return data_dict


def sample_episode(dataset,n=5,k=5,Q=10,seed=42):

  rng = random.Random(seed)
  random_classes = rng.sample(dataset.keys(),n)

  support = {}
  query = {}

  for j, cls in enumerate(random_classes):

    sampled = random.sample(dataset[cls],k+Q)
    support[j] = sampled[:k]
    query[j] = sampled[k:]


  return support, query






class few_shot_text(Dataset):
    """
    Opens the json and selects n shots and k ways at random from the set

    """

    def __init__(
        self, json_path, tokenizer, n_ways,
        k_shot, q_shot, episodes=10000, seed=42,
        entity_markers = True
    ):

        self.length = episodes
        self.text = load_fewrel(json_path,entity_markers = entity_markers)
        self.n_ways = n_ways
        self.k_shot = k_shot
        self.seed = seed
        self.tokenizer = tokenizer
        self.q_shot = q_shot



    def __len__(self):

        return self.length


    def __getItem__(self,idx):
        """
        idx does nothing, each episode is a random sample of classes
        """
        support, query = sample_episode(
                                        self.text,self.n_ways,
                                        self.k_shot,self.seed)

        support_list = []
        query_list = []

        for i in range(self.n_ways):
            support_texts = support[i]
            query_texts = query[i]

            support_list.extend(support_tokens)
            query_list.extend(query_texts)


        support_tokens = self.tokenizer(
                                        support_texts, padding=True,
                                        truncation=True, max_length = 128,
                                        return_tensors = 'pt')
        query_tokens = self.tokenizer(
                                        query_texts, padding=True,
                                        truncation=True, max_length = 128,
                                        return_tensors = 'pt')

        support_dict = {}
        support_dict['tokens'] = support_tokens
        support_dict['labels'] = torch.arange(self.n_ways).repeat_interleave(self.k_shot)

        query_dict = {}
        query_dict['tokens'] = query_tokens
        query_dict['labels'] = torch.arange(self.n_ways).repeat_interleave(self.q_shot)



        return support_dict, query_dict
