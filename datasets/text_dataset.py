# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset

from datasets.common import poison_single_sentence

"""
define text dataset, only support two parties, used for Yahoo
"""


class LabeledTextDataset(Dataset):
    # Data loader for labeled data
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len, aug=False, backdoor_indices=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len

        self.aug = aug
        self.trans_dist = {}

        if aug:
            print('Aug train data by back translation of German')
            self.en2de = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
            self.de2en = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

        self.backdoor_indices = backdoor_indices

    def __len__(self):
        return len(self.labels)

    def augment(self, text):
        if text not in self.trans_dist:
            self.trans_dist[text] = self.de2en.translate(self.en2de.translate(
                text,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
        return self.trans_dist[text]

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def text2data_length_pair(self, text):
        if self.aug:
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return (torch.tensor(text_result), torch.tensor(text_result2)), (text_length, text_length2)
        else:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return torch.tensor(encode_result), length

    def __getitem__(self, idx):
        text = self.text[idx]
        # split the text into two, each for one party
        text_a = text[:int(len(text)/2)]
        text_b = text[int(len(text)/2):]

        if self.backdoor_indices is not None and idx in self.backdoor_indices:
            text_b = poison_single_sentence(text_b)

        tensor_a, length_a = self.text2data_length_pair(text_a)
        tensor_b, length_b = self.text2data_length_pair(text_b)
        label = self.labels[idx]
        return (tensor_a, tensor_b), (label, length_a, length_b), idx


class UnlabeledTextDataset(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def text_id_2data_length_pair(self, text, idx):
        if self.aug is not None:
            u, v, ori = self.aug(text, self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), (length_u, length_v, length_ori))
        else:
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)

    def __getitem__(self, idx):
        text = self.text[idx]
        # split the text into two, each for one party
        text_a = text[:int(len(text) / 2)]
        text_b = text[int(len(text) / 2):]
        zip_a_3data_3length = self.text_id_2data_length_pair(text_a, idx)
        zip_b_3data_3length = self.text_id_2data_length_pair(text_b, idx)
        return zip_a_3data_3length, zip_b_3data_3length, idx
