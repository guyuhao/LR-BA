# -*- coding: utf-8 -*-
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

path = '../../data/yahoo/'


def train_val_split(labels, n_labeled_per_class, n_labels, seed=0):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class: n_labeled_per_class + 10000])
        val_idxs.extend(idxs[-3000:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)
    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

# back translate using Russian as middle language
def translate_ru(start, end, file_name, train_text):
    trans_result = {}
    for id in tqdm(range(start, end)):
        trans_result[id] = ru2en.translate(en2ru.translate(train_text[id], sampling=True, temperature=0.9), sampling=True, temperature=0.9)
        if id % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)


# back translate using German as middle language
def translate_de(start, end, file_name, train_text):
    trans_result = {}
    for id in tqdm(range(start, end)):
        trans_result[id] = de2en.translate(en2de.translate(train_text[id], sampling=True, temperature=0.9), sampling=True, temperature=0.9)
        if id % 500 == 0:
            with open(file_name, 'wb') as f:
                pickle.dump(trans_result, f)
    with open(file_name, 'wb') as f:
        pickle.dump(trans_result, f)


if __name__ == '__main__':
    # Load translation model
    en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
    ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')

    en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

    en2ru.cuda()
    ru2en.cuda()

    en2de.cuda()
    de2en.cuda()

    file = path + 'train.csv'
    df = pd.read_csv(file, header=None, keep_default_na=False)
    texts = (df[1] + " " + df[2]).values

    labels = df[0].values
    b_texts = []
    for text in texts:
        b_texts.append(text[int(len(text) / 2):])

    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(labels)
    translate_de(0, 100000, path + 'de_1.pkl', texts)
    translate_ru(0, 100000, path + 'ru_1.pkl', texts)

