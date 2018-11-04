path_to_model = '../save/run2.hdf5'
path_to_vord2vec = '../data/data/ubercorpus.lowercased.lemmatized.word2vec.300d'
path_to_lemma_dict = "../data/data/lemma_dict.txt"
path_to_stop_words = "../data/data/stop_words_mini.txt"
path_to_antonyms_dict = "../data/data/antonyms.txt"
path_to_tonal_dict = '../data/data/tonal_dict.tsv'
gpu_usage = 0.2
threshold = 0.2
limit = 70

import gensim
import sys
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import re




def get_lemma_dict(path=path_to_lemma_dict):
    lemma_dict = dict()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split()
            lemma_dict[l[0]] = l[1]
    return lemma_dict


def get_stop_words(path=path_to_stop_words):
    stop_words = set()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0] != '*':
                stop_words.add(line.strip())
    return stop_words


def get_ascii(word):
    l = "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя-\'"
    s = "!?.;\"'/\\,;()"
    new = ""
    for w in word:
        if w in l:
            new += w
        elif w in s and (new and new[-1] != ' '):
            new += " "
    return new


def get_lemma_word(word, use_stop_words=True):
    new_word = get_ascii(word.lower().strip())
    words = [x.strip() for x in new_word.split()]
    if len(words) <= 1:
        if new_word and new_word in lemma_dict:
            if not use_stop_words or new_word not in stop_words:
                return [lemma_dict[new_word]]
    else:
        res = []
        for word in words:
            if word and word in lemma_dict:
                if not use_stop_words or new_word not in stop_words:
                    res.append(lemma_dict[word])
        return res
    return [""]


def get_lemma_par(par):
    new = []
    for sent in par.split('.'):
        sent = get_lemma_sent(sent)
        if sent:
            new.append(sent)
    return new


# Main
def get_lemma_sent(sent, use_stop_words=True):
    new = []

    for word in sent.split():
        word = get_lemma_word(word, use_stop_words=use_stop_words)
        if word and word != [""]:
            for w in word:
                new.append(w)
    return new


def get_antonyms_dict(path=path_to_antonyms_dict):
    out = dict()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = line.split(",")
            key, value = row[0], row[1]
            key, value = key.strip(), value.strip()
            out[key] = value
    return out


def get_tonal_dict(path=path_to_tonal_dict):
    tonal_dict = dict()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split("\t")
            if len(l) < 2: return tonal_dict
            tonal_dict[l[0]] = int(l[1][:-1])
    return tonal_dict


def process_antonyms(sent):
    out = []
    next_skip = False
    for i in range(len(sent) - 1):
        if next_skip:
            next_skip = False
            continue
        if sent[i] in opposite_dict:
            if sent[i + 1] in antonyms_dict:
                sent[i + 1] = antonyms_dict[sent[i + 1]]
            else:
                next_skip = True
        else:
            out.append(sent[i])
    if len(sent) < 1: return None
    if not next_skip and sent[-1] not in opposite_dict: out.append(sent[-1])
    return out


def process_sent(sent, use_antonyms=True, use_stop_words=True):
    sent = get_lemma_sent(sent, use_stop_words=use_stop_words)
    if use_antonyms:
        sent = process_antonyms(sent)
    if sent is None or len(sent) < 1: return None
    return sent


def word_to_vect(word):
    try:
        return vec_model.wv[word]
    except KeyError:
        return None


def string_to_vects(data, max_len=limit, **kwargs):
    data = process_sent(data, **kwargs)
    out = []
    if not data: return
    for word in data:
        vect = word_to_vect(word)
        if vect is not None:
            out.append(vect)
    if out is None: return
    to_add = max_len - len(out)
    out += [[0 for _ in range(300)] for _ in range(to_add)]
    out = np.array(out)
    out = out.reshape(-1)
    return out


def df_to_vects(data, max_len=limit, **kwargs):
    """
    df
    """
    X = []
    y = []
    for idx, row in df.iterrows():
        new_row = string_to_vects(row["text"], **kwargs)
        if new_row is not None:
            X.append(new_row)
            y.append([row["tone"] == -1, row["tone"] == 0, row["tone"] == 1])
    if len(X) < 1: return
    X = np.array(X)
    y = np.array(y)
    return X, y


def sentence_reader(regr, message):
    vect = string_to_vects(message)
    if vect is None: return [np.array([0, 0, 0])]
    vect = vect.reshape(1, -1)
    return regr.predict(vect)


def vis(regr, message):
    out = sentence_reader(regr, message)
    res = list(out)
    print("Negative: ", round(res[0][0] * 100, 3), "%", sep="")
    print("Neutral: ", round(res[0][1] * 100, 3), "%", sep="")
    print("Positive: ", round(res[0][2] * 100, 3), "%", sep="")


opposite_dict = {'ні', 'не', 'без'}
lemma_dict = get_lemma_dict()
antonyms_dict = get_antonyms_dict()
stop_words = get_stop_words()


def analysis_api(classifier, text):
    out = dict()
    tmp = sentence_reader(classifier, text)[0]
    out["overall"] = [tmp[2], tmp[0]]
    out["words"] = []
    sentences = text.split(".")
    sentences = [sentence for sentence in sentences if len(sentence) > 0]
    for sentence in sentences:
        overall_sentence = sentence_reader(classifier, sentence)[0]
        idx = np.argmax(overall_sentence)
        for word in sentence.split():
            overall_word = sentence_reader(classifier, word)[0]
            value = overall_word[idx]
            if idx == 1: value = 0
            if value < threshold: value = 0
            if idx == 0: value *= -1
            out["words"].append(value)
    return out
