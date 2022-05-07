# Preprocessing Data
## Import Dependencies
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer

# Tokenization and Cleaning


def split_data(lines): 
    """splits the summary from the body and names the body lines"""
    flag = False
    for i in range(len(lines)):
        if lines[i][0] == '\n':
            summary = lines[:i]
            line = lines[i:]
            flag = True
    if flag is False:
        summary, line = None, None
    return summary, line
        

def split_string(lines):
    sent_tokens = []
    for line in lines:
        line = re.sub("\n", "", line)
        line = re.sub("\\\\.", "", line)
        line = re.sub("===", "", line)
        for sent in sent_tokenize(line):
            sent_tokens.append(sent)
    return sent_tokens

def split_summary(lines):
    sent_tokens = ["_START_"]
    for line in lines:
        line = re.sub("\n", "", line)
        line = re.sub("\\\\.", "", line)
        line = re.sub("===", "", line)
        line = " ".join(["_START_", line, "_END_"])
        for sent in sent_tokenize(line):
            sent_tokens.append(sent)
    sent_tokens.append("_END_")
    return sent_tokens

def one_hot_enc(ar, dic_size):
    #print(ar)
    n = np.max(ar) +1
    i = np.sum(np.eye(dic_size)[ar], axis= 0)
    bl = np.array(i, dtype=bool)
    oh = bl.astype(int)
    #padded = pad_sequences(oh, maxlen=x_voc_size, padding='post', truncating='post')
    return oh

def one_hot_all(x_train, dic_size):
    #print(len(x_train))
    #print(len(x_train[0]))
    #print(x_train[0])
    trains_oh = []
    for each in x_train:
        trains_oh.append(one_hot_enc(each, dic_size))
    return trains_oh


def data_preprocessing(directory= './data', save=True, limited=False):
    ## We find the most suitable maximal length in this article 
    text_overall, summary_overall, title_overall = [], [], [] 
    text_count, summary_count = [], []
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    #word tokenization + eliminating stopwords + words stemming + eliminating short words
    def clean(string):
        word_tokens = word_tokenize(string)
        filtered_sentence = [ps.stem(w) for w in word_tokens if (not w in stop_words) and (len(w)>3)]
        return filtered_sentence 

    for filename in os.listdir(directory):
        with open(directory+'/'+filename) as f:
            print(filename)
            lines = f.readlines()
            f.close()

            title = filename[:-4]
            summary, lines = split_data(lines)
            if lines is None:
                continue
            summary, lines = split_summary(summary), split_string(lines)

            clean_text = []
            tex_count = 0
            for line in lines:
                temp = clean(line)
                if temp != []:
                    tex_count += (len(temp))
                    for t in temp:
                        clean_text.append(t)  
            text_overall.append(clean_text)
            text_count.append(tex_count)

            clean_summary = []
            sum_count = 0
            for line in summary:
                temp = clean(line)
                if temp != []:
                    sum_count+= (len(temp))
                    for t in temp:
                        clean_summary.append(t)
            summary_overall.append(clean_summary)
            summary_count.append(sum_count)   

            title_overall.append(title)  

            if limited is True:
                if len(title_overall)>= 10:
                    break


    ## Store sentence vectors:
    max_text_len = 20000
    max_summary_len = 200
    x_train, x_test, y_train, y_test = train_test_split(text_overall, summary_overall, test_size=0.1, shuffle=True)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    #x_train = pad_sequences(x_train, maxlen=max_text_len, padding='post', truncating='post')
    #x_test = pad_sequences(x_train, maxlen=max_text_len, padding='post', truncating='post')


    x_voc_size = len(tokenizer.word_index)+1


    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(y_train)

    y_train = y_tokenizer.texts_to_sequences(y_train)
    y_test = y_tokenizer.texts_to_sequences(y_test)

    #y_train = pad_sequences(y_train, maxlen=max_summary_len, padding='post', truncating='post')
    #y_test = pad_sequences(y_train, maxlen=max_summary_len, padding='post', truncating='post')


    y_voc_size = len(y_tokenizer.word_index)+1

    x_train= one_hot_all(x_train, x_voc_size)
    y_train = one_hot_all(y_train, y_voc_size)
    x_test= one_hot_all(x_train, x_voc_size)
    y_test = one_hot_all(y_train, y_voc_size)

    ## Save the preprocessed data to file
    if save is True:
        np.savez('preprocessed', x_train=x_train, x_test=x_test, y_train=y_train,y_test=y_test, labels=title_overall,
                                max_text_len=max_text_len, max_summary_len=max_summary_len, x_voc_size=x_voc_size, y_voc_size=y_voc_size,
                                tokenizer = tokenizer, y_tokenizer=y_tokenizer)
    else: return (x_train, x_test, y_train, y_test, max_text_len, max_summary_len, x_voc_size, y_voc_size)

data_preprocessing()