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
    #n = np.max(ar) +1
    count = np.sum(np.eye(dic_size)[ar], axis= 0)
    bl = np.array(count, dtype=bool)
    ex = bl.astype(int)
    #padded = pad_sequences(oh, maxlen=x_voc_size, padding='post', truncating='post')
    return count, ex

def one_hot_all(input, dic_size):
    #print(len(x_train))
    #print(len(x_train[0]))
    #print(x_train[0])
    existence = []
    occurrence = []
    
    for each in input:
        count, ex = one_hot_enc(each, dic_size)
        existence.append(ex)
        occurrence.append(count)
    return existence, occurrence


def data_preprocessing(directory= './data', one_hot= False, limited=False):
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
    #max_text_len = 20000
    #max_summary_len = 200
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_overall)

    text_vec = tokenizer.texts_to_sequences(text_overall)

    #x_train = pad_sequences(x_train, maxlen=max_text_len, padding='post', truncating='post')
    #x_test = pad_sequences(x_train, maxlen=max_text_len, padding='post', truncating='post')


    text_voc_size = len(tokenizer.word_index)+1


    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(summary_overall)

    sum_vec = y_tokenizer.texts_to_sequences(summary_overall)


    #y_train = pad_sequences(y_train, maxlen=max_summary_len, padding='post', truncating='post')
    #y_test = pad_sequences(y_train, maxlen=max_summary_len, padding='post', truncating='post')


    sum_voc_size = len(y_tokenizer.word_index)+1


    

    ## Save the preprocessed data to file

    np.savez('preprocessed', text_word2vec = text_vec, summary_word2vec = sum_vec, labels=title_overall,
                                text_voc_size=text_voc_size, summary_voc_size=sum_voc_size,
                            )
    if one_hot is True:
        text_ex, text_count= one_hot_all(text_vec, text_voc_size)
        sum_ex, sum_count = one_hot_all(sum_vec, sum_voc_size)
        
        np.savez('preprocessed_oh', text_word2vec = text_vec, summary_word2vec = sum_vec, text_existence=text_ex,text_count=text_count, 
                                    summary_existence=sum_ex,summary_count=sum_count, labels=title_overall, text_voc_size=text_voc_size, summary_voc_size=sum_voc_size,
                            )



    
def vec2oh(filename):
    data = np.load(filename, allow_pickle=True)
    text_vec = data['text_word2vec']
    sum_vec = data['summary_word2vec']
    labels = data["labels"]
    text_voc_size = data['text_voc_size']
    sum_voc_size = data['summary_voc_size']
    text_ex, text_count= one_hot_all(text_vec, text_voc_size)
    sum_ex, sum_count = one_hot_all(sum_vec, sum_voc_size)
            
    np.savez('preprocessed_oh', text_word2vec = text_vec, summary_word2vec = sum_vec, text_existence=text_ex,text_count=text_count, 
                                        summary_existence=sum_ex,summary_count=sum_count, labels=labels, text_voc_size=text_voc_size, summary_voc_size=sum_voc_size,
                                )
vec2oh("./preprocessed.npz")