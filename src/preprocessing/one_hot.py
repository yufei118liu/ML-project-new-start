
## Import Dependencies

import numpy as np

from keras.preprocessing.sequence import pad_sequences


import tensorflow as tf


from multiprocessing import Pool


'''data = np.load('./preprocessed_topic_oh.npz', allow_pickle=True)
#text_vec = data['text_word2vec']
summary_vec = data['summary_word2vec']
#text_existence = data['text_existence']
#text_count = data['text_count']
summary_existence = data['summary_existence']
summary_count = data['summary_count']
labels = data["labels"]
#text_voc_size = data['text_voc_size']
summary_voc_size = data['summary_voc_size']

#print(text_vec[:3])
#print(summary_existence[:3])
#print(summary_count[:3])
print(len(summary_existence))
print(labels[:3])
#print([ex.shape for ex in summary_existence] )
'''
def one_hot_enc(tup):
    ar, dic_size = tup
    #print(ar)
    n = np.max(ar) +1
    #count = np.sum(np.eye(dic_size)[ar], axis= 0)
    count = np.sum(tf.one_hot(ar, n), axis= 0)/len(ar)
    bl = np.array(count, dtype=bool)
    ex = bl.astype(int)
    
    #padded = pad_sequences(oh, maxlen=x_voc_size, padding='post', truncating='post')
    return count, ex

def one_hot_all(input, dic_size):
    #print(len(x_train))
    #print(len(x_train[0]))
    #print(x_train[0])
    print("one hot all")
    existence = []
    occurrence = []
    p = Pool()
    vector_input = [(each, dic_size) for each in input]
    result = p.map(one_hot_enc,vector_input)
    for each in result:
        count, ex = each
        existence.append(ex)
        occurrence.append(count)

    ex = pad_sequences(existence, maxlen=dic_size, padding='post', truncating='post', dtype=float)
    count = pad_sequences(occurrence, maxlen=dic_size, padding='post', truncating='post', dtype=float)
    return ex, count

data = np.load('./preprocessed_topic.npz', allow_pickle=True)

summary_vec = data['summary_word2vec']
summary_voc_size = data['summary_voc_size']
labels = data["labels"]

sum_ex, sum_count = one_hot_all(summary_vec, summary_voc_size)
print(sum_ex, sum_count)
np.savez('preprocessed_topic_oh', summary_word2vec = summary_vec, labels=labels, 
                                        summary_existence=sum_ex,summary_count=sum_count, summary_voc_size=summary_voc_size,
                                )

#print(summary_vec[:3])


