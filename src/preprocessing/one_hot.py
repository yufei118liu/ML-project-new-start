
## Import Dependencies
import numpy as np
from keras.preprocessing.sequence import pad_sequences


data = np.load('./preprocessed.npz', allow_pickle=True)
x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']
labels = data["labels"]
max_text_len = data['max_text_len']
max_summary_len = data['max_summary_len']
x_voc_size = data['x_voc_size']
y_voc_size = data['y_voc_size']

def one_hot_enc(ar):
    #print(ar)
    n = np.max(ar) +1
    i = np.sum(np.eye(n)[ar], axis= 0)
    bl = np.array(i, dtype=bool)
    oh = bl.astype(int)
    #padded = pad_sequences(oh, maxlen=x_voc_size, padding='post', truncating='post')
    return oh

def one_hot_all(x_train):
    #print(len(x_train))
    #print(len(x_train[0]))
    #print(x_train[0])
    trains_oh = []
    for each in x_train:
        trains_oh.append(one_hot_enc(each))
    return trains_oh

x_train= one_hot_all(x_train)

y_train = one_hot_all(y_train)
x_test= one_hot_all(x_train)
y_test = one_hot_all(y_train)
np.savez('preprocessed_oh', x_train=x_train, x_test=x_test, y_train=y_train,y_test=y_test, labels=labels,
                        max_text_len=max_text_len, max_summary_len=max_summary_len, x_voc_size=x_voc_size, y_voc_size=y_voc_size)