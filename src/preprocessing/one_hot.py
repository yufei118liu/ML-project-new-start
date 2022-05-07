
## Import Dependencies
import numpy as np


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

def one_hot_enc(ar, dic_size, tokenizer):
    n = np.max(ar) +1
    if n>dic_size:
        print(ar)
        print(tokenizer.sequence_to_text())
        exit()
    i = np.sum(np.eye(dic_size)[ar], axis= 0)
    bl = np.array(i, dtype=bool)
    oh = bl.astype(int)
    return oh

def one_hot_all(x_train, dic_size, tokenizer):
    #print(len(x_train))
    #print(len(x_train[0]))
    #print(x_train[0])
    trains_oh = []
    flag = False
    for i, each in enumerate(x_train):
        if i == 1286:
            break
        

        trains_oh.append(one_hot_enc(each, dic_size))
        print(i)
    return trains_oh

x_train= one_hot_all(x_train, x_voc_size)
y_train = one_hot_all(y_train, y_voc_size)
x_test= one_hot_all(x_train, x_voc_size)
y_test = one_hot_all(y_train, y_voc_size)
np.savez('preprocessed_oh', x_train=x_train, x_test=x_test, y_train=y_train,y_test=y_test, labels=labels,
                        max_text_len=max_text_len, max_summary_len=max_summary_len, x_voc_size=x_voc_size, y_voc_size=y_voc_size)