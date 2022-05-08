import numpy as np
data = np.load('./preprocessed_oh.npz', allow_pickle=True)
#text_vec = data['text_word2vec']
summary_vec = data['summary_word2vec']
#text_existence = data['text_existence']
#text_count = data['text_count']
summary_existence = data['summary_existence']
summary_count = data['summary_count']
labels = data["labels"]
#text_voc_size = data['text_voc_size']
summary_voc_size = data['summary_voc_size']
print(len(summary_vec))
print(summary_count[:10])