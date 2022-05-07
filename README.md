# ML-project-new-start

# preprocessing

The preprocessing has one most important function, data_processing. It the following arguments:
 - directory: the directory of data to preprocess, by default './data'
 - one_hot: whether to output both normal and one_hot vectors. By default is False, which means only normal vectors are saved. The normal vectors are saved in 'preprocessed.npz', while the one_hot vectors are in 'preprocessed_oh.npz'. 
 -limited: whether to limit the number of input articles. For debugging purpose. By default is False. If True, only the first 10 articles will be preprocessed. 


 Further description on how to load the output. 
 If only normal vectors:
    data = np.load('./preprocessed.npz', allow_pickle=True)
    text_vec = data['text_word2vec']
    summary_vec = data['summary_word2vec']
    labels = data["labels"]
    text_voc_size = data['text_voc_size']
    summary_voc_size = data['summary_voc_size']
If with one_hot vectors:
    data = np.load('./preprocessed_oh.npz', allow_pickle=True)
    text_vec = data['text_word2vec']
    summary_vec = data['summary_word2vec']
    text_existence = data['text_existence']
    text_count = data['text_count']
    summary_existence = data['summary_existence']
    summary_count = data['summary_count']
    labels = data["labels"]
    text_voc_size = data['text_voc_size']
    summary_voc_size = data['summary_voc_size']

Note that one can also generate one_hot vectors based on a 'preprocessed.npz' file by calling vec2oh('preprocessed.npz')


