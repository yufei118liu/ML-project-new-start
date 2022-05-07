# ML-project-new-start

# General:
1) Modify list of queries (right now contains Math article data so don't)
2) run ```python3 main.py``` to scrape the articles if main is empty
3) Go to ```src/preprocessing``` run ```python3 preprocessing.py``` to get npz data file (preprocessed articles), file is too large to put on github to do this
4) Create your personal note book and load the ```preprocessed.npz``` similar to model_yao.ipynb and do experiments
5) Create 2d vector embedding from either summary or whole text
6) Cluster them!

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


