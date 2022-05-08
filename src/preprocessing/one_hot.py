
## Import Dependencies
from sklearn.feature_extraction.text import CountVectorizer
import os
import sys
if __name__ == "__main__":
    path = "./data/"
    filelist = [path+file for file in os.listdir("./data")]
    #for each in filelist:
        #print(each)
        #with open(each) as f:
    
    vectorizer = CountVectorizer(input='filename', strip_accents='ascii', stop_words='english')
    X = vectorizer.fit_transform(filelist)
        #f.close()
    print(X.toarray().all() == 0)