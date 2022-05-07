from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, to_tree, centroid, cut_tree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import TSNE

def compute_kmeans(X, no_clusters=3):
    """Compute K means of 3 groups by default"""
    kmeans = KMeans(n_clusters=no_clusters)
    res = kmeans.fit(X)
    #print(f"res: \n {res}")
    classes = kmeans.predict(X)
    #print(f"classes: {classes}")
    return classes

def convert_classes_to_clusters(classes):
    """takes in array of classes where each index indicates the class
    outputs a cluster list which is usuable by statistics"""
    labels =list()
    out = list()
    for i,cluster in enumerate(classes):
        if cluster not in labels:
            labels.append(cluster)
            out.append([i])
        else:
            idx = labels.index(cluster)
            out[idx].append(i)
    return labels, out