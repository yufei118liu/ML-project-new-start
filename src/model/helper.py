from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list, to_tree, centroid, cut_tree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA,KernelPCA
from sklearn.manifold import TSNE
import pandas as pd
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

def plot_tsne(matrix,chains_list,title="",perplexity = 50):
    """Reduces matrix to 2 dimensions using TSNE and plots it"""
    reduced_matrix =TSNE(n_components=2,init='pca',method='exact',perplexity=perplexity).fit_transform(matrix)
    axes =reduced_matrix.T

    plt.figure(figsize=(20,20))
    axis1 =axes[0].tolist()
    axis2 =axes[1].tolist()
    plt.scatter(axis1,axis2)
    for i,label in enumerate(chains_list):
        plt.annotate(label, (axis1[i], axis2[i]))
    plt.title(title)
    return reduced_matrix

def plot_kpca(matrix,labels, kernel="linear",title=""):
    """Reduces matrix to 2 dimensions using PCA and plots it
    kernel choices: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}
    """
    np.matrix(matrix)
    pca = KernelPCA(n_components=2,kernel=kernel)
    reduced_matrix=pca.fit_transform(matrix)
    print(f"shape reduced matrix: {np.shape(reduced_matrix)}")
    
    axes =reduced_matrix.T

    plt.figure(figsize=(20,20))
    axis1 =axes[0].tolist()
    axis2 =axes[1].tolist()
    plt.scatter(axis1,axis2)
    for i,label in enumerate(labels):
        plt.annotate(label, (axis1[i], axis2[i]))
    plt.title(title)
    return reduced_matrix

def plot_dendrogram(matrix, hierarchy_method = "complete",dist_metric = "cos", title= "", labels = None):
    """Plots dendro gram given matrix with parameters for the linkage
    labels: name labels on the dendrogram tree"""
    out = linkage(matrix, method = hierarchy_method, metric = dist_metric)
    plt.figure(figsize=(130,60))
    dn = dendrogram(out, labels = labels)
    plt.show()
    
def plot_pca(matrix,labels,title=""):
    """Reduces matrix to 2 dimensions using PCA and plots it"""
    np.matrix(matrix)
    pca = PCA(n_components=2)
    reduced_matrix=pca.fit_transform(matrix)
    print(f"shape reduced matrix: {np.shape(reduced_matrix)}")
    
    axes =reduced_matrix.T

    plt.figure(figsize=(20,20))
    axis1 =axes[0].tolist()
    axis2 =axes[1].tolist()
    plt.scatter(axis1,axis2)
    for i,label in enumerate(labels):
        plt.annotate(label, (axis1[i], axis2[i]))
    plt.title(title)
    return reduced_matrix

def heat_map(leaves_list,labels,matrix):
    """Prints heat map where rows are ordered by the clustering algorithm,
    columns are still chains list ordered"""
    rows = [labels[i] for i in leaves_list]
    ordered_mat = [matrix[i] for i in leaves_list]
    #print(f"rows: {rows}, chains_list {chains_list}")
    heat_frame = pd.df(ordered_mat,rows,labels)
    print(f"starting heat function 2")
    #f, ax = plt.subplots(figsize=(11, 9))
    #cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plt.figure(figsize=(1000,1000))
    plt.xticks(range(len(labels)),labels,rotation=90)
    plt.yticks(range(len(rows)),rows)
    plt.imshow(heat_frame, cmap='hot',interpolation="nearest")