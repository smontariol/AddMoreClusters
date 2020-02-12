from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter
from scipy.stats import entropy
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import csv
import re
import pickle
import json
import pandas as pd
from src.clustering_BERTemb import *

# perform clustering and other methods on saved BERT embeddings
gold_standard_file = "Gulordava_word_meaning_change_evaluation_dataset.csv"
f = open(gold_standard_file, 'r')
reader = csv.reader(f)
target_words = list(reader)
gold_standard_dict = {w[0]: float(re.sub(",", ".", w[-1])) for w in target_words[1:]}

embeddings_file = "bert_embeddings_gulordava_finetuned_epoch_5.pkl"
bert_embeddings = pickle.load(open(embeddings_file, 'rb'))
target_words = list(bert_embeddings.keys())

jsd_vec = []
cosine_dist_vec = []
gold_standard_vec = []
results_dict = {"word": [], "cosine_dist": [], "jsd": [], "gold_standard": [], "senses": [], "silhouette":[]}
cluster_labels_dict = {}
n_clusters = 7
thresh_t1 = 2
thresh_t2 = 2

print("Clustering BERT embeddings")
for i, word in enumerate(target_words):
    print("\n======= Word", i, ":", word, "=======")
    emb = bert_embeddings[word]
    decades = sorted(list(emb.keys()))
    embeddings1 = emb[decades[0]]
    embeddings2 = emb[decades[1]]

    # cluster embeddings from t1 and t2
    embeddings_concat = np.concatenate([embeddings1, embeddings2], axis=0)
    cluster_labels, cluster_centroids = cluster_word_embeddings_aff_prop(embeddings_concat)
    centroid_sim = cosine_similarity(cluster_centroids)
    cluster_neighbors = compute_nearest_cluster(centroid_sim)

    # construct count vectors from the cluster labels
    clusters1 = list(cluster_labels[:embeddings1.shape[0]])
    clusters2 = list(cluster_labels[embeddings1.shape[0]:])
    cluster_labels_dict[word] = {"1960": clusters1, "1990": clusters2}
    counts1 = Counter(clusters1)
    counts2 = Counter(clusters2)
    merged_cluster1 = merge_clusters(counts1, cluster_neighbors, weak_cluster_thresh=thresh_t1)
    merged_cluster2 = merge_clusters(counts2, cluster_neighbors, weak_cluster_thresh=thresh_t2)

    n_senses = list(set(cluster_labels))

    #if len(n_senses) > 1:
    #    sil_score = silhouette_score(X=embeddings_concat, labels=cluster_labels)

    t1 = [counts1[i] for i in n_senses]
    t2 = [counts2[i] for i in n_senses]

    print("t1:", t1)
    print("t2:", t2)

    # compute JS divergence between count vectors by turning them into distributions
    t1_dist = np.array(t1)/np.array(t1).sum()
    t2_dist = np.array(t2)/np.array(t2).sum()

    jsd = compute_jsd(t1_dist, t2_dist)
    cosine_dist = 1.0 - (cosine_similarity([t1_dist], [t2_dist])[0][0])
    cosine_dist_vec.append(cosine_dist)
    jsd_vec.append(jsd)
    gold_standard_vec.append(gold_standard_dict[word])


corr = spearmanr(jsd_vec, gold_standard_vec)
print("Spearman correlation w/ JSD:", corr)

corr = pearsonr(jsd_vec, gold_standard_vec)
print("Pearson correlation w/ JSD:", corr)

print("Saving cluster labels file")
labels_file = "labels_2stage_aff_prop_finetuned_epoch_5.pkl"
pf = open(labels_file, 'wb')
pickle.dump(cluster_labels_dict, pf)
pf.close()
