
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.stats import entropy
import numpy as np
from extraction_for_BERT import *


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def cluster_word_embeddings_aff_prop(word_embeddings):
    #print("Clustering word embeddings with Affinity Propagation")
    clustering = AffinityPropagation().fit(word_embeddings)
    labels = clustering.labels_
    counts = Counter(labels)
    print("Num of clusters:", len(counts))
    exemplars = clustering.cluster_centers_
    return labels, exemplars


def cluster_word_embeddings_dbscan(word_embeddings):
    #print("Clustering word embeddings with DBSCAN")
    clustering = DBSCAN().fit(word_embeddings)
    labels = clustering.labels_
    counts = Counter(labels)
    print("Num of clusters:", len(counts))
    return labels


def cluster_word_embeddings_k_means(word_embeddings, k=3):
    #print("Clustering word embeddings with KMeans, k =", k)
    clustering = KMeans(n_clusters=k, random_state=0).fit(word_embeddings)
    labels = clustering.labels_
    centroids = clustering.cluster_centers_
    return labels, centroids


def compute_nearest_cluster(centroid_sim):
    print("Getting neighboring clusters")
    n_clusters = centroid_sim.shape[0]
    neighbors = []
    for c in range(n_clusters):
        ranking = [w for _, w in sorted(zip(centroid_sim[c], range(n_clusters)), reverse=True)]
        neighbors.append(ranking)
    #print("Neighbors:", neighbors)
    return neighbors


def merge_clusters(cluster_counts, cluster_neighbors, weak_cluster_thresh):
    print("Merging clusters")
    print(cluster_counts)
    merged_cluster = {}
    for k in cluster_counts:
        if cluster_counts[k] < weak_cluster_thresh:
            nearest_cluster_found = False
            nearest_count = 0
            while nearest_cluster_found is False and nearest_count < len(cluster_neighbors):
                nearest_cluster = cluster_neighbors[k][nearest_count]
                if nearest_cluster in cluster_counts and cluster_counts[nearest_cluster] > 0:
                    nearest_cluster_found = True
                    #cluster_counts[nearest_cluster] += cluster_counts[k]
                    merged_cluster[nearest_cluster] = cluster_counts[k] + cluster_counts[nearest_cluster]
                    cluster_counts[k] = 0

                else:
                    nearest_count += 1
        else:
            merged_cluster[k] = cluster_counts[k]
    return merged_cluster


def compute_divergence_from_cluster_labels(labels1, labels2):
    labels_all = list(np.concatenate((labels1, labels2)))
    counts1 = Counter(labels1)
    counts2 = Counter(labels2)
    n_senses = list(set(labels_all))
    #print("Clusters:", len(n_senses))

    t1 = np.array([counts1[i] for i in n_senses])
    t2 = np.array([counts2[i] for i in n_senses])

    # compute JS divergence between count vectors by turning them into distributions
    t1_dist = t1/t1.sum()
    t2_dist = t2/t2.sum()

    jsd = compute_jsd(t1_dist, t2_dist)
    print("clustering JSD:", jsd)
    return jsd
