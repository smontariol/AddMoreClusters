from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import csv
import re
import pickle
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
cluster_centroids_dict = {}
n_clusters = [3]

print("Clustering BERT embeddings")
for i, word in enumerate(target_words):
    print("\n=======", i+1, "word:", word, "=======")
    emb = bert_embeddings[word]
    decades = sorted(list(emb.keys()))
    embeddings1 = emb[decades[0]]
    embeddings2 = emb[decades[1]]

    # cluster embeddings from t1 and t2
    embeddings_concat = np.concatenate([embeddings1, embeddings2], axis=0)
    cluster_labels, centroids = cluster_word_embeddings_aff_prop(embeddings_concat)

    # construct count vectors from the cluster labels
    clusters1 = list(cluster_labels[:embeddings1.shape[0]])
    clusters2 = list(cluster_labels[embeddings1.shape[0]:])
    counts1 = Counter(clusters1)
    counts2 = Counter(clusters2)
    n_senses = list(set(cluster_labels))
    print("Clusters:", len(n_senses))

    if len(n_senses) > 1:
        sil_score = silhouette_score(X=embeddings_concat, labels=cluster_labels)

    t1 = [counts1[i] for i in n_senses]
    t2 = [counts2[i] for i in n_senses]

    # compute JS divergence between count vectors by turning them into distributions
    t1_dist = np.array(t1)/np.array(t1).sum()
    t2_dist = np.array(t2)/np.array(t2).sum()

    jsd = compute_jsd(t1_dist, t2_dist)
    jsd_vec.append(jsd)
    cosine_dist = 1.0-(cosine_similarity([t1_dist], [t2_dist])[0][0])
    cosine_dist_vec.append(cosine_dist)
    gold_standard_vec.append(gold_standard_dict[word])

    # add results to dataframe for saving
    cluster_labels_dict[word] = {"1960": clusters1, "1990": clusters2}
    cluster_centroids_dict[word] = centroids
    results_dict["word"].append(word)
    results_dict["cosine_dist"].append(cosine_dist)
    results_dict["jsd"].append(jsd)
    results_dict["gold_standard"].append(gold_standard_dict[word])
    results_dict["senses"].append(len(n_senses))
    if sil_score is not None:
        results_dict["silhouette"].append(sil_score)
    else:
        results_dict["silhouette"].append(0)

# compute Spearman correlation between cosine distances obtained by our method and the human-annotated ground truth
mean_silhouette = np.mean(results_dict["silhouette"])
print("Mean silhouette score:", mean_silhouette)

pearson_corr = pearsonr(jsd_vec, gold_standard_vec)
print("Pearson correlation w/ JSD:", pearson_corr)

spearman_corr = spearmanr(jsd_vec, gold_standard_vec)
print("Spearman correlation w/ JSD:", spearman_corr)

# save everything
csv_file = "results_aff_prop_finetuned_epoch_5.csv"
labels_file = "labels_aff_prop_finetuned_epoch_5.pkl"
centroids_file = "centroids_aff_prop_finetuned_epoch_5.pkl"
# save results to CSV
results_df = pd.DataFrame.from_dict(results_dict)
results_df.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)
# save cluster labels to JSON
#print(cluster_labels_dict)
pf = open(labels_file, 'wb')
pickle.dump(cluster_labels_dict, pf)
pf.close()
pf2 = open(centroids_file, 'wb')
pickle.dump(cluster_centroids_dict, pf2)
pf2.close()
