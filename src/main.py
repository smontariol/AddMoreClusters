import numpy as np
import pickle
import csv
import re
import os
from nltk import sent_tokenize
from collections import Counter
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

from extraction_for_BERT import get_embeddings_for_word
from clustering_BERTemb import cluster_word_embeddings_aff_prop, compute_jsd

def load_coha_sentences(decade, coha_path):
    coha_path += str(decade)
    print("Loading COHA sentences from", coha_path)
    coha_files = os.listdir(coha_path)
    sentences = []
    for coha_file in coha_files:
        if ".txt" in coha_file:
            coha_filepath = coha_path + '/' + coha_file
            try:
                text = open(coha_filepath, 'r').read().lower()
            except:
                text = open(coha_filepath, 'rb').read().decode('utf-8').lower()
            sentences.extend(sent_tokenize(text))
    return sentences


# Extract BERT embeddings for target words
gold_standard_file = "Gulordava_word_meaning_change_evaluation_dataset.csv"
f = open(gold_standard_file, 'r')
reader = csv.reader(f)
target_words = list(reader)
word_tuples = [(w[0], float(re.sub(",", ".", w[-1]))) for w in target_words[1:]]

# time slice 1
print("Timeslice 1")
decades1 = [1960]
sentences1 = []
for dec in decades1:
    print("Decade:", dec)
    sentences1.extend(load_coha_sentences(decade=dec, coha_path="corpora/COHA/text/"))

# time slice 2
print("Timeslice 2")
decades2 = [1990]
sentences2 = []
for dec in decades2:
    print("Decade:", dec)
    sentences2.extend(load_coha_sentences(decade=dec, coha_path="corpora/COHA/text"))

embeddings_dict = {}
for i, word_tuple in enumerate(word_tuples):
    word = word_tuple[0]
    print("\n=====", i, "word:", word.upper(), "=====\n")
    # todo attention: Here, doing this includes occurences of the word inside another word. Example: "against"-sentences are selected when looking for "gains"-sentences.
    #  Thus more sentences are selected than it should be. Gulardova's target words are not misleading, but adding this for generalisation purpose.
    # sentences_for_word1 = [s for s in sentences1 if word in re.sub("[^\w]", " ", s).split()]
    # sentences_for_word2 = [s for s in sentences2 if word in re.sub("[^\w]", " ", s).split()]
    sentences_for_word1 = [s for s in sentences1 if word in s.split()]
    sentences_for_word2 = [s for s in sentences2 if word in s.split()]

    if len(sentences_for_word1) > 0 and len(sentences_for_word2) > 0:

        embeddings1, valid_sentences1 = get_embeddings_for_word(word=word, sentences=sentences_for_word1)
        embeddings2, valid_sentences2 = get_embeddings_for_word(word=word, sentences=sentences_for_word2)

        if len(embeddings1) > 0 and len(embeddings2) > 0:
            embeddings_dict[word] = {"1960": embeddings1, "1990": embeddings2}

outfile = "bert_embeddings_gulordava_finetuned_epoch_5.pkl"
f = open(outfile, 'wb')
pickle.dump(embeddings_dict, f)
f.close()

# Evaluate against gold standard words
gold_standard_dict = {w[0]: float(re.sub(",", ".", w[-1])) for w in target_words[1:]}
embeddings_file = "bert_embeddings_gulordava_finetuned_epoch_5.pkl"
bert_embeddings = pickle.load(open(embeddings_file, 'rb'))
target_words = list(bert_embeddings.keys())

jsd_vec = []
cosine_dist_vec = []
gold_standard_vec = []
results_dict = {"word": [], "aff_prop": [], "gold_standard": [], "senses": []}
cluster_labels_dict = {}
cluster_centroids_dict = {}
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

    t1 = [counts1[i] for i in n_senses]
    t2 = [counts2[i] for i in n_senses]

    # compute JS divergence between count vectors by turning them into distributions
    t1_dist = np.array(t1)/np.array(t1).sum()
    t2_dist = np.array(t2)/np.array(t2).sum()

    jsd = compute_jsd(t1_dist, t2_dist)
    cosine_dist = 1.0-(cosine_similarity([t1_dist], [t2_dist])[0][0])
    cosine_dist_vec.append(cosine_dist)
    jsd_vec.append(jsd)
    gold_standard_vec.append(gold_standard_dict[word])

    print("JS divergence:", jsd)
    print("Gold standard:", gold_standard_dict[word])

    # add results to dataframe for saving
    cluster_labels_dict[word] = {"1960": clusters1, "1990": clusters2}
    cluster_centroids_dict[word] = centroids
    results_dict["word"].append(word)
    results_dict["aff_prop"].append(jsd)
    results_dict["gold_standard"].append(gold_standard_dict[word])
    results_dict["senses"].append(len(n_senses))


# Computr correlations between gold standard and aff prop clustering

pearson_corr = pearsonr(jsd_vec, gold_standard_vec)
print("Pearson correlation:", pearson_corr)

spearman_corr = spearmanr(jsd_vec, gold_standard_vec)
print("Spearman correlation:", spearman_corr)

