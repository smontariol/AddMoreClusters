
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import sent_tokenize
import glob
import datetime as dt
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
import os
import re
import pickle
import json
import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
from src.extraction_for_BERT import *

def detect_drift(target_words, embeddings):
    drift_measure = []
    decades = sorted(list(embeddings[target_words[0]].keys()))

    for i, word in enumerate(target_words):
        print('word ' + str(i) + '/' + str(len(target_words)) + ': ' + word)
        # print("\n======= Word:", word, "=======")
        emb60 = embeddings[word][decades[0]]
        emb70 = embeddings[word][decades[1]]
        emb80 = embeddings[word][decades[2]]
        emb90 = embeddings[word][decades[3]]

        np.random.shuffle(embeddings)
        mean_emb = np.mean(embeddings, 0)  # average embedding on the full corpus
        # Variation coefficient: mean cosine distance between all embeddings of a word occurrences in a corpus and their average
        variation_coef = 1 - np.mean(cosine_similarity(mean_emb.reshape(1, -1), embeddings))

        # Variation coef evolution:
        var_coef_60 = 1 - np.mean(cosine_similarity(np.mean(emb60, 0).reshape(1, -1), emb60))
        var_coef_70 = 1 - np.mean(cosine_similarity(np.mean(emb70, 0).reshape(1, -1), emb70))
        var_coef_80 = 1 - np.mean(cosine_similarity(np.mean(emb80, 0).reshape(1, -1), emb80))
        var_coef_90 = 1 - np.mean(cosine_similarity(np.mean(emb90, 0).reshape(1, -1), emb90))
        var_coef_evol = ((var_coef_70 - var_coef_60) + (var_coef_80 - var_coef_70) + (var_coef_90 - var_coef_80))/3

        # drift : similarity between the mean embeddings of the first and last decade (as in Martinc et al. 2019)
        total_drift = 1 - cosine_similarity(np.mean(emb90, 0).reshape(1, -1), np.mean(emb60, 0).reshape(1, -1))

        # similarity between each decade and the following one, summed.
        sum_drifts = (1 - cosine_similarity(np.mean(emb90, 0).reshape(1, -1), np.mean(emb80, 0).reshape(1, -1))) \
                     + (1 - cosine_similarity(np.mean(emb80, 0).reshape(1, -1), np.mean(emb70, 0).reshape(1, -1))) \
                     + (1 - cosine_similarity(np.mean(emb70, 0).reshape(1, -1), np.mean(emb60, 0).reshape(1, -1)))
        # mean of all the similarities
        mean_drifts = sum_drifts / (len(decades) - 1)

        drift_measure.append((word, variation_coef, var_coef_evol, total_drift.flatten().item(), mean_drifts.flatten().item()))

    drifts = pd.DataFrame(drift_measure, columns=("word", "variation_coef", "var_coef_evol", "total_drift", "mean_drifts"))

    outfile = "variation_measures.pkl"
    f = open(outfile, 'wb')
    pickle.dump(drift_measure, f)
    f.close()

    return drift_measure
