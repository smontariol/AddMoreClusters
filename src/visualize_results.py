import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


def visualize_word_clusters():
    embedding_file = "bert_embeddings_gulordava_finetuned_epoch_5.pkl"
    clustering_file = "labels_aff_prop_finetuned_epoch_5.pkl"
    sentences_file = "coha_sentences_gulordava.pkl"
    centroids_file = "centroids_aff_prop_finetuned_epoch_5.pkl"
    results_file = "results_aff_prop_finetuned_epoch_5.csv"
    cluster_data = pickle.load(open(clustering_file, 'rb'))
    #results_data = pickle.load(open(results_file, 'r'))
    centroids = pickle.load(open(centroids_file, 'rb'))
    embeddings = pickle.load(open(embedding_file, 'rb'))
    sentences = pickle.load(open(sentences_file, 'rb'))

    #target_words = ['vector']
    #for word in target_words:
    word = 'neutron'
    embeddings_t0 = embeddings[word]['1960']
    embeddings_t1 = embeddings[word]['1990']
    labels_t0 = cluster_data[word]['1960']
    labels_t1 = cluster_data[word]['1990']
    sentences_t0 = sentences[word]['1960']
    sentences_t1 = sentences[word]['1990']
    df = {}
    df['cluster'] = np.concatenate([labels_t0, labels_t1], axis=0)
    df['sentences'] = np.concatenate([sentences_t0, sentences_t1], axis=0)
    df['time_slice'] = np.concatenate([['1960']*len(labels_t0), ['1990']*len(labels_t1)], axis=0)
    df = pd.DataFrame.from_dict(df)
    counts = Counter(df['cluster'])
    n_clusters = len(counts)
    combined_embeddings = np.concatenate([embeddings_t0, embeddings_t1], axis=0)
    centroids_word = centroids[word]
    # for each cluster, find instance that is nearest to the centroid
    valid_clusters = []
    for cluster_index in range(n_clusters):
        instance_indexes = list(df[df['cluster'] == cluster_index].index)
        if len(instance_indexes) > 15:
            valid_clusters.append(cluster_index)
        cluster_centroid = centroids_word[cluster_index]
        smallest_dist = 1.0
        nearest_instance = 0
        for instance_index in instance_indexes:
            dist = 1.0-(cosine_similarity([cluster_centroid], [combined_embeddings[instance_index]])[0][0])
            if dist < smallest_dist:
                nearest_instance = instance_index
                smallest_dist = dist
        print("Cluster:", cluster_index)
        print("Instances:", len(instance_indexes))
        #print("Nearest centroid instance:", nearest_instance)
        print("Nearest centroid sentence:", df['sentences'][nearest_instance])

        pca = PCA(n_components=2)
        arr = np.array(combined_embeddings)
        pca_result = pca.fit_transform(arr)
        df['PCA1'] = pca_result[:,0]
        df['PCA2'] = pca_result[:,1]
        #df2 = df[(df['clusters'] !=4) & (df['clusters'] != 5) & (df['clusters'] != 6) & (df['clusters'] != 9)]
        df2 = df[df.cluster.isin(valid_clusters)]
        plt.clf()
        plt.figure(figsize=(10,10))
        # scatterplot the two principal components
        n_colors = len(set(df2['cluster']))
        #print("colors: ", n_colors)
        #print(df)
        ax = sns.scatterplot(
            x='PCA1',
            y='PCA2',
            hue='cluster',
            s=50,
            style='time_slice',
            palette=sns.color_palette("Set1", n_colors),
            data=df2,
            legend='full',
            alpha=1.0
        )
        fig = ax.get_figure()
        fig.savefig('figures/word_clusters/' + word + '.png')
        plt.close()



def visualize_frequencies():
    # load files
    freq_file = "word_frequency.csv"
    freq_df = pd.read_csv(freq_file, delimiter="\t")

    results_file = "results_aff_prop_finetuned_epoch_5.csv"
    res_df = pd.read_csv(results_file, delimiter="\t")

    centroids_file = open("centroids_aff_prop_finetuned_epoch_5.pkl", 'rb')
    centroids = pickle.load(centroids_file)

    # merge and compute correlations
    merged_df = pd.merge(res_df, freq_df, how='left', on=['word'])
    merged_df['freq'] = merged_df['1960']+merged_df['1990']

    spearman_corr = spearmanr(merged_df['freq'], merged_df['senses'])
    print("Spearman correlation:", spearman_corr)

    pearson_corr = pearsonr(merged_df['freq'], merged_df['senses'])
    print("Pearson correlation:", pearson_corr)


    # plot
    sorted_df = merged_df.sort_values(by=['freq'])
    splot = sns.regplot(x="freq", y="senses",
                        data=sorted_df, fit_reg=False)
    splot.set(xlabel='frequency', ylabel='clusters', xscale="log")

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            if point['val'] in words_to_label:
                ax.text(point['x']+.05, point['y']+0.5, str(point['val']))

    words_to_label = ['negligence', 'woman', 'contact', 'disk', 'cent', 'gain']
    label_point(sorted_df['freq'], sorted_df['senses'], sorted_df['word'], plt.gca())

    plt.show()
    #scatter_ax.set(xlabel='clusters', ylabel='frequency')


    # ax = sns.distplot(a=res_df["senses"], hist=True, kde=False, rug=False)
    # ax.set(xlabel='clusters', ylabel='words')
    # plt.show()
    #
    # freq_ax = sns.distplot(a=(freq_df['1960'] + freq_df['1990']), hist=True, kde=False, rug=False)
    # freq_ax.set(xlabel='frequency', ylabel='words')
    # plt.show()

visualize_word_clusters()