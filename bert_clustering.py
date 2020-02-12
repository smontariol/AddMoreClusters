import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from nltk.stem.wordnet import WordNetLemmatizer
import csv
import pickle

from src.extraction_for_BERT import *

lemma = WordNetLemmatizer()
# # Load pre-trained model (weights)
# print("Loading BERT pre-trained models")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# Load fine-tuned model from Matej
print("Load BERT fine-tuned model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
state_dict = torch.load("COHA_BERT/model_coha_epoch_3/pytorch_model.bin", map_location=torch.device('cpu'))
model = BertModel.from_pretrained('bert-base-uncased', state_dict=state_dict)
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

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
    sentences1.extend(load_coha_sentences(decade=dec))

# time slice 2
print("Timeslice 2")
decades2 = [1990]
sentences2 = []
for dec in decades2:
    print("Decade:", dec)
    sentences2.extend(load_coha_sentences(decade=dec))

cosine_dist_vec = []
jsd_vec = []
ground_truth_vec = []
embeddings_dict = {}
sentences_dict = {}

for i, word_tuple in enumerate(word_tuples):
    word = word_tuple[0]
    print("\n=====", i, "word:", word.upper(), "=====\n")
    # todo attention: Here, doing this includes occurences of the word inside another word. Example: "against"-sentences are selected when looking for "gains"-sentences.
    #  Thus more sentences are selected than it should be. Gulardova's target words are not misleading, but adding this for generalisation purpose.
    sentences_for_word1 = [s for s in sentences1 if word in re.sub("[^\w]", " ", s).split()]
    sentences_for_word2 = [s for s in sentences2 if word in re.sub("[^\w]", " ", s).split()]

    if len(sentences_for_word1) > 0 and len(sentences_for_word2) > 0:

        embeddings1, valid_sentences1 = get_embeddings_for_word(word=word, sentences=sentences_for_word1)
        embeddings2, valid_sentences2 = get_embeddings_for_word(word=word, sentences=sentences_for_word2)

        if len(embeddings1) > 0 and len(embeddings2) > 0:
            embeddings_dict[word] = {"1960": embeddings1, "1990": embeddings2}
            sentences_dict[word] = {"1960": valid_sentences1, "1990": valid_sentences2}

outfile = "bert_embeddings.pkl"
f = open(outfile, 'wb')
pickle.dump(embeddings_dict, f)
f.close()
print("***** Done saving embeddings to", outfile, "! *****")


