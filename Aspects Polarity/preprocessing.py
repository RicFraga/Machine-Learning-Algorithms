""" This code has 3 main purposes

    1) Join the reviews in one unique string and segment it in sentences
    2) Once an aspect is chosen, we are going to find the strings which contain it and store them
    3) We are going to get the polarity value of all the sentences which contained the aspect
    4) Display the aspect polarity

This is going to be done for 7 aspects, I'm going to use the music reviews
"""

import spacy
import pickle
from os import scandir
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

path_to_data = "/home/ricardo/Documentos/Materias/NLP/Detección de Polaridad por Aspectos/SFU_Spanish_Review_Corpus/musica/"
n_grams = 2

# Function to get the features we are going to use
def compute_ngrams(sequence, n):
    return(zip(*[sequence[index:]
            for index in range(n)]))

# Function to clean text
def remove_stopwords(text):
    spanish_sw = stopwords.words('spanish')

    return [word for word in text if not word in spanish_sw]

# Function to normalize (lemmatize) our data and leave only nouns
def lemmatize(text, only_nouns):
    nlp = spacy.load('es_core_news_sm')
    doc = nlp(text)

    if(only_nouns == True):
        return ([word.lemma_ for word in doc if word.pos_ == 'NOUN'])

    else:
        return ' '.join([word.lemma_ for word in doc])

# Function to get the elements of a directory
def ls2(path):
    return [obj.name for obj in scandir(path) if obj.is_file()]

# Function to read the data from 'path_to_data', it returns all the reviews in one single string
def read_data(path):
    list_of_files = ls2(path)

    string_total_reviews = ""

    for file in list_of_files:
        f = open(path + file, encoding='latin-1')
        content = f.read()
        f.close()

        string_total_reviews = string_total_reviews + content

    return string_total_reviews

# Function to create a dictionary with the ngram and the occurence
def count_ngrams(ngrams):
    return Counter(ngrams)

# Function to save the sentences of all the reviews
def all_reviews_to_sentences(data):
    sentences = str(sent_tokenize(data))
    lemmatized_sentences = lemmatize(sentences, False)

    f = open("sentences", "ab")
    pickle.dump(lemmatized_sentences, f)
    f.close()

# Getting the string
print("Getting the data and transforming it into a single string...")
data = read_data(path_to_data)
sentences = all_reviews_to_sentences(data)

# Lemmatizing the string
print("Lemmatizing the string...")
lemmatized_text = lemmatize(data, True)

# Cleaning data
print("Cleaning text...")
clean_data = remove_stopwords(data)

# Getting the ngrams
print("Computing bigrams...")
bigrams = list(compute_ngrams(lemmatized_text, n_grams))

# Counting frequent ngrams
counts = dict(count_ngrams(bigrams))
ordered_counts = sorted(counts.items(), key = lambda x: x[1], reverse=True)

# Saving the data
print("Saving the data...")

f = open("Bigramas", "w")
for count in ordered_counts:
    #f.write(str(count[0][0]) + ": " + str(count[1]) + "\n")
    f.write(
    str(count[0][0]) +
    " " +
    str(count[0][1]) +
    ": " +
    str(count[1]) +
    "\n"
    )
f.close()
