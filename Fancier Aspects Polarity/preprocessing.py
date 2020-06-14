"""
This preprocessing code has 3 main purposes:
* Get the no and yes reviews, each one in a single string
* Lemmatize each string
* Sent tokenize each string and save it for the next operations



"""

import spacy
import pickle
from os import listdir
from os.path import isfile
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

path_to_reviews = "/home/ricardo/Documentos/Materias/NLP/SFU_Spanish_Review_Corpus/musica/"

def ls1(path):
    return [obj for obj in listdir(path) if isfile(path + obj)]

def get_groups(path):
    files = ls1(path)

    yes_files = []
    no_files = []

    for file in files:
        if('yes' in file):
            yes_files.append(file)

        else:
            no_files.append(file)

    return yes_files, no_files

# Function to normalize (lemmatize) our data and leave only nouns
def lemmatize(text, only_nouns):
    nlp = spacy.load('es_core_news_sm')
    doc = nlp(text)

    if(only_nouns == True):
        return [word.lemma_ for word in doc if word.pos_ == 'NOUN']

    else:
        return ' '.join([word.lemma_ for word in doc])

# Function to read the data from 'path_to_data', it returns all the reviews in one single string
def read_data(file_names, origin):
    string_total_reviews = ""

    for file in file_names:
        f = open(origin + file, encoding='latin-1')
        content = f.read()
        f.close()

        string_total_reviews = string_total_reviews + content

    return string_total_reviews

# Function to save the sentences of all the reviews
def reviews_to_sentences(data):
    sentences = str(sent_tokenize(data))
    lemmatized_sentences = lemmatize(sentences, False)

    return sent_tokenize(lemmatized_sentences)

# Function to clean text
def remove_stopwords(text):
    spanish_sw = stopwords.words('spanish')

    return [word for word in text if not word in spanish_sw]

print("Getting the groups of yes/no files...")
yes_files, no_files = get_groups(path_to_reviews)

print("Reading the yes data...")
yes_string = read_data(yes_files, path_to_reviews)

print("Lemmatizing the yes data and saving it into sentences...")
yes_sentences = reviews_to_sentences(yes_string)

print("Saving the yes data...")
f = open("yes_sentences", 'ab')
pickle.dump(yes_sentences ,f)
f.close()

# Free memory
del yes_sentences

print("Reading the no data...")
no_string = read_data(no_files, path_to_reviews)

print("Lemmatizing the no data and saving it into sentences...")
no_sentences = reviews_to_sentences(no_string)

print("Saving the no data...")
f = open("no_sentences", 'ab')
pickle.dump(no_sentences ,f)
f.close()

# Free memory (unnecessary at this point but ok)
del no_sentences
