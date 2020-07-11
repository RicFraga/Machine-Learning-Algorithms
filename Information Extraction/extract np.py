import time
import nltk
import spacy
from os import listdir
from os.path import isfile, isdir
from nltk.tokenize import word_tokenize, sent_tokenize

path = 'musica/'
nlp = spacy.load("es_core_news_sm")

# Function to list the files in a directory from:
# https://www.altaruru.com/python-como-listar-los-archivos-de-un-directorio/
def ls1(path):
    return [obj for obj in listdir(path) if isfile(path + obj)]

# This function takes all the reviews and joins them in one single string
def join_reviews(file_names, path):
    # List to store all the files
    unique = []

    for file in file_names:
        f = open(path + file, encoding = 'latin-1')
        unique.append(f.read())
        f.close()

    return ''.join(unique)

# This function builds an array of tuples, each tuple contains the word and
# the pos-tag of the word
def pos_tag(sentences):

    tagged = []

    for sentence in sentences:
        doc = nlp(sentence)

        sentence_tagged = []

        for token in doc:
            cut = token.tag_.index('_')
            sentence_tagged.append((str(token), str(token.tag_[:cut])))

        tagged.append(sentence_tagged)

    return tagged

inicio = time.time()

# Getting the files in the directory
files = ls1(path)

# Getting all the reviews in one single string
single_string = join_reviews(files, path)

# Now we are going to tokenize the string in sentences
sentences = sent_tokenize(single_string)

# Then we are going to pos-tag the words of each sentence
pos_tag = pos_tag(sentences)

# This regular expression indentifies the noun phrases
grammar = """NP:
                {<NOUN><ADP><PROPN>}
                {<ADJ>*<NOUN><ADP><NOUN>}
                {<DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN>}
                {<ADV><CCONJ><ADV>*<ADJ>}
          """

cp = nltk.RegexpParser(grammar)

# Saving the np to a file
f = open('noun phrases', 'w')

for tag in pos_tag:
    try:
        result = cp.parse(tag)
    except:
        pass

    for i in result.subtrees():
        if(i.label() == 'NP'):
            f.write(str(i) + '\n')

f.close()

# Getting the phrases of the form
# np + de + np

grammar = """PS:
                {<NOUN><ADP><PROPN><ADP><NOUN><ADP><PROPN>}
                {<NOUN><ADP><PROPN><ADP><ADJ>*<NOUN><ADP><NOUN>}
                {<NOUN><ADP><PROPN><ADP><DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN>}
                {<NOUN><ADP><PROPN><ADP><ADV><CCONJ><ADV>*<ADJ>}

                {<ADJ>*<NOUN><ADP><NOUN><ADP><NOUN><ADP><PROPN>}
                {<ADJ>*<NOUN><ADP><NOUN><ADP><ADJ>*<NOUN><ADP><NOUN>}
                {<ADJ>*<NOUN><ADP><NOUN><ADP><DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN>}
                {<ADJ>*<NOUN><ADP><NOUN><ADP><ADV><CCONJ><ADV>*<ADJ>}

                {<DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN><ADP><NOUN><ADP><PROPN>}
                {<DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN><ADP><ADJ>*<NOUN><ADP><NOUN>}
                {<DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN><ADP><DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN>}
                {<DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN><ADP><ADV><CCONJ><ADV>*<ADJ>}

                {<ADV><CCONJ><ADV>*<ADJ><ADP><NOUN><ADP><PROPN>}
                {<ADV><CCONJ><ADV>*<ADJ><ADP><ADJ>*<NOUN><ADP><NOUN>}
                {<ADV><CCONJ><ADV>*<ADJ><ADP><DET>*<NOUN><ADJ>*<ADP><DET>*<NOUN>}
                {<ADV><CCONJ><ADV>*<ADJ><ADP><ADV><CCONJ><ADV>*<ADJ>}
          """

cp = nltk.RegexpParser(grammar)

# Saving the np to a file
f = open('posession', 'w')

for tag in pos_tag:
    try:
        result = cp.parse(tag)
    except:
        pass

    for i in result.subtrees():
        if(i.label() == 'PS'):
            f.write(str(i) + '\n')

f.close()

final = time.time()
print('Time taken: ', final - inicio, "seconds")
