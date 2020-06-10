"""
Once the aspects are chosen we are going to calculate the polarity
value of each sentence that contains each aspect and we are going
to display the results.

The chosen aspects are:
- disco
- canción
- grupo
- voz
- músico
- guitarra
- sonido
"""
import pickle
from prettytable import PrettyTable
from nltk.tokenize import word_tokenize, sent_tokenize

path_to_polarity_dictionary = "/home/ricardo/Documentos/Materias/NLP/Detección de Polaridad por Aspectos/pols"
path_to_sentences = "/home/ricardo/Documentos/Materias/NLP/Detección de Polaridad por Aspectos/sentences"
aspects = ('disco', 'canción', 'grupo', 'voz', 'músico', 'guitarra', 'sonido')

def load_info(path):
    f = open(path, 'rb')
    sentences = pickle.load(f)
    f.close()

    return sentences

def search_aspect(sentences, aspect):
    selected_sentences = []
    for sentence in sentences:
        if(aspect in sentence):
            selected_sentences.append(sentence)

    return selected_sentences

def compute_polarity(polarity_dictionary, sentences):
    total_sum = 0
    total_words = 0

    for sentence in sentences:
        for word in word_tokenize(str(sentence)):
            try:
                total_sum = total_sum + polarity_dictionary[word]
                total_words = total_words + 1
            except:
                pass

    return total_sum / total_words

print("Getting the sentences...")
sentences = sent_tokenize(load_info(path_to_sentences))

print("Getting the polarity dictionary...")
polarity_dictionary = load_info(path_to_polarity_dictionary)

print("Computing polarities...")
aspect_polarities = []
for aspect in aspects:
    aspect_pol = 0
    hit_sentences = search_aspect(sentences, aspect)
    aspect_pol = aspect_pol + compute_polarity(polarity_dictionary, hit_sentences)

    aspect_polarities.append(aspect_pol)

t = PrettyTable(['Aspect', 'Polarity'])
for i in range(len(aspects)):
    t.add_row([aspects[i], aspect_polarities[i]])
print(t)
