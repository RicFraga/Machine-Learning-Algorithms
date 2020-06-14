import pickle
from collections import Counter
from prettytable import PrettyTable
from nltk.tokenize import sent_tokenize, word_tokenize

path_to_yes_sentences = "/home/ricardo/Escritorio/Fancier Aspects Polarity/yes_sentences"
path_to_no_sentences = "/home/ricardo/Escritorio/Fancier Aspects Polarity/no_sentences"
path_to_polarity_dictionary = "/home/ricardo/Documentos/Materias/NLP/pols"
aspects = ('disco', 'canción', 'grupo', 'voz', 'músico', 'guitarra', 'sonido')
NUMBER_OF_ELEMENTS = 5

# Function to load data
def load_info(path):
    f = open(path, 'rb')
    sentences = pickle.load(f)
    f.close()

    return sentences

# This function takes a word(aspect) and searches it in sentences
def search_aspect(sentences, aspect):
    selected_sentences = []
    for sentence in sentences:
        if(aspect in sentence):
            selected_sentences.append(sentence)

    return selected_sentences

# This functions uses a pol dictionary to compute polarity values
def compute_polarity(polarity_dictionary, sentences):
    total_positive = 0
    total_negative = 0

    for sentence in sentences:
        for word in word_tokenize(str(sentence)):
            try:
                if(polarity_dictionary[word] >= 0):
                    total_positive += 1

                else:
                    total_negative += 1

            except:
                pass

    #print("Total positive {}".format(total_positive))
    #print("Total negative {}".format(total_negative))

    if(total_positive > total_negative):
        polarity = 'positive'

    elif(total_negative > total_positive):
        polarity = 'negative'

    else:
        polarity = 'neutral'

    return polarity

def search_word_in_sentences(word, sentences):
    sentences = word_tokenize(' '.join(sentences))

    return sentences.count(word)

print("Loading the preprocessed data...")
yes_sentences = sent_tokenize(load_info(path_to_yes_sentences))
no_sentences = sent_tokenize(load_info(path_to_no_sentences))

print("Getting the polarity dictionary...")
polarity_dictionary = load_info(path_to_polarity_dictionary)
dic_keys = polarity_dictionary.keys()

t = PrettyTable(['Aspect',
                 'PIY',
                 'PIN',
                 'T5PY',
                 'T5NY',
                 'T5PN',
                 'T5NN'])

for aspect in aspects:
    print("Getting info of {}...". format(aspect))

    hit_yes = search_aspect(yes_sentences, aspect)
    polarity_yes = compute_polarity(polarity_dictionary, hit_yes)

    hit_no = search_aspect(no_sentences, aspect)
    polarity_no = compute_polarity(polarity_dictionary, hit_no)

    count_yes = Counter(word_tokenize(' '.join(hit_yes)))
    count_no = Counter(word_tokenize(' '.join(hit_no)))

    max_pos_yes = []
    max_neg_yes = []

    max_pos_no = []
    max_neg_no = []

    for word, count in count_yes.most_common(100):
        try:
            if(len(max_pos_yes) != NUMBER_OF_ELEMENTS):
                if(polarity_dictionary[word] >= 0):
                    max_pos_yes.append(word)

            if(len(max_neg_yes) != NUMBER_OF_ELEMENTS):
                if(polarity_dictionary[word] < 0):
                    max_neg_yes.append(word)

            if(len(max_pos_yes) == NUMBER_OF_ELEMENTS and len(max_neg_yes) == NUMBER_OF_ELEMENTS):
                break

        except:
            pass

    for word, count in count_no.most_common(100):
        try:
            if(len(max_pos_no) != NUMBER_OF_ELEMENTS):
                if(polarity_dictionary[word] >= 0):
                    max_pos_no.append(word)

            if(len(max_neg_no) != NUMBER_OF_ELEMENTS):
                if(polarity_dictionary[word] < 0):
                    max_neg_no.append(word)

            if(len(max_pos_no) == NUMBER_OF_ELEMENTS and len(max_neg_no) == NUMBER_OF_ELEMENTS):
                break

        except:
            pass

    t.add_row([aspect, polarity_yes, polarity_no, '\n'.join(max_pos_yes),
              '\n'.join(max_neg_yes), '\n'.join(max_pos_no),
              '\n'.join(max_neg_no)])

print(t)
