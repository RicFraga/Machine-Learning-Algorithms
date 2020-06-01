import os
import pickle
import fnmatch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to where the dataset is contained
path_to_data = "/home/fraga/Escritorio/corpusCine/corpusCriticasCine/"

rank_format = ".xml"

# How many .xml files are contained?
amount_of_ranks = len(fnmatch.filter(os.listdir(path_to_data), '*.xml'))

print("Amount of *.xml files", amount_of_ranks)

missing_files = 0
Y = np.zeros(amount_of_ranks, dtype = np.uint8)
index = 0

for i in range(1, amount_of_ranks):
    try:
        # Building the path to the file
        a = open(path_to_data + '/' + str(i) + rank_format, "r", encoding = "ISO-8859-1")

        # Getting the content and forming a single string with it
        content = a.readlines()
        content = "".join(content)
        
        # Finding the position of rank and moving 6 index forward to get the value (every review follows this structure)
        position_of_rank = content.find("rank")
        rank = np.uint8(content[position_of_rank + 6])
        
        Y[index] = rank
        index += 1
        
        a.close()
        
    except:
        missing_files += 1
        #print("No file named", str(i) + rank_format)
        continue

print("There are", missing_files, "missing files")
print("Possible ranks:", set(Y))

yfile = open('yfile', 'ab') 

pickle.dump(Y, yfile)                      
yfile.close()

pos_format = ".review.pos"

# How many .pos files are contained?
amount_of_pos = len(fnmatch.filter(os.listdir(path_to_data), '*.pos'))

print("Amount of *.pos files", amount_of_pos)

global_voc = []
local_voc = []
i = 1

while(i < amount_of_pos):
    try:
        # Building the path to the file
        a = open(path_to_data + str(i) + pos_format, "r", encoding = "ISO-8859-1")

        # Getting the content and forming a single string with it
        content = a.readlines()
        
        aux = []
        
        # Getting the lemmatized word of each line (this word is in the second position of each line)
        for j in range(len(content)):
            try:
                global_voc.append(content[j].split(' ')[1])
                aux.append(content[j].split(' ')[1])
            except:
                continue
        
        local_voc.append(aux)
        i += 1
        a.close()
        
    except:
        i += 1

print("Vocabulary length:", len(local_voc))

vectorizer = TfidfVectorizer()

# Tokenizing and building vocab
vectorizer.fit(global_voc)

space = " "
vectors = []

for i in range(len(local_voc)):
    vector = vectorizer.transform([space.join(local_voc[i])])
    vectors.append(vector.toarray().reshape(45472, ))

X = np.asarray(vectors)
print("Final shape:", X.shape)

xfile = open('xfile', 'ab')

pickle.dump(X, xfile)                
xfile.close()