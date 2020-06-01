import pickle
import mord as md
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix

alpha = 0.001

print("Loading y...")
yfile = open('yfile', 'rb')      
Y = pickle.load(yfile)
yfile.close()

print("Loading x...")
xfile = open('xfile', 'rb')
X = pickle.load(xfile)
xfile.close()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 1)

#model = md.LogisticIT(alpha = alpha)
model = md.LogisticSE(alpha = alpha)

print("Alpha:", alpha)

print("Training model...")
model.fit(X_train, y_train)

print("Saving model")
model_file = open('model_file', 'ab')
pickle.dump(model, model_file)
model_file.close()

print("Making predictions")
#predictions = model.predict(X_train)
predictions = model.predict(X_test)

print("Computing precission")
#print("Precision:", precision_score(y_train, predictions, average = 'macro', zero_division = 1))
print("Precision:", precision_score(y_test, predictions, average = 'macro', zero_division = 1))

print("Computing recall")
#print("Recall:", recall_score(y_train, predictions, average = 'macro', zero_division = 1))
print("Recall:", recall_score(y_test, predictions, average = 'macro', zero_division = 1))

print("Computing F1")
#print("F1:", f1_score(y_train, predictions, average = 'macro', zero_division = 1))
print("F1:", f1_score(y_test, predictions, average = 'macro', zero_division = 1))

print("Computing confusion matrix")
c_m = confusion_matrix(y_test, predictions, labels = [1, 2, 3, 4, 5])
print(c_m)