import mlp as mlp
import arff
import csv

arff_path = r"./data/data_banknote_authentication.arff"
bank_note = arff.Arff(arff=arff_path, label_count=1)
data = bank_note.data[:, 0:-1]
labels = bank_note.data[:, -1].reshape(-1, 1)

MLPClass = mlp.MLPClassifier([8], lr=0.1, momentum=0.5, shuffle=False, deterministic=10)
MLPClass.fit(data, labels)
weights = MLPClass.get_weights()

file_path = './evaluation.csv'
fd = open(file_path, "w+")

for weight in weights:
    fd.write(str(weight) + '\n')

fd.close()
