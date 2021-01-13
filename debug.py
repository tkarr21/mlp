import mlp as mlp
import arff

arff_path = r"./data/linsep2nonorigin.arff"
linsep2nonorigin = arff.Arff(arff=arff_path, label_count=1)
data = linsep2nonorigin.data[:,0:-1]
labels = linsep2nonorigin.data[:, -1].reshape(-1, 1)

MLPClass = mlp.MLPClassifier([4], lr=0.1 ,momentum=0.5, shuffle=False, deterministic=10)  #, deterministic=10
MLPClass.fit(data, labels)

print(MLPClass.get_weights())

