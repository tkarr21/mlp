import mlp as mlp
import arff
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # to run matplotlib on Macos
from matplotlib import pyplot as plt


arff_path = r"./data/iris_plants_database.arff"
iris = arff.Arff(arff=arff_path, label_count=1)
data = iris.data[:, 0:-1]
labels = iris.data[:, -1].reshape(-1, 1)

MLPClass = mlp.MLPClassifier([8], lr=0.1, shuffle=True, validation_size=.25)
MLPClass.fit(data, labels, initial_weights=True)

print('DONE TRAIN')

plt.figure()
#plt.style.use("fivethirtyeight")
plt.plot(MLPClass.learn_log[3][:- MLPClass.toleration], label='TS MSE')
plt.plot(MLPClass.learn_log[1][:- MLPClass.toleration], label='VS MSE')
plt.plot(MLPClass.learn_log[0][:- MLPClass.toleration], label='VS Accuracy %')
plt.ylabel("MSE and Class Accuracy")
plt.xlabel("Number of Epochs")
plt.title("BackProp: Iris data set")
plt.legend()
plt.show()
