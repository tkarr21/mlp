import mlp as mlp
import arff
import numpy as np
import matplotlib
matplotlib.use("TkAgg") # to run matplotlib on Macos
from matplotlib import pyplot as plt

arff_path = r"./data/vowel_data.arff"
vowel = arff.Arff(arff=arff_path, label_count=1)
data = vowel.data[:, 3:-1]
labels = vowel.data[:, -1].reshape(-1, 1)

#print(data.shape[1])
#print(data[1])

'''MLPClass = mlp.MLPClassifier([20], lr=0.1, shuffle=True, validation_size=.25)
MLPClass.fit(data, labels, initial_weights=True)
print(MLPClass.learn_log[0][-1])

plt.figure()
#plt.style.use("fivethirtyeight")
#plt.plot(MLPClass.learn_log[0][:- MLPClass.toleration], label='VS ACC')
plt.plot(MLPClass.learn_log[3][:- MLPClass.toleration], label='TS MSE')
plt.plot(MLPClass.learn_log[1][:- MLPClass.toleration], label='VS MSE')
plt.plot(MLPClass.learn_log[5][:- MLPClass.toleration], label='TestS MSE')
plt.ylabel("MSE")
plt.xlabel("Number of Epochs")
plt.title("BackProp: Vowel data set LR 0.1")
plt.legend()
plt.show()


MLP2 = mlp.MLPClassifier([20], lr=0.3, shuffle=True, validation_size=.25)
MLP2.fit(data, labels, initial_weights=True)


plt.figure()
#plt.style.use("fivethirtyeight")
plt.plot(MLP2.learn_log[3][:- MLP2.toleration], label='TS MSE')
plt.plot(MLP2.learn_log[1][:- MLP2.toleration], label='VS MSE')
plt.plot(MLP2.learn_log[5][:- MLP2.toleration], label='TestS MSE')
plt.ylabel("MSE")
plt.xlabel("Number of Epochs")
plt.title("BackProp: Vowel data set LR 0.3")
plt.legend()
plt.show()


MLP3 = mlp.MLPClassifier([20], lr=0.5, shuffle=True, validation_size=.25)
MLP3.fit(data, labels, initial_weights=True)


plt.figure()
#plt.style.use("fivethirtyeight")
plt.plot(MLP3.learn_log[3][:- MLP3.toleration], label='TS MSE')
plt.plot(MLP3.learn_log[1][:- MLP3.toleration], label='VS MSE')
plt.plot(MLP3.learn_log[5][:- MLP3.toleration], label='TestS MSE')
plt.ylabel("MSE")
plt.xlabel("Number of Epochs")
plt.title("BackProp: Vowel data set LR 0.5")
plt.legend()
plt.show()



MLP4 = mlp.MLPClassifier([20], lr=0.9, shuffle=True, validation_size=.25)
MLP4.fit(data, labels, initial_weights=True)

plt.figure()
#plt.style.use("fivethirtyeight")
plt.plot(MLP4.learn_log[3][:- MLP4.toleration], label='TS MSE')
plt.plot(MLP4.learn_log[1][:- MLP4.toleration], label='VS MSE')
plt.plot(MLP4.learn_log[5][:- MLP4.toleration], label='TestS MSE')
plt.ylabel("MSE")
plt.xlabel("Number of Epochs")
plt.title("BackProp: Vowel data set LR 0.9")
plt.legend()
plt.show()
'''



'''
objects = ('0.1', '0.3', '0.5', '0.9', '0.95')
y_pos = np.arange(len(objects))
performance = [178, 102, 54, 52, 31]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('Learning Rate')
plt.ylabel('Epochs')
plt.title('Epochs for different Learning Rates')

plt.show()
'''

'''MLP5 = mlp.MLPClassifier([1], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("1")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])


MLP5 = mlp.MLPClassifier([2], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("2")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])


MLP5 = mlp.MLPClassifier([4], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("4")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])

MLP5 = mlp.MLPClassifier([8], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("8")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])


MLP5 = mlp.MLPClassifier([16], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("16")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])'''

'''MLP5 = mlp.MLPClassifier([32], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("32")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])'''
'''
MLP5 = mlp.MLPClassifier([64], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("64")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])

MLP5 = mlp.MLPClassifier([128], lr=0.9, shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print("128")
print(MLP5.learn_log[1][-MLP5.toleration])
print(MLP5.learn_log[3][-MLP5.toleration])
print(MLP5.learn_log[5][-MLP5.toleration])
'''

'''
plt.figure()
plt.plot([1,2,4,8,16,32,64,128], [0.07465, 0.07390, 0.05476, 0.04863, 0.03343, 0.01895, 0.02236, 0.02908], label='TS MSE')
plt.plot([1,2,4,8,16,32,64,128], [0.07534, 0.07581, 0.06237, 0.05542, 0.04188, 0.03276, 0.03142, 0.03569], label='VS MSE')
plt.plot([1,2,4,8,16,32,64,128], [0.07655, 0.07616, 0.06115, 0.05478, 0.03906, 0.03235, 0.02998, 0.03615], label='TestS MSE')
plt.ylabel("MSE")
plt.xlabel("Number of Nodes")
plt.title("BackProp: Vowel data set Hidden Nodes")
plt.legend()
plt.show()
'''

'''MLP5 = mlp.MLPClassifier([64], lr=0.9, momentum=.5 ,shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print(MLP5.epoch_count)

MLP5 = mlp.MLPClassifier([64], lr=0.9, momentum=.6 ,shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print(MLP5.epoch_count)

MLP5 = mlp.MLPClassifier([64], lr=0.9, momentum=.7 ,shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print(MLP5.epoch_count)

MLP5 = mlp.MLPClassifier([64], lr=0.9, momentum=.8 ,shuffle=True, validation_size=.25)
MLP5.fit(data, labels, initial_weights=True)
print(MLP5.epoch_count)
'''

plt.figure()
plt.plot([.5, .6, .7, .8], [35, 25, 10, 15])
plt.ylabel("Epochs")
plt.xlabel("Momentum")
plt.title("BackProp: Vowel data set Momentum")
plt.legend()
plt.show()