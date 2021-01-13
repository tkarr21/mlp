def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import arff
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


arff_path = r"./data/vowel_data.arff"
vowel = arff.Arff(arff=arff_path, label_count=1)
vowel_data = vowel.data[:, 3:-1]
vowel_labels = vowel.data[:, -1]

V_train, V_test, v_train, v_test = train_test_split(vowel_data, vowel_labels)

arff_path = r"./data/iris_plants_database.arff"
iris = arff.Arff(arff=arff_path, label_count=1)
iris_data = iris.data[:, 0:-1]
iris_labels = iris.data[:, -1]
I_train, I_test, i_train, i_test = train_test_split(iris_data, iris_labels)


digits, dig_target = load_digits(return_X_y=True)

D_train, D_test, d_train, d_test = train_test_split(digits, dig_target)

clf = MLPClassifier(activation='tanh', alpha=1e-05, max_iter=1000,
              early_stopping=False,
              hidden_layer_sizes=(32),
              learning_rate='constant', learning_rate_init=0.6,
              momentum=0.7, 
              nesterovs_momentum=False, 
              )


clf1 = MLPClassifier(activation='tanh', alpha=1e-07, max_iter=1000,
              early_stopping=False,
              hidden_layer_sizes=(5, 14),
              learning_rate='constant', learning_rate_init=0.001,
              momentum=0.5, 
              nesterovs_momentum=True, 
              )



clf2 = MLPClassifier(activation='identity', max_iter=1000,
              alpha=1e-02, 
              early_stopping=False,
              hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.2,
              momentum=0.9, 
              nesterovs_momentum=True, 
              )


clf.fit(V_train, v_train)
clf1.fit(V_train, v_train)
clf2.fit(V_train, v_train)
pred = clf.predict(V_test)
pred1 = clf1.predict(V_test)
pred2 = clf2.predict(V_test)
print("VOWEL")
print(accuracy_score(v_test, pred))
print(accuracy_score(v_test, pred1))
print(accuracy_score(v_test, pred2))


clf.fit(I_train, i_train)
clf1.fit(I_train, i_train)
clf2.fit(I_train, i_train)
pred = clf.predict(I_test)
pred1 = clf1.predict(I_test)
pred2 = clf2.predict(I_test)
print("IRIS")
print(accuracy_score(i_test, pred))
print(accuracy_score(i_test, pred1))
print(accuracy_score(i_test, pred2))


clf.fit(D_train, d_train)
clf1.fit(D_train, d_train)
clf2.fit(D_train, d_train)
pred = clf.predict(D_test)
pred1 = clf1.predict(D_test)
pred2 = clf2.predict(D_test)
print("DIGIT")
print(accuracy_score(d_test, pred))
print(accuracy_score(d_test, pred1))
print(accuracy_score(d_test, pred2))
