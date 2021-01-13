import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.


class MLPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, hidden_layer_widths, lr=.1, momentum=0, shuffle=True, deterministic=None, validation_size=0.0):
        """ Initialize class with chosen hyperparameters.
        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        
        self.validation_size = validation_size
        self.deterministic = deterministic
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.num_in = X.shape[1]
        self.outclasses = np.unique(y)
        self.num_out = self.outclasses.size

        #print(f"features {self.num_in}")
        #print(f"out classes {self.num_out}")

        # put in parameter; we are gonna run
        # initialize weights no matter what
        self.initialize_weights(initial_weights)
        
        #print(f"hid {self.hid}")
        #print(f"out {self.out}")


        #split data into train, test, and validation sets
        if self.validation_size:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1)
        
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=self.validation_size, random_state=1)
        else:
            X_train = X
            y_train = y

        #for graphs
        self.learn_log = [[], [], [], [], [], []]  

        # stopping criterion
        self.bssf = [0, 100, [], []]  #hold best weights And VS MSE so far
        self.toleration = 10
        mistake = 0
        deter = 1
        self.epoch_count = 0

        out_change = np.zeros(self.out.shape)
        hid_change = np.zeros(self.hid.shape)

        # run the epochs
        while True:

            # shuffle before epoch
            if self.shuffle:
                X_train, y_train = self._shuffle_data(X_train, y_train)

            

            ''' BackProp Alg '''
            for i in range(X_train.shape[0]):
                # calc activations (forward pass)
                Input = np.append(X_train[i, :], 1)   #add the bias
                target = self.onehot(y_train[i, 0]) if self.num_out > 2 else y_train[i, 0]
                hid_net = np.matmul(self.hid, Input)
                hid_act = np.append(1 / (1 + np.exp(-1 * hid_net)), 1)  #add bias

                #print(f"inp {Input}")
                #print(f"tar {target}")
                #print(f"hid {self.hid}")
                #print(f"hid net {hid_net}")
                #print(f"hid act {hid_act}")

                out_net = np.matmul(self.out, hid_act)
                out_act = 1 / (1 + np.exp(-1 * out_net))
                
                #print(f"out {self.out}")
                #print(f"out net {out_net}")
                #print(f"out act {out_act}")

                # calc error/delta (backward pass)
                out_error = (target - out_act)
                out_delta = out_error * out_act * (1 - out_act)


                hid_delta = np.zeros(hid_act.shape[0] - 1)
                for j in range(hid_delta.shape[0]):
                    delta_sum = 0
                    for k in range(out_delta.shape[0]):
                        delta_sum += out_delta[k] * self.out[k][j]

                    hid_delta[j] = hid_act[j] * (1 - hid_act[j]) * delta_sum

                #print(f"out err {out_error}")
                #print(f"out del {out_delta}")
                #print(f"hid del {hid_delta}")


                # calc change
                
                for j in range(out_delta.shape[0]):
                    for k in range(hid_act.shape[0]):
                        if self.momentum:
                            out_change[j][k] = self.lr * out_delta[j] * hid_act[k] + self.momentum * out_change[j][k]
                        else:
                            out_change[j][k] = self.lr * out_delta[j] * hid_act[k]

                
                for j in range(hid_delta.shape[0]):
                    for k in range(Input.shape[0]):
                        if self.momentum:
                            #print(f"lr ({self.lr}) * hid_delta{j} ({hid_delta[j]}) * input{k} ({Input[k]}) + m ({self.momentum}) * hid_change{j}{k} ({hid_change[j][k]})")
                            hid_change[j][k] = self.lr * hid_delta[j] * Input[k] + self.momentum * hid_change[j][k]
                            #print(f"hid_change{j}{k} ({hid_change[j][k]})")
                        else:
                            hid_change[j][k] = self.lr * hid_delta[j] * Input[k]

                #print(f"hid cha {hid_change}")
                #print(f"out cha {out_change}")
                #print(f"END INPUT LOOP")

                # update
                self.out = self.out + out_change
                self.hid = self.hid + hid_change

                #FIXME debug
                #if i == 2: break

            

            # logic for when to stop training
            if self.deterministic != None:
                if deter < self.deterministic:
                    deter += 1
                else:
                    break
            else:
                # score the model
                val_acc, val_mse = self.score(X_val, y_val)
                train_acc, train_mse = self.score(X_train, y_train)
                test_acc, test_mse = self.score(X_test, y_test)

                # log the results
                self.learn_log[0].append(val_acc)
                self.learn_log[1].append(val_mse)
                self.learn_log[2].append(train_acc)
                self.learn_log[3].append(train_mse)
                self.learn_log[4].append(test_acc)
                self.learn_log[5].append(test_mse)

                # stopping criteria
                if val_mse > self.bssf[1]:
                    mistake += 1
                    if mistake > self.toleration:
                        print(f"NUM EPOCHS: {self.epoch_count}")
                        break
                else:
                    mistake = 0
                    self.bssf[0] = val_acc
                    self.bssf[1] = val_mse
                    self.bssf[2] = self.hid.copy()
                    self.bssf[3] = self.out.copy()

            #FIXME debug
            #break
            self.epoch_count += 1
   
        self.epoch_count -= self.toleration
        return self

    
    def onehot(self, value):
        """ One hot encode output 
        Args:
            value: an output value that is an element of self.outclasses
        Returns:
            array, shape (n_classes,)
                array of zeros except for the index of the class given by value
        """
        index = np.where(self.outclasses == value)
        tars = np.zeros((self.num_out))
        tars[index] = 1

        return tars


    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pred_shape = (X.shape[0], self.num_out) if self.num_out > 2 else (X.shape[0], 1)
        predictions = np.zeros(pred_shape)

        for i in range(X.shape[0]):
            # calc activations (forward pass)
            Input = np.append(X[i, :], 1)   #add the bias
            
            hid_net = np.matmul(self.hid, Input)
            hid_act = np.append(1 / (1 + np.exp(-1 * hid_net)), 1)  #add bias


            out_net = np.matmul(self.out, hid_act)
            out_act = 1 / (1 + np.exp(-1 * out_net))
            predictions[i] = out_act

            #FIXME
            #if i == 2: break


        return predictions

            

    def initialize_weights(self, initial_weights):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """

        #shapes for the weight vectors
        hidtup = (self.hidden_layer_widths[0], self.num_in + 1)
        outtup = (self.num_out, self.hidden_layer_widths[0] + 1) if self.num_out > 2 else (1, self.hidden_layer_widths[0] + 1)

        if not initial_weights:
            #print("inited weights!")
            self.hid = np.zeros(hidtup)
            self.out = np.zeros(outtup)
        else:
            np.random.seed(seed=5)
            self.hid = np.random.normal(loc=0, scale=1.0, size=hidtup)
            self.out = np.random.normal(loc=0, scale=1.0, size=outtup)
           
        return 

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        predicts = self.predict(X)
        #print(f"PREDICTS SHAPE {predicts.shape}")
        hits = 0
        sse = 0
        target = []

        for i in range(predicts.shape[0]):
            target = self.onehot(y[i, 0]) if self.num_out > 2 else y[i, 0]
            sse = sse + np.sum(np.square(target - predicts[i]))

            if self.num_out > 2:
                #print(f"predicts{i} index {np.argmax(predicts[i])}")
                #print(f"targets index {np.argmax(target)}")
                if np.argmax(predicts[i]) == np.argmax(target):
                    hits += 1
            else:
                if predicts[i] < .5:
                    pred_class = 0
                else:
                    pred_class = 1

                if pred_class == target:
                    hits += 1

            #FIXME
            #if i == 2: break

        
        #print(f"HITS: {hits}")


        return hits / predicts.shape[0], sse/(predicts.shape[0] * target.shape[0])
        

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        return shuffle(X, y, random_state=0)

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        all_weights = []

        for i in range(self.out.shape[0]):
            for j in range(self.out.shape[1]):
                all_weights.append(self.out[i][j])
        for i in range(self.hid.shape[0]):
            for j in range(self.hid.shape[1]):
                all_weights.append(self.hid[i][j])

        return all_weights

if __name__ == "__main__":
    mlp = MLPClassifier([5,3])
    
    hid_weights = mlp.initialize_weights()
    print(hid_weights)
    print(np.mean(hid_weights[0]))
    print(np.mean(hid_weights[1]))