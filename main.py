import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import NeuralNet as NNet

if __name__ == '__main__':
    data = pd.read_csv('datasets/data_banknote_authentication.txt', sep=',', header=None)
    X = data.values[:,0:4]
    Y = data.values[:,4]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    NN = NNet.NeuralNetwork(X_train.shape[1])
    NN.add_layer(20, activation='relu')
    NN.add_layer(7, activation='relu')
    NN.add_layer(5, activation='relu')
    NN.add_layer(1, activation='sigmoid')
    NN.fit(X_train, y_train, num_iterations=2000, learning_rate=0.008, loss='binary')
    A = NN.predict(X_test)
    print (NN.accuracy(y_test, A))
    NN.accuracy()
