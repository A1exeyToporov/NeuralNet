from numpy import exp


def relu(Z):
    Z[Z <= 0] = 0
    return Z


def sigmoid(Z):
    return 1 / (1 + exp(-Z))