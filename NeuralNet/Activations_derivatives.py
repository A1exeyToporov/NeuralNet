from numpy import array, exp

def relu_derivative(dA, linear_cache):
    Z = linear_cache
    dZ = array(dA, copy=True)
    dZ[Z < 0] = 0
    return dZ


def sigmoid_derivative(dA, linear_cache):
    Z = linear_cache
    s = 1 / (1 + exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ