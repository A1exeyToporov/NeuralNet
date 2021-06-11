from numpy import sum, squeeze, log, mean
def binary_cost(A, Y):
    Y = Y.reshape(A.shape)
    m = Y.shape[1]
    cost = -sum(Y * log(A) + (1 - Y) * log(1 - A), axis=1, keepdims=True) / m
    cost = squeeze(cost)
    return cost


def MSE_cost(A, Y):
    Y = Y.reshape(A.shape)
    # return np.mean(np.square(A - Y))
    return mean(abs(Y - A))